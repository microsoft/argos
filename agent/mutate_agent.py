import logging
import os
import random
import threading
import time
from typing import List
from agent.agent import LLM, Agent
from agent.prompts.mutate import generate_mutate_prompt

MAX_ITER_TRY = 3

probaility_mutate_rules={
    "remove(s)": 0.30, # remove rule (otherwise llm just keep adding rule) 0.05
    "free": 0.30, # free to mutate -> default                              0.84
    "same": 0.0, # keep the same                                          0.8
    "change-only": 0.30, # change the rule                                0.99
    "remove-all-but-one": 0.10 # remove all but one rule                  1.0
}


class MutateAgent(Agent):
    def __init__(self, chunk_size, llm_engine="gpt-4o", timeout=150) -> None:
        self.chunk_size = chunk_size
        mutate_agent_prompt = generate_mutate_prompt(chunk_size)
        self.LLM = LLM(
            system_prompt=mutate_agent_prompt.strip(),
            temperature=0.75,
            past_message_num=10,
            engine=llm_engine,
        )
        self.name = "MutateAgent"
        self.max_time = timeout * 60
        print(
            f"[MutateAgent] Initialized with chunk_size={chunk_size}, llm_engine={llm_engine}, timeout={timeout}"
        )

    def extract_code(self, text, lang = "python", function_name = "def inference"):
        return super().extract_code(text, lang, function_name)
    
    def select_mutate_rule_with_probability(self):
        rules = list(probaility_mutate_rules.keys())
        probailities = list(probaility_mutate_rules.values())
        return random.choices(rules, probailities)[0]
    
    def populate_missing_mutated_item(self, curr_rules_path: List[str], k = 1):
        # randomly select a rule from the list
        if not curr_rules_path:
            print("[MutateAgent] No rules to process.")
            return
        
        random_index = random.randint(0, len(curr_rules_path) - 1)
        curr_rule_path = curr_rules_path[random_index]

        with open(curr_rule_path, "r") as f:
            rule = f.read()
        parent_dir = os.path.dirname(curr_rule_path)

        files = os.listdir(parent_dir)

        mutated_file = [f for f in files if "mutated" in f and f.endswith(".py")]
        if len(mutated_file) < k:
            for i in range(k - len(mutated_file)):
                path_to_save = curr_rule_path.replace(".py", f"_mutated_{k - i - 1}.py")
                with open(path_to_save, "w") as f:
                    f.write(rule)

        print(f"[MutateAgent] Populated missing {k - len(mutated_file)} mutated items in {curr_rule_path}")
        return
    
    def normal_rule_protection(self, original_code, code):

        # find the normal rules
        normal_rules = []
        for line in original_code.split("\n"):
            if line.strip().startswith("# Normal Rule"):
                normal_rules.append(line)

        # find the new normal rules
        new_normal_rule = []
        for line in code.split("\n"):
            if line.strip().startswith("# Normal Rule"):
                new_normal_rule.append(line)

        # find the abnormal rules
        len_normal_rules = len(normal_rules)
        len_new_normal_rules = len(new_normal_rule)

        if len_new_normal_rules < len_normal_rules:
            return len_normal_rules - len_new_normal_rules

        return 0

    def run_mutation_thread(self, curr_rule_path, i):
        tries = 0
        while tries < MAX_ITER_TRY:
            try:
                path_to_save = curr_rule_path.replace(".py", f"_mutated_{i}.py")
                mutate_rule = self.select_mutate_rule_with_probability()
                new_system_prompt = generate_mutate_prompt(self.chunk_size, mutate_rule)
                print(f"[MutateAgent] Mutate rule: {mutate_rule}")

                if not hasattr(self, "_llm_lock"):
                    self._llm_lock = threading.Lock()

                with self._llm_lock:
                    self.LLM.reset_system_prompt(new_system_prompt)
                    self.generate_mutate_code(curr_rule_path, mutate_rule, path_to_save)
                break  # Successfully mutated, move to the next iteration
            except Exception as e:
                tries += 1
                print(f"[MutateAgent] Error during mutation attempt {tries}: {e}")
        else:
            print(f"[MutateAgent] Failed to mutate the code in {curr_rule_path} after {MAX_ITER_TRY} tries")
        


    def run(self, curr_rule_path, k=1) -> None:
        print(f"[MutateAgent] Start to mutate the code in {curr_rule_path}")
        threads = []
        for i in range(k):
            thread = threading.Thread(target=self.run_mutation_thread, args=(curr_rule_path, i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        print(f"[MutateAgent] Finished mutating the code in {curr_rule_path}")

    def run_with_top_pick(self, curr_rules_path: List[str], k=1, keep_one: bool=False) -> None:
        if not curr_rules_path:
            print("[MutateAgent] No rules to process.")
            return
        
        if len(curr_rules_path) > k:
            # downsample the rules to k
            curr_rules_path = random.sample(curr_rules_path, k)
        print(f"[MutateAgent] Start to mutate the code in {curr_rules_path}")

        mutate_per_pipeline = max(int(k / len(curr_rules_path)), 1)

        threads = []

        # create mutate_per_pipeline threads per rule
        for i, rule_path in enumerate(curr_rules_path):
            for j in range(mutate_per_pipeline):
                thread_index = i * mutate_per_pipeline + j

                if keep_one and j == 0:
                    path_to_save = rule_path.replace(".py", f"_mutated_{thread_index}.py")
                    with open(path_to_save, "w") as f:
                        with open(rule_path, "r") as original_file:
                            rule = original_file.read()
                            f.write(rule)
                    continue

                thread = threading.Thread(
                    target=self.run_mutation_thread,
                    args=(rule_path, thread_index)
                )
                threads.append(thread)
                thread.start()

        # handle remaining threads (if k is not divisible evenly)
        total_threads_created = len(curr_rules_path) * mutate_per_pipeline
        remaining = k - total_threads_created
        for i in range(remaining):
            rule_path = random.choice(curr_rules_path)
            thread_index = total_threads_created + i
            thread = threading.Thread(
                target=self.run_mutation_thread,
                args=(rule_path, thread_index)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        print(f"[MutateAgent] Finished mutating the code in {curr_rules_path}")

    def wrap_code(self, code):
        return f"```python\n{code}\n```"
    
    def generate_mutate_code(self, curr_rule_path, mutate_rule, path_to_save):

        if mutate_rule == "same":
            with open(curr_rule_path, "r") as f:
                rule = f.read()
            with open(path_to_save, "w") as f:
                f.write(rule)
            print(f"[MutateAgent] Keep the code in {path_to_save}")
            return
        
        start_time = time.time()

        while time.time() - start_time < self.max_time:

            error_message = None

            # read code into str
            with open(curr_rule_path, "r") as f:
                rule = f.read()

            try:
                self.LLM.reset()
                ans = self.LLM.query("\nThe function need to mutate is as below:\n" + self.wrap_code(rule))
                try:
                    code = self.extract_code(ans)
                    with open(path_to_save, "w") as f:
                        f.write(code)
                    f.close()
                except Exception as e:
                    raise Exception(f"Error in extracting code: {e}")
                print(f"[MutateAgent] Mutated code in {path_to_save}")
                break
            except Exception as e:
                error_message = f"Error in mutating the code: {e}"
                logging.error(error_message)
                if self.get_elapsed_time() < self.max_time:
                    print(
                        f"[MutateAgent] Restart to mutate the code in {curr_rule_path}"
                    )
                else:
                    print(
                        f"[MutateAgent] Timeout in mutating the code in {curr_rule_path}"
                    )
                    break
        end_time = time.time()
        logging.info(
            f"[MutateAgent] Finished mutating the code in {curr_rule_path} in {end_time-start_time} seconds with {mutate_rule}"
        )
