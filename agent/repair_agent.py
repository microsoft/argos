# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import time

from agent.agent import LLM, TIMEOUT, TIMEOUT_INFERENCE, Agent
from common.common import cleanup_global_env, format_check, run_with_timeout
from common.exception import RuntimeException, SyntaxException


class RepairAgent(Agent):
    def __init__(self, chunk_size, llm_engine="gpt-4o", timeout=150) -> None:
        self.chunk_size = chunk_size
        repair_agent_prompt = f"""
You are an AI assistant that fixs syntax and runtime erros in Python code. You will be given a Python code snippet along with the error message indicating details for syntax and/or runtime errors. You should fix the errors in the code and make sure the code can be executed without any error. 

You should achieve the task in the following steps:

1. You are given a python function `inference(sample: np.ndarray) -> labels: np.ndarray` to write various rules to describe and remember the pattern of given negative/abnormal samples and exclude all given positive/normal samples. The function will take a sample of numpy array with shape ({self.chunk_size}, 2) as input, where each row is a tuple of (value, index). The function will determine whether the given sample has a similar pattern as previous negative/abnormal or positive/normal samples. The function will return the labels as an np.ndarray of shape ({self.chunk_size}), and for each index, value=1 means the data of the index is abnormal, and value=0 means the data of the index is normal. The code will be given in the following format:
##### CODE
```python
# import necessary libraries
def inference(sample: np.ndarray) -> np.ndarray:
    # Comment to describe how normal data behave
    # Normal Rule 1
    # Normal Rule 2
    # Code to detect if the given sample is abnormal
    # Abnormal Rule 1
    if ...
    # Abnormal Rule 2
    if ...
    # return labels as a 1d numpy array indicating abnormal/normal of each index
```
2. IMPORTANT: You should output the fixed code following the same format as the input code, wrapping the code with ```python as the first line and ``` as the last line. You must only use ```python and ``` to wrap your fixed code for only once, don't use them for any other purpose.
3. You should only focus on fixing the errors in the code and make sure the code can be executed without any error. You must not change other logics of the code unrelated to the errors.
        """.strip()
        self.LLM = LLM(
            system_prompt=repair_agent_prompt.strip(),
            temperature=0.75,
            past_message_num=10,
            engine=llm_engine,
        )
        self.name = "RepairAgent"
        self.max_time = timeout * 60
        print(
            f"[RepairAgent] Initialized with chunk_size={chunk_size}, llm_engine={llm_engine}, timeout={timeout}"
        )

    def run(self, curr_df, curr_rule_path) -> None:
        logging.info(f"[RepairAgent] Start to repair the code in {curr_rule_path}")

        # drop label column if there is label column
        curr_df = curr_df.copy()
        if "label" in curr_df.columns:
            curr_df.drop(columns=["label"], inplace=True)

        start_time = time.time()

        while self.get_elapsed_time() < self.max_time:

            error_message = None

            # read code into str
            with open(curr_rule_path, "r") as f:
                rule = f.read()

            # print(inference(current_data.values).shape)
            try:
                exec(rule, globals())
                # execute_and_cleanup(rule)
                labels = run_with_timeout(inference, TIMEOUT_INFERENCE, curr_df.values)
                # labels = inference(curr_df.values)
                cleanup_global_env()
                format_check(curr_df, curr_rule_path, labels)
            except Exception as e:
                cleanup_global_env()
                error_message = str(e)

            if not error_message:
                logging.info(
                    f"[RepairAgent] The code in {curr_rule_path} has no error."
                )
                break
            else:
                logging.info(
                    f"[RepairAgent] The code in {curr_rule_path} has error: {error_message}"
                )
            # current_data_str = curr_df.to_string(index=False, header=False)

            final_query = "##### CODE" + rule
            final_query += (
                "\n"
                + "##### ERROR FROM EXECUTING CODE, PLEASE FIX IT\n"
                + error_message
            )

            ans = self.LLM.query(final_query)
            self.LLM.reset()

            # if inference not in answer, then we assume it fails to generate code, and directly retry
            try:
                logging.info(f"[RepairAgent] Extract code from LLM: {ans}")
                code = self.extract_code(ans)
            except Exception as e:
                logging.info(
                    f"[RepairAgent] LLM did not generate a function with correct format, retry now: {ans}"
                )
                continue
            self.save_rule(code, curr_rule_path)

        if self.get_elapsed_time() >= self.max_time:
            logging.info(
                f"[RepairAgent] Time out to repair the code in {curr_rule_path}"
            )
            return
