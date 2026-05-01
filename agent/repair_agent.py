# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import time

from agent.agent import LLM, TIMEOUT, TIMEOUT_INFERENCE, Agent
from agent.prompts.repair import build_repair_agent_prompt
from common.common import cleanup_global_env, format_check, run_with_timeout
from common.exception import RuntimeException, SyntaxException


class RepairAgent(Agent):
    def __init__(self, chunk_size, llm_engine="gpt-4o", timeout=150) -> None:
        self.chunk_size = chunk_size
        repair_agent_prompt = build_repair_agent_prompt(self.chunk_size)
        self.LLM = LLM(
            system_prompt=repair_agent_prompt,
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

        while self.get_elapsed_time() < self.max_time:

            error_message = None

            # read code into str
            with open(curr_rule_path, "r") as f:
                rule = f.read()

            # print(inference(current_data.values).shape)
            try:
                local_env = {}
                exec(rule, local_env)
                # execute_and_cleanup(rule)
                inference_fn = local_env["inference"]
                labels = run_with_timeout(inference_fn, TIMEOUT_INFERENCE, curr_df.values)
                # labels = inference(curr_df.values)
                # cleanup_global_env()
                format_check(curr_df, curr_rule_path, labels)
            except Exception as e:
                # cleanup_global_env()
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

            try:
                ans = self.LLM.query(final_query)
            except TimeoutError as e:
                continue
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
