# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import os
import subprocess
from typing import Optional

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from agent.agent import (LLM, TIMEOUT, TIMEOUT_FIRST_REVIEW, TIMEOUT_INFERENCE,
                         Agent)
from agent.prompts.llmad import LLMAD_PROMPT
from common.common import (calculate_performance, combine_labels, format_check,
                           get_gt_labels, get_model_labels, get_model_scores,
                           get_rule_labels, preprocess_labels,
                           run_with_timeout, smooth_labels)
from common.exception import (RuntimeException, SyntaxException,
                              TimeoutException)
from eval_metrics.event_f1pa import EventF1PA
from eval_metrics.point_f1 import PointF1
from eval_metrics.point_f1pa import PointF1PA


class LLMAD(Agent):
    def __init__(self, chunk_size, mode="baseline-LLMAD", llm_engine="gpt-4o") -> None:
        self.chunk_size = chunk_size
        if mode == "baseline-LLMAD":
            llmad_prompt = LLMAD_PROMPT
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self.LLM = LLM(
            system_prompt=llmad_prompt.strip(),
            temperature=0.75,
            past_message_num=10,
            engine=llm_engine,
        )
        self.mode = mode

    def run(self, curr_df, curr_json_path) -> Optional[dict]:
        logging.info(f"[LLMAD] Start to detect the anomalies")

        while self.get_elapsed_time() < TIMEOUT:

            current_data_str = curr_df.to_string(index=False, header=False)

            # ##Data Please analyze the latest data with the highest level of diligence and caution: - Historical normal data sequence: {normal_data} - Historical anomaly data sequence(*XXX* is anomaly point), {anomaly_data} - The latest {data_len} data points for evaluation: {data}
            final_query = (
                "##Data Please analyze the latest data with the highest level of diligence and caution:\n"
                + "#####Data\n"
                + current_data_str
                + "\n"
            )

            # logging.info(f"[LLMAD] Query to LLM: {final_query}")

            ans = self.LLM.query(final_query)
            self.LLM.reset()

            # if inference not in answer, then we assume it fails to generate code, and directly retry
            # if "inference" in ans:
            #     logging.info(f"[DetectionAgentV3] Extract code from LLM: {ans}")
            #     code = self.extract_code(ans)
            #     if code == "":
            #         continue
            #     self.save_rule(code, curr_rule_path)
            #     break
            # else:
            #     logging.info(f"[DetectionAgentV3] LLM did not generate a function with name inference, retry now: {ans}")
            try:
                logging.info(f"[LLMAD] Extract json from LLM: {ans}")
                json_obj = self.extract_json(ans)
                self.json_format_check(json_obj, curr_df)
            except Exception as e:
                logging.info(f"[LLMAD] Exception raised, retrying: {e}")
                continue
            self.save_json(json_obj, curr_json_path)
            break

        if self.get_elapsed_time() >= TIMEOUT:
            logging.info(
                f"[LLMAD] Time out to detect the anomaly and save in {curr_json_path}"
            )

            return None
        return json_obj

    def extract_json(self, text: str) -> str:
        json_obj = json.loads(text)
        return json_obj

    def save_json(self, json_obj, json_path):
        with open(json_path, "w") as f:
            json.dump(json_obj, f)
        logging.info(f"[LLMAD] Save json in {json_path}")

    def json_format_check(self, json_obj, df):
        indices = df["index"].values
        for key in [
            "briefExplanation",
            "is_anomaly",
            "anomalies",
            "reason_for_anomaly_type",
            "anomaly_type",
            "reason_for_alarm_level",
            "alarm_level",
        ]:
            if key not in json_obj:
                raise ValueError(f"Key {key} not in the json object")
        # check if is_anomaly is boolean
        if not isinstance(json_obj["is_anomaly"], bool):
            raise ValueError(f"is_anomaly should be boolean")
        # check if anomalies is a list of indices
        if not isinstance(json_obj["anomalies"], list):
            raise ValueError(f"anomalies should be a list")
        # check if indices in anomalies are in the data
        if json_obj["is_anomaly"]:
            for idx in json_obj["anomalies"]:
                if idx not in indices:
                    raise ValueError(f"Index {idx} not in the data")

    # TODO: add eval metric info in the figure name
    def eval(self, scores, eval_df):
        """
        Evaluate the scores with the evaluation dataset

        Args:
            scores ([np.array]): model scores
            eval_df ([pd.DataFrame]): evaluation dataset
        """
        assert len(scores) == len(
            eval_df
        ), "Length of scores and evaluation dataset should be the same"
        report = classification_report(
            eval_df["label"], scores, labels=[0, 1], zero_division=0
        )
        logging.info(report)
        eval_interface = PointF1()
        eval_res_pf1 = eval_interface.calc(scores, eval_df["label"].values, None)
        logging.info(eval_res_pf1.to_dict())

        # # count how many 1s in the labels
        # count = np.count_nonzero(labels)
        # logging.info(f"[ReviewAgent] Number of anomalies: {count}")
        eval_interface = PointF1PA()
        eval_res_pf1pa = eval_interface.calc(scores, eval_df["label"].values, None)
        logging.info(eval_res_pf1pa.to_dict())
        eval_interface = EventF1PA(mode="squeeze")
        eval_res_ef1pa = eval_interface.calc(scores, eval_df["label"].values, None)
        logging.info(eval_res_ef1pa.to_dict())

        # combine 3 dicts in to a final_res_dict
        final_res_dict = {
            **eval_res_pf1.to_dict(),
            **eval_res_pf1pa.to_dict(),
            **eval_res_ef1pa.to_dict(),
        }
        # final_res_path = rule_file.replace(".py", "_eval_res.json")
        # with open(final_res_path, "w") as f:
        #     json.dump(final_res_dict, f)

        # self.visualize(eval_df[["value", "label", "index"]].values, labels)
        # labels = np.zeros(shape=(len(eval_df),))
        # threshold = final_res_dict["event-based f1 under pa with mode squeeze"]["threshold"]
        # labels[scores >= threshold] = 1
        return final_res_dict
