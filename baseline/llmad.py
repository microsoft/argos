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
from common.common import (calculate_performance, combine_labels, format_check,
                           get_gt_labels, get_model_labels, get_model_scores,
                           get_rule_labels, preprocess_labels,
                           run_with_timeout, smooth_labels)
from common.exception import (RuntimeException, SyntaxException,
                              TimeoutException)
from eval_metrics.point_f1 import calculate_point_f1


class LLMAD(Agent):
    def __init__(self, chunk_size, mode="baseline-LLMAD", llm_engine="gpt-4o") -> None:
        self.chunk_size = chunk_size
        if mode == "baseline-LLMAD":
            llmad_prompt = """
##Instructions 
Determine if there are any anomalies in the provided AIOPS flow data sequence. ##Following Rules: 
1. A data point is considered an anomaly if it is part of a sequence of at least one consecutive anomalous points or continues to plummet or surge abruptly.
2. Given that the vast majority of data points are expected to be no anomaly, Anomalies are exceedingly rare and should only be identified with absolute certainty. 
3. Normal data may exhibit volatility, which should not be mistaken for anomalies. 
4. Mislabeling normal data as an anomaly can lead to catastrophic failures. Exercise extreme caution. False positives are unacceptable. 
5. If do not have 100 percent confidence that data is an anomaly, do not flag it as an anomaly. 
6. The output of anomaly intervals needs to be accurately located and should not be excessively long. 
7. anomaly_type should be one of the following: 
- PersistentLevelShiftUp: The data shifts to a higher value and maintains that level consistently, do not return to the original baseline. like 1 2 1 2 1 2 *500* *480* *510* *500* *500* 
- PersistentLevelShiftDown: The data shifts to a lower value and maintains that level consistently, do not return to the original baseline. like 1 2 1 2 *-100* *-102* *-104* *-110* *-110* 
- TransientLevelShiftUp: The data temporarily shifts to a higher value and then returning to the original baseline, the anomaly maintains for at least 5 data points and return to baseline like 1 2 1 2 1 2 *500* *500* *499* *510* *500* 1 2 1 2 
- TransientLevelShiftDown: The data temporarily shifts to a lower value and then returning to the original baseline, the anomaly maintains for at least 5 data points return to baseline like 1 2 1 2 *-100* *-102* *-104* *-110* *-100* 1 2 1 2 
- SingleSpike: A brief, sharp rise in data value followed by an immediate return to the baseline. like 1 2 1 2 1 2 *200* *500* 1 2 
- SingleDip: A brief, sharp drop in data value followed by an immediate return to the baseline. like 1 2 1 2 *-500* *-200* 1 2 1 2 
- MultipleSpikes: Several brief, sharp rises in data value, each followed by a return to the baseline. like 1 2 *500* 3 2 *510* *200* 1 2 *480* 1 2 
- MultipleDips: Several brief, sharp drops in data value, each followed by a return to the baseline. like 1 2 *-100* 3 2 *-110* *-200* 1 2 *-120* 1 2 
8. alarm_level should be one of the following: 
- Urgent/Error: This category is for values that represent a severe risk, potentially causing immediate damage or harm across all event types whether increases, decreases, spikes, dips, or multiple occurrences. 
- Important: Allocated for moderate value changes (both increases and decreases) that could escalate to future problems or system stress but are not immediately hazardous. This also covers upward transient level shifts that concern system longevity and potential failure indications from downward shifts. 
- Warning: Used for noticeable deviations from the norm that are not yet critical but merit close monitoring. This includes single spikes and dips that are moderate in nature, as well as multiple non-critical spikes and level shifts that are significant but not yet dangerous. 
9. The briefExplanation must comprise a explicit three-step analysis utilizing precise data (do not only repeat the rule): 
- Step 1: Assess the overall trend to ascertain if it aligns with expected patterns, thereby identifying any overarching anomalies. 
- Step 2: Examine the local data segments to detect any specific deviations or anomalies. 
- Step 3: Reassess the identified points to confirm their anomalous nature, given the rarity of true anomalies. 
This step ensures that the detected points are not merely normal fluctuations or seasonal variations. 
10. Provide responses in a strict JSON format suitable for direct parsing, without any additional textual commentary. 
## Data Format
1. You will be given the data sample in following format:
##### DATA
    <multiple known data samples, each sample contains continuous samples, each sample is a tuple of (value, index)>

##Response Format 
{ "briefExplanation": {"step1_global": analysis reason, "step2_local": analysis reason, "step3_reassess": analysis reason}, "is_anomaly": false/true, "anomalies": []/[index1, index2, index3, ...], "reason_for_anomaly_type": "no"/"reason for anomaly type", "anomaly_type": "no"/"classification of main anomaly",(only one) "reason_for_alarm_level": "no"/"reason for alarm level", "alarm_level": "no"/"Urgent/Error"/"Important"/"Warning",(only one) }       
            """
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
        
        eval_res_pf1 = calculate_point_f1(scores, eval_df["label"].values)
        logging.info(eval_res_pf1)

        final_res_dict = eval_res_pf1

        return final_res_dict
