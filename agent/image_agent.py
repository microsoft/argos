# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
import pickle
from datetime import datetime

from agent.agent import LLM, TIMEOUT_IMAGE, Agent
from common.exception import TimeoutException


class ImageAgent(Agent):
    def __init__(self, chunk_size, mode="train-LLM-only", llm_engine="gpt-4o") -> None:
        self.chunk_size = chunk_size
        self.mode = mode
        image_agent_prompt = f"""You are an AI assistant that reads images of time series data and provides insights. Specifically, you would observe data patterns in the image and summarize out anomaly types from the data. You will be given the image in the following format.
            
        The image shows a one-dimentional performance metric of size {self.chunk_size} from cloud infrastructure. The x-axis is the timestamp and the y-axis is the value for the metric. The line for the values is in blue color and the anomalies are labeled as red dots. The anomalies are the points that are significantly different from the normal data points. The normal data points are the ones that are not labeled as anomalies. 
        
        Your task is to identify the anomalies in the image and provide a summary of the anomaly types. The anomaly type should be given as a name followed by a brief explanation for the pattern. 
        
        You should return your answer following this format strictly: 
        
        ### Anomaly Types BEGIN ###
        1. AnomalyType1: Explanation1
        2. AnomalyType2: Explanation2
        3. AnomalyType3: Explanation3
        4. AnomalyType4: Explanation4
        5. AnomalyType5: Explanation5
        ### Anomaly Types END ###

        IMPORTANT:
        1. The explanation should be accurate and reflect the anomaly pattern in the image and how it differs from normal patterns. You should also give examples of the anomaly type in the image, detailing how metric value changes in the anomaly type, or the frequency of the anomaly type, or the duration of the anomaly type, etc. You SHOULD give concrete values for your example. But you MUST NOT include information about timestamp or mark or range or anything related to INDEX in the example to prevent overfitting.
        2. Optionally, you will be given the anomaly types derived from last iteration, following after "### Anomaly Types From Last Iteration ###", you can refer to them and provide new anomaly types or update the existing ones. Be extremely cautious when you are updating existing anomaly types, if you find the existing anomaly types are still valid, you should keep them unchanged. If you find there are new examples of the old anomaly types, you should keep the old anomaly types and add new examples to the old anomaly types. If you find the existing anomaly types are not valid, such as they are reflecting the normal patterns, you should remove them. 
        3. Please merge instances of the same anomaly types. For example, if you find one instance of "MultipleSpikes" in one figure and another instance of "MultipleSpikes" in another figure, you should merge them into one instance of "MultipleSpikes", instead of two separate instances.

        """.strip()
        # OLD
        #         1. The explanation should be accurate and reflect the pattern in the image.
        # 2. If possible, please a brief explanation for each anomaly type. You can describe how the value changes in the anomaly type, or the frequency of the anomaly type, or the duration of the anomaly type, etc. But DO NOT include information about timestamp in the explanation to prevent overfitting. You should also give examples of the anomaly type in the image, detailing how value changes in the anomaly type, or the frequency of the anomaly type, or the duration of the anomaly type, etc. But DO NOT include information about timestamp in the example to prevent overfitting.
        # 3. Optionally, you will be given the anomaly types derived from last iteration, following after "### Anomaly Types From Last Iteration ###", you can refer to them and provide new anomaly types or update the existing ones. Be extremely cautious when you are updating existing anomaly types, if you find the existing anomaly types are still valid, you should keep them unchanged. If you find there are new examples of the old anomaly types, you should keep the old anomaly types. If you find the existing anomaly types are not valid, such as they are reflecting the normal patterns, you should remove them.
        # NEW
        #        1. The explanation should be accurate and reflect the anomaly pattern in the image and how it differs from normal patterns. You should also give examples of the anomaly type in the image, detailing how metric value changes in the anomaly type, or the frequency of the anomaly type, or the duration of the anomaly type, etc. You SHOULD give concrete values for your example. But you MUST NOT include information about timestamp or mark or range or anything related to INDEX in the example to prevent overfitting.
        # 3. Optionally, you will be given the anomaly types derived from last iteration, following after "### Anomaly Types From Last Iteration ###", you can refer to them and provide new anomaly types or update the existing ones. Be extremely cautious when you are updating existing anomaly types, if you find the existing anomaly types are still valid, you should keep them unchanged. If you find there are new examples of the old anomaly types, you should keep the old anomaly types and add new examples to the old anomaly types. If you find the existing anomaly types are not valid, such as they are reflecting the normal patterns, you should remove them.
        self.LLM = LLM(
            system_prompt=image_agent_prompt.strip(),
            temperature=0.75,
            past_message_num=10,
            engine=llm_engine,
        )
        self.name = "ImageAgent"

    def run(self, image_path, last_anomaly_types=None) -> str:
        logging.info(
            f"[ImageAgent] Start to understand anomaly types and patterns from image in {image_path}"
        )

        while self.get_elapsed_time() < TIMEOUT_IMAGE:
            final_query = ""

            if last_anomaly_types is not None:
                final_query += (
                    "### Anomaly Types From Last Iteration ###\n"
                    + last_anomaly_types
                    + "\n"
                )

            # logging.info(f"[DetectionAgentV3] Query to LLM: {final_query}")

            ans = self.LLM.query_with_image(final_query, image_path)
            self.LLM.reset()

            # if inference not in answer, then we assume it fails to generate code, and directly retry
            if "### Anomaly Types BEGIN ###" and "### Anomaly Types END ###" in ans:
                logging.info(f"[ImageAgent] LLM generated the following answer:")
                logging.info(ans)

                # extract the anomaly types between  ### Anomaly Types BEGIN ### and ### Anomaly Types END ###
                anomaly_types = (
                    ans.split("### Anomaly Types BEGIN ###")[1]
                    .split("### Anomaly Types END ###")[0]
                    .strip()
                )
                return anomaly_types
            else:
                logging.info(
                    f"[ImageAgent] LLM did not generate an answer with correct format, retrying:"
                )
                logging.info(ans)

        if self.get_elapsed_time() >= TIMEOUT_IMAGE:
            logging.info(
                f"[ImageAgent] Time out to understand pattern in image {image_path}"
            )
            raise TimeoutException()
