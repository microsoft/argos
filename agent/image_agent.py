# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
import pickle
from datetime import datetime

from agent.agent import LLM, TIMEOUT_IMAGE, Agent
from agent.prompts.image import build_image_agent_prompt
from common.exception import TimeoutException


class ImageAgent(Agent):
    def __init__(self, chunk_size, mode="train-LLM-only", llm_engine="gpt-4o") -> None:
        self.chunk_size = chunk_size
        self.mode = mode
        image_agent_prompt = build_image_agent_prompt(self.chunk_size)
        self.LLM = LLM(
            system_prompt=image_agent_prompt,
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
