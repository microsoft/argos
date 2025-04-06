# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import base64
import logging
import os
import pprint
import re
import sys
import time
from abc import ABC, abstractmethod

from openai import AzureOpenAI

from common.common import num_tokens_from_messages

MAX_ITER = 100
TIMEOUT = 30 * 60 * 5
TIMEOUT_FIRST_REVIEW = 2 * 60
TIMEOUT_LLM = 5 * 60
TIMEOUT_PER_REVIEW = 2 * 60
TIMEOUT_IMAGE = 10 * 60
TIMEOUT_INFERENCE = 1 * 60


class LLM:
    def __init__(
        self,
        system_prompt: str = "You are an AI assistant that helps people find information.",
        engine: str = "gpt-4o",
        temperature: float = 0.1,
        past_message_num: int = sys.maxsize,
    ) -> None:
        self.name = engine
        self.engine = engine
        self.system_prompt = system_prompt
        self.past_message_num = max(0, past_message_num)
        # list of tuple of (input_token_count, output_token_count)
        self.input_output_token_count = []
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

        self.parameters = {
            "model": engine,
            "temperature": temperature,
            "max_tokens": 2000,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "n": 1,
        }

        self._openai = AzureOpenAI(
            api_key=os.environ["OPENAI_AZURE_API_KEY"],
            azure_endpoint=os.environ["OPENAI_AZURE_ENDPOINT"],
            api_version=os.environ["OPENAI_AZURE_API_VERSION"],
        )

        logging.info(f"Initialized LLM with the following parameters:")
        logging.info(pprint.pformat(self.parameters, width=120, compact=True))

    def reset(self) -> None:
        self.update_messages(reset=True)

    def update_messages(self, reset=False) -> None:
        if self.past_message_num > 0 and not reset:
            self.messages = [self.messages[0]] + self.messages[1:][
                -self.past_message_num :
            ]
        else:
            self.messages = [self.messages[0]]

    def query(self, user_prompt: str) -> str:
        self.messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        return self.send_messages()

    def query_with_image(self, user_prompt: str, image_path: str) -> str:

        assert self.engine == "gpt-4o", "Only gpt-4o engine supports image input."

        encoded_image = base64.b64encode(open(image_path, "rb").read()).decode("ascii")
        content = [
            {
                "type": "text",
                "text": user_prompt,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded_image}",
                },
            },
        ]
        self.messages.append(
            {
                "role": "user",
                "content": content,
            }
        )

        # print(f"[LLM] Start to query {self.name} with the following prompts and image from image path {image_path}:")
        # print(pprint.pformat(user_prompt, width=120, compact=True))

        return self.send_messages()

    def send_messages(self):
        ans, timeout = "", 2
        input_token_count = num_tokens_from_messages(self.messages, model=self.engine)

        start_time = time.time()
        while not ans and time.time() - start_time < TIMEOUT_LLM:
            try:
                time.sleep(timeout)
                response = self._openai.chat.completions.create(
                    messages=self.messages, **self.parameters
                )
                ans = response.choices[0].message.content
            except Exception as e:
                logging.exception(e)
            if not ans:
                timeout = timeout + 1 if timeout < 5 else timeout * 2
                logging.info(f"Will retry after {timeout} seconds ...")

        if not ans:
            raise TimeoutError(f"Timeout after {TIMEOUT_LLM} seconds.")

        elapsed_time = time.time() - start_time

        logging.info(
            f"[LLM] Query {self.name} finished with the following answer after {elapsed_time} seconds:"
        )
        logging.info(ans)

        self.messages.append(
            {
                "role": "assistant",
                "content": ans,
            }
        )
        output_messages = [self.messages[-1]]
        output_token_count = num_tokens_from_messages(
            output_messages, model=self.engine
        )

        self.input_output_token_count.append((input_token_count, output_token_count))

        self.update_messages()
        return ans

    def get_token_count(self):
        total_input_token_count = 0
        total_output_token_count = 0
        for input_token_count, output_token_count in self.input_output_token_count:
            total_input_token_count += input_token_count
            total_output_token_count += output_token_count
        return (
            self.input_output_token_count,
            total_input_token_count,
            total_output_token_count,
        )


class Agent(ABC):
    LLM: LLM
    name: str

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def name(self):
        return self.name

    def extract_code(
        self, text: str, lang: str = "python", function_name: str = "def inference"
    ) -> str:
        start, end = f"```{lang}", "```"
        # use regex to extract the code block that start with start and end with end
        pattern = re.compile(f"{start}.*{end}", re.DOTALL)
        matches = pattern.findall(text)
        for match in matches:
            if function_name in match:
                return match.replace(start, "").replace(end, "").strip()
        raise ValueError(
            f"Cannot find code block with function name {function_name} in the text."
        )

    def save_rule(self, rule: str, save_path: str) -> None:
        with open(save_path, "w") as f:
            f.write(rule)
        logging.info(f"Write rule to {save_path}")

    def set_start_time(self) -> None:
        self.start_time = time.time()

    def get_elapsed_time(self) -> float:
        assert hasattr(self, "start_time"), "Please set start time first."
        return time.time() - self.start_time

    def get_token_count(self):
        return self.LLM.get_token_count()
