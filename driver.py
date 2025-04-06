# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import tqdm

from common.exception import RuntimeException, SyntaxException
from runtime.engine import Engine


def load_data(data_file):
    try:
        data = np.load(data_file)
        assert len(data["feature"]) == len(
            data["label"]
        ), "Length of feature and label do not match"
        return data
    except Exception as e:
        logging.exception(f"Failed to load data file from {data_file}: {e}")
        raise e


def main(args):
    rule_path = args.rule_path
    result_path = args.result_path
    dataset_path = args.dataset_path
    mode = args.mode
    dataset_mode = args.dataset_mode
    train_test_split = args.train_test_split
    model_res_path = args.model_res_path
    chunk_size = args.chunk_size
    image_chunk_size = args.image_chunk_size
    image_subplots = args.image_subplots
    # remove "," from image_subplots
    image_subplots = (image_subplots[0], image_subplots[2])
    repeat = args.repeat
    top_k = args.top_k
    llm_engine = args.llm_engine
    timeout = args.timeout
    sample_per_prompt = args.sample_per_prompt

    if repeat > 1:
        assert (
            mode != "eval-LLM-only" and mode != "eval-combined"
        ), "Evaluation mode does not support repeat"

    for i in tqdm.tqdm(range(repeat), dynamic_ncols=True):
        postfix = dataset_path.split("/")[-1].split(".")[0]
        final_result_path = os.path.join(
            result_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + postfix
        )

        if mode == "eval-LLM-only":
            engine = Engine(
                dataset_path=dataset_path,
                chunk_size=chunk_size,
                image_chunk_size=image_chunk_size,
                rule_path=final_result_path,
                model_res_path=model_res_path,
                mode="train-LLM-only",
                dataset_mode=dataset_mode,
                train_test_split=train_test_split,
                top_k=top_k,
            )
        elif mode == "eval-combined":
            engine = Engine(
                dataset_path=dataset_path,
                chunk_size=chunk_size,
                image_chunk_size=image_chunk_size,
                rule_path=final_result_path,
                model_res_path=model_res_path,
                mode="train-combined-fn",
                dataset_mode=dataset_mode,
                train_test_split=train_test_split,
                top_k=top_k,
            )
        else:
            engine = Engine(
                dataset_path=dataset_path,
                chunk_size=chunk_size,
                image_chunk_size=image_chunk_size,
                rule_path=final_result_path,
                model_res_path=model_res_path,
                mode=mode,
                dataset_mode=dataset_mode,
                train_test_split=train_test_split,
                top_k=top_k,
                image_subplots=image_subplots,
                llm_engine=llm_engine,
                timeout=timeout,
                sample_per_prompt=sample_per_prompt,
            )

        logging.info(f"args: {args}")
        print(f"args: {args}")

        if mode == "eval-LLM-only":
            assert rule_path, "Rule path must be provided for evaluation mode"
            logging.info(f"Start to evaluate rule {rule_path}")
            try:
                eval_res, labels, _ = engine.eval(rule_path, "test")
                res_path = os.path.join(final_result_path, f"eval_result.json")
                with open(res_path, "w") as f:
                    json.dump(eval_res, f)
            except Exception as e:
                logging.exception(f"Failed to evaluate rule {rule_path}: {e}")
                repair_agent = engine.get_repair_agent()
                repair_agent.set_start_time()
                if isinstance(e, SyntaxException):
                    repair_agent.run(engine.get_test_df(), e.rule_path)
                elif isinstance(e, RuntimeException):
                    repair_agent.run(e.df, e.rule_path)
                else:
                    raise e
        elif mode == "eval-combined":
            assert rule_path, "Rule path must be provided for evaluation mode"
            assert (
                model_res_path
            ), "Model result path must be provided for evaluation mode"
            logging.info(f"Start to evaluate rule {rule_path}")
            engine.combined_eval(rule_path, "test")
        else:
            logging.info(f"Start to train rules")
            engine.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/t-yilegu/AgentAD/datasets/KPI/segments/train_02e99bd4f6cfb33f_1493568000_1501475640.csv",
        help="The data file to load",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="/home/t-yilegu/AgentAD/generated_rules/debug",
        help="The path to save the generated rules",
    )
    parser.add_argument(
        "--rule_path",
        type=str,
        default="",
        help="The path for the rule to evaluate, if provided, will skip the training process",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train-LLM-only",
        choices=[
            "train-LLM-only",
            "train-LLM-only-image",
            "train-combined-fn",
            "train-combined-fp",
            "eval-LLM-only",
            "eval-combined",
            "baseline-LLMAD",
            "ablation-detection-only",
        ],
        help="The mode to run the engine",
    )
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="one-by-one",
        choices=["one-by-one", "all-in-one"],
        help="The dataset mode to run the engine",
    )
    parser.add_argument(
        "--train_test_split",
        type=float,
        default=0.7,
        help="The ratio to split the train and test data, by default the last part is used for test",
    )
    parser.add_argument(
        "--model_res_path",
        type=str,
        default="/home/t-yilegu/Results",
        help="The path for results from EasyTSAD's model output",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="The size of the chunk to process for raw data",
    )
    parser.add_argument(
        "--image_chunk_size",
        type=int,
        default=10000,
        help="The size of the chunk to process for image understanding",
    )
    parser.add_argument(
        "--image_subplots",
        type=tuple,
        default=(1, ",", 1),
        help="The number of subplots for the image",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="The number of times to repeat the training process",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="The number of top k to select for the rule",
    )
    parser.add_argument(
        "--llm_engine",
        type=str,
        default="gpt-4-32k",
        choices=["gpt-4-32k", "gpt-4o", "gpt-35-turbo-16k"],
        help="The engine to use for LLM",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=150,
        help="The timeout for the training process, in minutes",
    )
    parser.add_argument(
        "--sample_per_prompt",
        type=int,
        default=1,
        help="The number of samples per prompt",
    )
    args = parser.parse_args()
    main(args)
