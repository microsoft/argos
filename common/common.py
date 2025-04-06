import builtins
import concurrent.futures
import json
import logging
import os

import numpy as np
import pandas as pd
import tiktoken

from common.exception import RuntimeException

EPS = 1e-10


# Define the function num_tokens_from_messages, which returns the number of tokens used by a set of messages.
def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """Return the number of tokens used by a list of messages."""
    # Try to get the encoding for the model
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # If the model is not found, use the cl100k_base encoding and give a warning
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    # Set the token count for different models
    if model in {
        # "gpt-3.5-turbo-0613",
        # "gpt-3.5-turbo-16k-0613",
        # "gpt-4-0314",
        # "gpt-4-32k-0314",
        # "gpt-4-0613",
        # "gpt-4-32k-0613",
        "gpt-35-turbo-16k",
        "gpt-4o",
        "gpt-4-32k",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # Each message follows the format {role/name}\n{content}\n
        )
        tokens_per_name = -1  # If there is a name, the role will be omitted
    elif "gpt-3.5-turbo" in model:
        # For gpt-3.5-turbo, updates may occur. Here, the token count assumes gpt-3.5-turbo-0613 and gives a warning
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        # For gpt-4, updates may occur. Here, the token count assumes gpt-4-0613 and gives a warning
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        # For unimplemented models, raise a NotImplementedError
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    # Calculate the token count for each message
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # Each reply begins with the assistant
    return num_tokens


def smooth_labels(labels: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Smooth the labels.
    For every index, take the average of window_size samples around it and add the average to the new labels.
    """
    new_labels = np.zeros_like(labels, dtype=float)
    for i in range(len(labels)):
        start = max(0, i - window_size // 2)
        end = min(len(labels), i + window_size // 2 + 1)
        mean_value = np.mean(labels[start:end])
        for j in range(start, end):
            new_labels[j] += mean_value
    return new_labels


def get_model_scores(dataset_name, scores_dir_path):
    final_score_path = scores_dir_path + f"/{dataset_name}.npy"
    scores = np.load(final_score_path)
    return scores


def get_model_labels(scores, dataset_name, eval_dict_path):
    with open(eval_dict_path, "r") as f:
        eval_dict = json.load(f)
    threshold = eval_dict[dataset_name]["event-based f1 under pa with mode squeeze"][
        "threshold"
    ]
    # threshold = eval_dict[dataset_name]["best f1 under pa"]["threshold"]
    # print(f"[Model] Threshold for {dataset_name}: {threshold}")
    # print(f"[Model] Scores for {dataset_name}: {scores} Avg score: {np.mean(scores)} Std score: {np.std(scores)}")
    labels = np.zeros_like(scores)
    labels[scores >= threshold] = 1
    return labels


def get_rule_labels(eval_dict, rule_scores):
    threshold = eval_dict["threshold"]
    # print(f"[Rule] Threshold for rule: {threshold}")
    # print(f"[Rule] Scores for rule: {scores} Avg score: {np.mean(scores)} Std score: {np.std(scores)}")

    labels = np.zeros_like(rule_scores)
    labels[rule_scores >= threshold] = 1

    # preprocess the labels
    # labels = preprocess_labels(model_scores, labels)
    return labels


def preprocess_labels(scores, labels, margins=[0, 5]):
    """Adapted from EasyTSAD/EasyTSAD/Evaluations/Performance.py"""
    try:
        if not isinstance(scores, np.ndarray):
            raise TypeError(
                "Invalid scores type. Make sure that scores are np.ndarray\n"
            )
            # return False, "Invalid scores type. Make sure that scores are np.ndarray\n"
        if scores.ndim != 1:
            raise ValueError("Invalid scores dimension, the dimension must be 1.\n")
            # return False, "Invalid scores dimension, the dimension must be 1.\n"
        if len(scores) > len(labels):
            raise AssertionError(
                "Score length must less than label length! Score length: {}; Label length: {}".format(
                    len(scores), len(labels)
                )
            )
        labels = labels[len(labels) - len(scores) :]
        # gt_labels = gt_labels[:len(self.scores)]

        # avoid negative value in scores
        scores = scores - scores.min()
        assert len(scores) == len(labels)

    except Exception as e:
        # return False, traceback.format_exc()
        raise e

    # pre_margin, post_margin = margins[0], margins[1]
    # if pre_margin == 0 and post_margin == 0:
    #     return

    # # collect label segments
    # ano_seg = []
    # flag, start, l = 0, 0, len(labels)
    # for i in range(l):
    #     if i == l - 1 and flag == 1:
    #         ano_seg.append((start, l))
    #     elif labels[i] == 1 and flag == 0:
    #         flag = 1
    #         start = i
    #     elif labels[i] == 0 and flag == 1:
    #         flag = 0
    #         ano_seg.append((start, i))

    # ano_seg_len = len(ano_seg)
    # if ano_seg_len == 0:return
    # # # process pre_margin
    # labels[max(0, ano_seg[0][0] - pre_margin): ano_seg[0][0] + 1] = 1
    # for i in range(1, ano_seg_len):
    #     labels[max(ano_seg[i-1][1] + 2, ano_seg[i][0] - pre_margin):ano_seg[i][0] + 1] = 1

    # # process post_margin
    # for i in range(ano_seg_len - 1):
    #     labels[ano_seg[i][1] - 1: min(ano_seg[i][1] + post_margin, ano_seg[i+1][0] - 2 - pre_margin)] = 1
    # labels[ano_seg[-1][1] - 1: min(ano_seg[-1][1] + post_margin, l)] = 1

    return labels


def get_gt_labels(train_df):
    gt_labels = train_df["label"].values.copy()
    # gt_labels = preprocess_labels(scores, gt_labels)
    return gt_labels


def combine_labels(model_labels, rule_labels, mode="train-combined-fn"):
    assert len(model_labels) == len(rule_labels)
    # or operation
    if mode == "train-combined-fn":
        # see how many labels are where model is 1 and rule is 0
        count = 0
        for i in range(len(model_labels)):
            if model_labels[i] == 1 and rule_labels[i] == 0:
                count += 1
        logging.info(
            f"[FN] Number of labels where model is 1 and rule is 0: {count}, total: {len(model_labels)}, explainability: {1 - count / len(model_labels)}"
        )
        return np.maximum(model_labels, rule_labels)
    elif mode == "train-combined-fp":
        # see how many labels are where model is 1 and rule is 0
        count = 0
        for i in range(len(model_labels)):
            if model_labels[i] == 0 and rule_labels[i] == 1:
                count += 1
        logging.info(
            f"[FP] Number of labels where model is 0 and rule is 1: {count}, total: {len(model_labels)}, explainability: {1 - count / len(model_labels)}"
        )
        return np.minimum(model_labels, rule_labels)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    # return np.maximum(model_labels, rule_labels)
    # and operation
    # return np.minimum(model_labels, rule_labels)


def calculate_performance(labels, gt_labels):
    # if len(labels) > len(gt_labels):
    #     raise ValueError("Length of labels is longer than length of gt_labels")
    # gt_labels = gt_labels[:len(labels)]
    assert len(labels) == len(gt_labels)

    # event-based f1 under pa with mode squeeze
    func = lambda x: 1
    # point f1 pa
    # func = lambda x: x

    tot_anomaly = 0
    ano_flag = 0
    ll = len(gt_labels)

    abnormal_segments = []
    for i in range(gt_labels.shape[0]):
        if gt_labels[i] > 0.5 and ano_flag == 0:
            ano_flag = 1
            start = i

        # alleviation
        elif gt_labels[i] <= 0.5 and ano_flag == 1:
            ano_flag = 0
            end = i
            tot_anomaly += func(end - start)
            abnormal_segments.append((start, end))

        # marked anomaly at the end of the list
        if ano_flag == 1 and i == ll - 1:
            ano_flag = 0
            end = i + 1
            tot_anomaly += func(end - start)
            abnormal_segments.append((start, end))

    true_positives = 0
    represented_positives = 0
    # for all abnormal segments, if in the segment, at least one point is marked as anomaly, then it is a true positive
    true_positive_segments = []
    for start, end in abnormal_segments:
        if np.sum(labels[start:end]) > 0:
            # event-based f1 under pa with mode squeeze
            true_positives += 1
            # point f1 pa
            # true_positives += (end - start)
            represented_positives += np.sum(labels[start:end])
            true_positive_segments.append((start, end))

    # for all labels not in any of abnormal segments, add them to the false positive
    false_positives = np.sum(labels) - represented_positives
    all_positives = true_positives + false_positives

    # print(f"True positives: {true_positives}, False positives: {false_positives}, Total anomalies: {tot_anomaly}")
    # print(f"True positive segments: {true_positive_segments}")

    precision = true_positives / (all_positives + EPS)
    recall = true_positives / (tot_anomaly + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    res_dict = {"precision": precision, "recall": recall, "f1": f1}
    return res_dict

def format_check(df, rule_path, labels):
    if len(df) != len(labels):
        raise RuntimeException(
            f"The length of the labels is not equal to the length of the data, len(labels)={len(labels)}, len(data)={len(df)}",
            df,
            rule_path,
        )


def run_with_timeout(func, timeout, *args, **kwargs):
    return func(*args, **kwargs)
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     future = executor.submit(func, *args, **kwargs)
    #     try:
    #         result = future.result(timeout=timeout)
    #         return result
    #     except concurrent.futures.TimeoutError:
    #         logging.info(f"Function '{func.__name__}' timed out after {timeout} seconds")
    #         return None


def cleanup_global_env():
    return
    # Capture the current state of globals
    essential_keys = set(dir(builtins)) | {
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__annotations__",
        "__builtins__",
        "__file__",
        "__cached__",
    }

    # Remove all other keys
    for key in list(globals().keys()):
        if key not in essential_keys:
            del globals()[key]

