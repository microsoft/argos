# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

from agent.agent import LLM, TIMEOUT, Agent
from agent.prompts.detection import (BASE_MODEL_PROMPT, DETECTION_AGENT_V1_PROMPT,
                                     build_detection_agent_v2_prompt,
                                     build_detection_agent_v3_prompt)
from eval_metrics.event_f1pa import EventF1PA
from eval_metrics.point_f1 import PointF1
from eval_metrics.point_f1pa import PointF1PA


class DetectionAgent(Agent):
    def __init__(self, dataset, rule_path="/tmp") -> None:
        detection_agent_prompt = DETECTION_AGENT_V1_PROMPT.strip()
        self.LLM = LLM(
            system_prompt=detection_agent_prompt,
            temperature=0.75,
            past_message_num=10,
        )

        self.rule_path = rule_path
        self.update_rule_path_prefix()
        os.makedirs(self.rule_path, exist_ok=True)
        self.rule_file = os.path.join(self.rule_path, "rule.py")

        idx = int(len(dataset["label"]) * 0.70)
        feats, labels = np.array_split(dataset["feature"][idx:], 2), np.array_split(
            dataset["label"][idx:], 2
        )
        self.feats = {"val": feats[0], "test": feats[1]}
        self.labels = {"val": labels[0], "test": labels[1]}
        self.batch_size = 8
        self.max_iter = 5

        self.base_model_name = ""
        self.gen_base_model()

        if self.base_model_name:
            with open(self.base_model_name, "rb") as f:
                self.base_model = pickle.load(f)
            logging.info(f"Loaded {self.base_model_name}.")

            self.pred_labels = {
                "val": self.base_model.predict(
                    self.feats["val"].reshape(self.feats["val"].shape[0], -1)
                ),
                "test": self.base_model.predict(
                    self.feats["test"].reshape(self.feats["test"].shape[0], -1)
                ),
            }
        else:
            self.pred_labels = {
                "val": np.negative(self.labels["val"]),
                "test": np.negative(self.labels["test"]),
            }

        self.metrics = {
            "val": {
                "fp": [],
                "precision": [],
                "recall": [],
            },
            "test": {
                "fp": [],
                "precision": [],
                "recall": [],
            },
        }
        self.eval("val")
        self.eval("test")

    def update_rule_path_prefix(self) -> None:
        self.rule_path_prefix = os.path.join(
            self.rule_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )

    def gen_base_model(self) -> None:
        base_model_prompt = BASE_MODEL_PROMPT.strip()
        ans = LLM(base_model_prompt, temperature=0.7, past_message_num=10).query(
            "Please give the python function to train the model"
        )
        self.train_base_model()

    def train_base_model(self) -> None:
        def clf_gpt(features, labels):
            # Create the Isolation Forest model
            model = IsolationForest(
                contamination=min(0.5, float(sum(labels == -1)) / len(labels)),
                random_state=42,
            )
            model.fit(features.reshape(features.shape[0], -1))
            return model

        logging.info("Training base model ...")
        clf = clf_gpt(self.feats["val"], self.labels["val"])
        self.base_model_name = os.path.join(self.rule_path, "base.pkl")
        with open(self.base_model_name, "wb") as f:
            pickle.dump(clf, f)
        logging.info(f"Base model saved to {self.base_model_name}")

        pred_label = clf.predict(
            self.feats["test"].reshape(self.feats["test"].shape[0], -1)
        )
        print(classification_report(self.labels["test"], pred_label, labels=[-1, 1]))
        # tn, fp, fn, tp = confusion_matrix(self.labels["test"], pred_label).ravel()

    def save_fig(self) -> str:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        for i in range(2):
            metric = self.metrics["val"] if i == 0 else self.metrics["test"]
            ax1 = axes[i]
            (line1,) = ax1.plot(metric["precision"], label="Precision", marker="o")
            (line2,) = ax1.plot(metric["recall"], label="Recall", marker="s")
            ax1.set_xlabel("Iteration No.")
            ax1.set_ylabel("Precision / Recall")
            ax1.set_title(f"{'Validation' if i == 0 else 'Test'} Set")
            ax1.xaxis.set_major_locator(MultipleLocator(1))

            ax2 = ax1.twinx()
            (line3,) = ax2.plot(
                metric["fp"], label="# Missed", color="gray", linestyle="dashed"
            )
            ax2.set_ylabel("# Missed")

            lines = [line1, line2, line3]
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc="lower left")

        plt.tight_layout()
        fig_name = self.rule_path_prefix
        fig_name += f".{int(self.metrics['test']['precision'][-1] * 1000)}-{int(self.metrics['test']['recall'][-1] * 1000)}.jpg"
        fig.savefig(fig_name)
        plt.close(fig)
        logging.info(f"Save figure to {fig_name}.")
        return fig_name

    def save_rule(self, rule: str) -> None:
        with open(self.rule_file, "w") as f:
            f.write(rule)
        logging.info(f"Write rule to {self.rule_file}")

    def eval(self, sn: str = "test", rule: callable = None) -> None:
        pred_label = np.copy(self.pred_labels[sn])
        if rule:
            rule_label = np.array([-1 if rule(each) else 1 for each in self.feats[sn]])
            pred_label[np.where(rule_label == -1)[0]] = -1

        logging.info(f"Evaluating on {sn} set.")
        logging.info(
            str(
                classification_report(
                    self.labels[sn], pred_label, labels=[-1, 1], zero_division=0
                )
            )
        )
        report = classification_report(
            self.labels[sn],
            pred_label,
            labels=[-1, 1],
            output_dict=True,
            zero_division=0,
        )
        for n in ["precision", "recall"]:
            self.metrics[sn][n].append(round(report["-1"][n], 4))
        tn, fp, fn, tp = confusion_matrix(
            self.labels[sn], pred_label, labels=[-1, 1]
        ).ravel()
        self.metrics[sn]["fp"].append(fp)

    def extract_code(self, text: str, lang: str = "py") -> str:
        start, end = f"```{lang}", "```"
        code = text[
            text.find(start) + len(start) : text.find(end, text.find(start) + 1)
        ].strip()
        return code.strip()

    def run(self) -> bool:
        self.update_rule_path_prefix()
        # label = -1 is abnormal, label = 1 is normal
        # we will feed the abnormal data where current model predicts as normal wrongly
        _err_loc = np.where(
            np.logical_and(self.labels["val"] == -1, self.pred_labels["val"] == 1)
        )[0]
        _acc_loc = np.where(
            np.logical_and(self.labels["val"] == 1, self.pred_labels["val"] == 1)
        )[0]
        iter, err_loc, violation_loc = 0, np.copy(_err_loc), np.array([])

        best_iter = -1
        while iter < self.max_iter and (len(err_loc) > 0 or len(violation_loc) > 0):
            iter += 1
            logging.info(f"=== Iteration {iter} ===")

            curr_batch = (
                self.feats["val"][err_loc[: self.batch_size], :]
                if len(err_loc) > 0
                else np.array(["None"])
            )
            violation_batch = (
                self.feats["val"][violation_loc[: self.batch_size], :]
                if len(violation_loc) > 0
                else np.array(["None"])
            )
            ans = self.LLM.query(
                f"##### NEGATIVE/ABNORMAL DATA\n{chr(10).join([str(each.tolist()) for each in curr_batch])}\n##### POSITIVE/NORMAL DATA\n{chr(10).join([str(each.tolist()) for each in violation_batch])}"
            )
            if "CODE" in ans:
                exec(self.extract_code(ans), globals())
            else:
                logging.info(f"Unknown answer from LLM: {ans}")
            try:
                self.eval(sn="val", rule=is_negative)
                self.eval(sn="test", rule=is_negative)
            except Exception as e:
                logging.warn(f"Failed to execute rule: {e}")
                self.LLM.reset()
                iter -= 1
                continue

            # update error samples
            rule_err_new_label = np.array(
                [-1 if is_negative(self.feats["val"][i]) else 1 for i in _err_loc]
            )
            rule_acc_new_label = np.array(
                [-1 if is_negative(self.feats["val"][i]) else 1 for i in _acc_loc]
            )
            err_loc = np.delete(_err_loc, np.where(rule_err_new_label == -1)[0])
            violation_loc = np.delete(_acc_loc, np.where(rule_acc_new_label == 1)[0])

            for n in ["val", "test"]:
                metrics = self.metrics[n]
                logging.info(
                    f"{n.title()} set: #FP {str(metrics['fp'])}, precisions {str(metrics['precision'])}, recalls {str(metrics['recall'])}"
                )
            if self.metrics["val"]["precision"][-1] > self.metrics["val"]["precision"][
                0
            ] and self.metrics["val"]["precision"][-1] == max(
                self.metrics["val"]["precision"]
            ):
                self.save_rule(self.extract_code(ans))
                best_iter = iter
            if iter == self.max_iter and best_iter < 0:
                self.save_rule(self.extract_code(ans))
            # if max(self.metrics["val"]["precision"]) - self.metrics["val"]["precision"][-1] > 0.1:
            #     self.LLM.reset()
            #     continue

        fig_name = self.save_fig()
        return (
            bool(best_iter > 0),
            fig_name,
            f"precision/recall changed from {self.metrics['val']['precision'][0]}, {self.metrics['val']['recall'][0]} to {self.metrics['val']['precision'][best_iter]}, {self.metrics['val']['recall'][best_iter]}",
        )


class DetectionAgentV2(Agent):
    def __init__(self, dataset_path, rule_path="/tmp") -> None:
        self.chunk_size = 2500
        detection_agent_prompt = build_detection_agent_v2_prompt(self.chunk_size)
        self.LLM = LLM(
            system_prompt=detection_agent_prompt,
            temperature=0.75,
            past_message_num=10,
        )

        df = pd.read_csv(dataset_path)
        # for every value, chunk to 3 digits after the decimal point
        df["value"] = df["value"].apply(lambda x: round(x, 3))

        self.train_test_split = 0.7

        self.train_df = df[: int(len(df) * self.train_test_split)]
        self.test_df = df[int(len(df) * self.train_test_split) :]

        self.cur_iter = 0
        self.max_iter = 50

        self.rule_path = rule_path
        self.update_rule_path_prefix()
        os.makedirs(self.rule_path, exist_ok=True)
        self.rule_file_base = os.path.join(self.rule_path, "rule")
        # save logs to file
        logging.basicConfig(
            filename=os.path.join(self.rule_path, "output.log"),
            level=logging.INFO,
        )

    def update_rule_path_prefix(self) -> None:
        self.rule_path_prefix = os.path.join(
            self.rule_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )

    def extract_code(self, text: str, lang: str = "python") -> str:
        start, end = f"```{lang}", "```"
        code = text[
            text.find(start) + len(start) : text.find(end, text.find(start) + 1)
        ].strip()
        return code.strip()

    def save_rule(self, rule: str) -> None:
        with open(self.get_rule_path(), "w") as f:
            f.write(rule)
        logging.info(f"Write rule to {self.get_rule_path()}")

    def run(self) -> bool:
        self.update_rule_path_prefix()

        self.cur_iter = 0
        error_message = None
        last_rule_path = None
        while self.cur_iter < self.max_iter:
            logging.info(f"=== Iteration {self.cur_iter} ===")

            if self.cur_iter >= 1:
                # read code into str
                with open(last_rule_path, "r") as f:
                    rule = f.read()
            else:
                rule = None

            start = self.cur_iter * self.chunk_size
            end = min((self.cur_iter + 1) * self.chunk_size, len(self.train_df))
            current_data = self.train_df[start:end]
            current_data_str = current_data.to_string(index=False, header=False)
            # save current data to a file
            # current_data.to_csv(f"{self.rule_path}/data_{self.cur_iter}.csv", index=False)
            # print(current_data_str)
            final_query = "##### DATA\n" + current_data_str + "\n"
            if rule:
                final_query += "##### CODE FROM LAST ITERATION\n" + rule
            if error_message:
                final_query += (
                    "\n"
                    + "##### ERROR FROM EXECUTING CODE, PLEASE FIX IT\n"
                    + error_message
                )
            ans = self.LLM.query(final_query)
            self.LLM.reset()
            # if inference not in answer, then we assume it fails to generate code, and directly retry
            if "inference" in ans:
                logging.info(f"Extract code from LLM: {ans}")
                self.save_rule(self.extract_code(ans))
                last_rule_path = self.get_rule_path()
                try:
                    self.eval(self.get_rule_path(), self.test_df)
                except Exception as e:
                    logging.exception(e)
                    error_message = str(e)
                    continue
                self.cur_iter += 1
                error_message = None

    def eval(self, rule_file, eval_df):
        """
        Evaluate the rule on the given dataset
        """
        # test 1 2 3 4 5
        # label 0 0 0 1 1
        # window = 3, how to generate anomaly score
        # label-after 0 0 0.33 0.66 1
        # redo it for a few iterations

        # ground-truth: 0 0 1 1 0
        # precision - abnormal

        with open(rule_file, "r") as f:
            rule = f.read()
        # logging.info(f"Start to evaluate the rule on train dataset")
        # abnormal_indices = []
        # for i in range(len(self.train_df) // self.chunk_size):
        #     start = i * self.chunk_size
        #     end = (i + 1) * self.chunk_size
        #     current_data = self.train_df[start:end]
        #     current_data_str = current_data.to_string(index=False, header=False)
        #     exec(rule, globals())
        #     abnormal_indices.append(is_negative(current_data.values))
        # print(abnormal_indices)
        logging.info(f"Start to evaluate the rule on test dataset")
        labels = np.ndarray(shape=(len(eval_df),), dtype=int)
        for i in range(len(eval_df) // self.chunk_size):
            start = i * self.chunk_size
            end = (i + 1) * self.chunk_size
            current_data = eval_df[start:end].copy()
            # drop label column
            current_data.drop(columns=["label"], inplace=True)
            exec(rule, globals())
            # print(inference(current_data.values).shape)
            try:
                labels[start:end] = inference(current_data.values)
            except Exception as e:
                logging.exception(e)
                labels[start:end] = np.zeros(shape=(self.chunk_size))
                # throw exception to parent function
                raise e
        report = classification_report(
            eval_df["label"], labels, labels=[0, 1], zero_division=0
        )
        logging.info(report)
        eval_interface = PointF1()
        eval_res = eval_interface.calc(labels, eval_df["label"].values, None)
        logging.info(eval_res.to_dict())
        eval_interface = PointF1PA()
        eval_res = eval_interface.calc(labels, eval_df["label"].values, None)
        logging.info(eval_res.to_dict())
        eval_interface = EventF1PA(mode="squeeze")
        eval_res = eval_interface.calc(labels, eval_df["label"].values, None)
        logging.info(eval_res.to_dict())
        self.visualize(eval_df[["value", "label", "index"]].values, labels)

    def visualize(self, test_data, labels):
        """
        test_data: np.ndarray (shape=(n, 3), (value, label, index))
        labels: np.ndarray (shape=(n,), 0 for normal or 1 for abnormal)
        """
        # shift the index to start from 0
        test_data[:, 2] = test_data[:, 2] - test_data[0, 2]

        fig, ax = plt.subplots(figsize=(30, 10))
        # first plot value vs index
        ax.plot(test_data[:, 2], test_data[:, 0])
        # now highlight the anomaly point with red dot
        for i in range(len(test_data)):
            if test_data[i, 1] == 1:
                ax.plot(test_data[i, 2], test_data[i, 0], "ro")

        # count anomaly segment
        ano_seg = []
        ano_flag = 0
        start, end = 0, 0
        for i in range(len(labels)):
            if labels[i] == 1 and ano_flag == 0:
                start = i
                ano_flag = 1
            elif labels[i] == 0 and ano_flag == 1:
                end = i
                ano_flag = 0
                ano_seg.append((start, end))

            if i == len(labels) - 1 and labels[i] == 1:
                end = i
                ano_seg.append((start, end))
        # do a vertical span on every label=1 point in labels
        for seg in ano_seg:
            ax.axvspan(seg[0], seg[1], alpha=0.5, color="pink")
        figure_path = os.path.join(self.rule_path, f"figure_{self.cur_iter}.png")
        plt.savefig(figure_path)
        # plt.show()

    def get_train_df(self):
        return self.train_df

    def get_test_df(self):
        return self.test_df

    def get_rule_path(self):
        rule_path = self.rule_file_base + f"_{self.cur_iter}.py"
        # rule_path = self.rule_file_base + ".py"
        return rule_path


# TODO: make 5 parallel process agents based on current iteration, then pick the best out of 5, or make 10 parallel process agents, then pick the top 2 out of 10, and iterate
# TODO: upload a figure first and let LLM summarize anomaly type overview, and then based on these types and actual data write rules
class DetectionAgentV3(Agent):
    def __init__(
        self, chunk_size, mode="train-LLM-only", llm_engine="gpt-4o", timeout=150
    ) -> None:
        self.chunk_size = chunk_size
        self.mode = mode
        detection_agent_prompt = build_detection_agent_v3_prompt(self.chunk_size, mode)
        self.LLM = LLM(
            system_prompt=detection_agent_prompt,
            temperature=0.75,
            past_message_num=10,
            engine=llm_engine,
        )
        self.name = "DetectionAgentV3"
        self.max_time = timeout * 60
        print(f"[DetectionAgentV3] Initialized with mode {mode} and timeout {timeout}")

    def run(
        self,
        curr_dfs,
        curr_rule_path,
        last_rule_path=None,
        additional_dfs=None,
        anomaly_types=None,
        image_path=None,
    ) -> None:
        logging.info(
            f"[DetectionAgentV3] Start to detect the pattern and save in {curr_rule_path}"
        )

        while self.get_elapsed_time() < self.max_time:

            if last_rule_path:
                # read code into str
                with open(last_rule_path, "r") as f:
                    rule = f.read()
            else:
                rule = None
            final_query = ""
            for i, curr_df in enumerate(curr_dfs):
                current_data_str = curr_df.to_string(index=False, header=False)
                final_query += f"##### DATA {i}\n" + current_data_str + "\n"
            if additional_dfs is not None:
                for i, additional_df in enumerate(additional_dfs):
                    if additional_df is None:
                        continue
                    additional_data_str = additional_df.to_string(
                        index=False, header=False
                    )
                    if self.mode == "train-combined-fn":
                        final_query += (
                            f"##### NORMAL DATA {i} \n" + additional_data_str + "\n"
                        )
                    elif self.mode == "train-combined-fp":
                        final_query += (
                            f"##### ABNORMAL DATA {i}\n" + additional_data_str + "\n"
                        )
                    else:
                        raise ValueError(f"Invalid mode: {self.mode}")
            if anomaly_types is not None:
                # split by newline
                splits = anomaly_types.split("\n")
                # add a # in front of each line
                for split in splits:
                    split = "#  " + split
                anomaly_types_str = "\n".join(splits)
                final_query += (
                    "##### Anomaly Types BEGIN #####\n"
                    + anomaly_types_str
                    + "\n##### Anomaly Types END #####\n"
                )

            if rule:
                final_query += "##### CODE FROM LAST ITERATION\n" + rule

            # logging.info(f"[DetectionAgentV3] Query to LLM: {final_query}")

            if image_path:
                logging.info(
                    f"[DetectionAgentV3] Query to LLM with image: {image_path}"
                )
                ans = self.LLM.query_with_image(final_query, image_path)
            else:
                ans = self.LLM.query(final_query)
            self.LLM.reset()

            try:
                logging.info(f"[DetectionAgentV3] Extract code from LLM: {ans}")
                code = self.extract_code(ans)
            except Exception as e:
                logging.info(
                    f"[DetectionAgentV3] Failed to extract code from LLM: {ans}"
                )
                continue
            self.save_rule(code, curr_rule_path)
            break

        if self.get_elapsed_time() >= self.max_time:
            logging.info(
                f"[DetectionAgentV3] Time out to detect the pattern and save in {curr_rule_path}"
            )
            return
