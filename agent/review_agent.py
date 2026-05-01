# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import os
import subprocess
import time

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from agent.agent import (LLM, TIMEOUT, TIMEOUT_FIRST_REVIEW, TIMEOUT_INFERENCE,
                         TIMEOUT_PER_REVIEW, Agent)
from agent.prompts.review import build_review_agent_prompt
from common.common import (calculate_performance, cleanup_global_env,
                           combine_labels, format_check, get_gt_labels,
                           get_model_labels, get_model_scores, get_rule_labels,
                           preprocess_labels, run_with_timeout, smooth_labels)
from common.exception import (RuntimeException, SyntaxException,
                              TimeoutException)
from datasets.dataset import ArgosDataset
from eval_metrics.event_f1pa import EventF1PA
from eval_metrics.point_f1 import PointF1
from eval_metrics.point_f1pa import PointF1PA

EVOLUTION_THRESHOLD = 0.10

class ReviewAgent(Agent):
    def __init__(
        self,
        chunk_size,
        dataset,
        dataset_name=None,
        model_res_path=None,
        mode="train-LLM-only",
        llm_engine="gpt-4o",
        timeout=150,
    ) -> None:
        review_agent_prompt = build_review_agent_prompt(mode)

        self.LLM = LLM(
            system_prompt=review_agent_prompt,
            temperature=0.75,
            past_message_num=10,
            engine=llm_engine,
        )
        self.name = "ReviewAgent"

        self.dataset = dataset
        self.dataset_name = dataset_name
        self.chunk_size = chunk_size
        self.mode = mode
        self.model_res_path = model_res_path
        self.max_time = timeout * 60
        self.consecutive_val_decreases = 0
        self.best_val_f1 = 0.0
        print(
            f"[ReviewAgent] Initialized with chunk_size={chunk_size}, mode={mode}, llm_engine={llm_engine}, timeout={timeout}"
        )

    def run(self, curr_rule_path, last_rule_path=None) -> None:
        logging.info(f"[ReviewAgent] Start to review the code in {curr_rule_path}")
        self.set_start_time()

        while self.get_elapsed_time() < TIMEOUT_PER_REVIEW:
            # if not last_rule_path:
            #     break
            # curr_validate_res, _ = self.eval(curr_rule_path, self.train_df)
            # last_validate_res, _ = self.eval(last_rule_path, self.train_df)
            if self.dataset.get_dataset_mode() == "one-by-one":
                if self.mode == "train-LLM-only" or self.mode == "train-LLM-only-image" or self.mode=="train-evolution" or self.mode=="train-LLM-only-parallel":
                    curr_validate_res, _, _ = self.eval(
                        curr_rule_path, self.dataset.get_train_df()
                    )
                else:
                    curr_validate_res, _, _ = self.combined_eval(
                        curr_rule_path, "train"
                    )
            elif self.dataset.get_dataset_mode() == "all-in-one":
                if self.mode == "train-LLM-only" or self.mode == "train-LLM-only-image":
                    train_dict = self.dataset.get_train_dict()
                    curr_validate_res, _, _ = self.eval_all_in_one(
                        curr_rule_path, train_dict
                    )
                # TODO: support combined evaluation
                else:
                    curr_validate_res, _, _ = self.combined_eval_all_in_one(
                        curr_rule_path, "train"
                    )
            else:
                raise ValueError(
                    f"Unsupported dataset mode: {self.dataset.get_dataset_mode()}"
                )

            final_res_path = curr_rule_path.replace(".py", "_eval_res_train.json")
            with open(final_res_path, "w") as f:
                json.dump(curr_validate_res, f)
            if not last_rule_path:
                if self.mode == "train-LLM-only" or self.mode == "train-LLM-only-image" or self.mode=="train-evolution" or self.mode=="train-LLM-only-parallel":
                    break
                if self.dataset.get_dataset_mode() == "one-by-one":
                    last_validate_res, _ = self.get_baseline_performance("train")
                else:
                    last_validate_res, _ = self.get_baseline_performance_all_in_one(
                        "train"
                    )
            else:
                if self.dataset.get_dataset_mode() == "one-by-one":
                    if (
                        self.mode == "train-LLM-only"
                        or self.mode == "train-LLM-only-image"
                        or self.mode == "train-evolution"
                        or self.mode == "train-LLM-only-parallel"
                    ):
                        last_validate_res, _, _ = self.eval(
                            last_rule_path, self.dataset.get_train_df()
                        )
                    else:
                        last_validate_res, _, _ = self.combined_eval(
                            last_rule_path, "train"
                        )
                elif self.dataset.get_dataset_mode() == "all-in-one":
                    if (
                        self.mode == "train-LLM-only"
                        or self.mode == "train-LLM-only-image"
                    ):
                        train_dict = self.dataset.get_train_dict()
                        last_validate_res, _, _ = self.eval_all_in_one(
                            last_rule_path, train_dict
                        )
                    # TODO: support combined evaluation
                    else:
                        last_validate_res, _, _ = self.combined_eval_all_in_one(
                            last_rule_path, "train"
                        )

                else:
                    raise ValueError(
                        f"Unsupported dataset mode: {self.dataset.get_dataset_mode()}"
                    )

            has_regression = False
            for key in curr_validate_res.keys():
                # only compare f1 score
                if key != "f1":
                    continue
                # TODO: maybe performance staying the same is also a form of regression
                if self.mode == "train-evolution":
                    if curr_validate_res[key] < last_validate_res[key] - EVOLUTION_THRESHOLD:
                        has_regression = True
                        break
                else:
                    if curr_validate_res[key] < last_validate_res[key]:
                        has_regression = True
                        break

            if not has_regression:
                logging.info(
                    f"[ReviewAgent] The code has no performance regression, no need to propose modifications"
                )
                break

            with open(curr_rule_path, "r") as f:
                rule = f.read()
            final_query = "##### CODE" + rule

            if last_rule_path:
                # get code diff
                result = subprocess.run(
                    ["diff", last_rule_path, curr_rule_path], stdout=subprocess.PIPE
                )

                final_query += (
                    "\n" + "##### CODE DIFFERENCE\n" + result.stdout.decode("utf-8")
                )

            final_query += "\n" + "##### PERFORMANCE METRICS\n"
            for key in curr_validate_res.keys():
                if key == "rule_path" or key == "threshold":
                    continue
                final_query += f"{key}: {curr_validate_res[key]} ({curr_validate_res[key] - last_validate_res[key]}),"

            # Collect regression samples where previous rule was correct but current rule is wrong
            if last_rule_path:
                if self.dataset.get_dataset_mode() == "one-by-one":
                    regression_text = self.collect_regression_samples(
                        curr_rule_path, last_rule_path, self.dataset.get_train_df()
                    )
                elif self.dataset.get_dataset_mode() == "all-in-one":
                    # For all-in-one, collect from the first few train chunks
                    train_dict = self.dataset.get_train_dict()
                    import pandas as pd
                    sample_dfs = [train_dict[k] for k in list(train_dict.keys())[:5]]
                    sample_df = pd.concat(sample_dfs) if sample_dfs else None
                    regression_text = self.collect_regression_samples(
                        curr_rule_path, last_rule_path, sample_df
                    ) if sample_df is not None else ""
                else:
                    regression_text = ""
                final_query += regression_text

            logging.info(
                f"[ReviewAgent] The code has performance regression, start to propose modifications: {final_query}"
            )

            ans = self.LLM.query(final_query)
            self.LLM.reset()

            # if inference not in answer, then we assume it fails to generate code, and directly retry
            # if "inference" in ans:
            #     logging.info(f"[ReviewAgent] Extract code from LLM: {ans}")
            #     code = self.extract_code(ans)
            #     if code == "":
            #         continue
            #     self.save_rule(code, curr_rule_path)
            # else:
            #     logging.info(f"[ReviewAgent] LLM did not generate a function with name inference, retry now: {ans}")
            try:
                logging.info(f"[ReviewAgent] Extract code from LLM: {ans}")
                code = self.extract_code(ans)
            except Exception as e:
                logging.info(
                    f"[ReviewAgent] LLM did not generate a function with correct format, retry now: {ans}"
                )
                continue
            self.save_rule(code, curr_rule_path)

            if self.get_elapsed_time() >= TIMEOUT_FIRST_REVIEW and not last_rule_path:
                logging.info(
                    f"[ReviewAgent] First iteration timed out when reviewing the code in {curr_rule_path}"
                )
                raise TimeoutException()

            # if self.get_elapsed_time() >= TIMEOUT_PER_REVIEW:
            #     logging.info(f"[ReviewAgent] Time out to review and fix the code in {curr_rule_path}")
            #     exit(1)

        if self.get_elapsed_time() >= TIMEOUT_PER_REVIEW:
            logging.info(
                f"[ReviewAgent] Time out to review the code in {curr_rule_path}"
            )
            
            raise TimeoutException()

        return self.eval_test(curr_rule_path, output_full_res=True)
        
    def eval_val(self, curr_rule_path, output_full_res=False):
        """Evaluate the rule on the validation set."""
        if self.dataset.get_dataset_mode() == "one-by-one":
            if self.mode == "train-combined-fp" or self.mode == "train-combined-fn":
                return self.combined_eval(curr_rule_path, "train")
            else:
                return self.eval(
                    curr_rule_path,
                    self.dataset.get_val_df(),
                    output_full_res=output_full_res,
                )
        elif self.dataset.get_dataset_mode() == "all-in-one":
            if self.mode == "train-combined-fp" or self.mode == "train-combined-fn":
                return self.combined_eval_all_in_one(curr_rule_path, "train")
            else:
                val_dict = self.dataset.get_val_dict()
                return self.eval_all_in_one(curr_rule_path, val_dict)
        else:
            raise ValueError(
                f"Unsupported dataset mode: {self.dataset.get_dataset_mode()}"
            )

    def check_overfitting(self, val_f1, max_consecutive_decreases=3):
        """Check if the model is overfitting based on validation F1 score.

        Returns True if validation F1 has decreased for N consecutive iterations,
        indicating overfitting and that training should be early-stopped.
        """
        if val_f1 < self.best_val_f1:
            self.consecutive_val_decreases += 1
        else:
            self.consecutive_val_decreases = 0
            self.best_val_f1 = val_f1

        is_overfitting = self.consecutive_val_decreases >= max_consecutive_decreases
        if is_overfitting:
            logging.info(
                f"[ReviewAgent] Overfitting detected: validation F1 decreased for "
                f"{self.consecutive_val_decreases} consecutive iterations "
                f"(best={self.best_val_f1:.4f}, current={val_f1:.4f})"
            )
        return is_overfitting

    def collect_regression_samples(self, curr_rule_path, last_rule_path, eval_df, max_samples=3):
        """Collect training samples where the previous rule labeled correctly
        but the new rule labels incorrectly (regression samples).

        Returns a formatted string describing the regression samples for the LLM prompt.
        """
        if last_rule_path is None:
            return ""

        try:
            _, curr_labels, _ = self.eval(curr_rule_path, eval_df, log=False)
            _, last_labels, _ = self.eval(last_rule_path, eval_df, log=False)
        except Exception:
            return ""

        gt_labels = eval_df["label"].values

        # Find indices where prev was correct but curr is wrong
        prev_correct = (last_labels == gt_labels)
        curr_wrong = (curr_labels != gt_labels)
        regression_mask = prev_correct & curr_wrong

        regression_indices = np.where(regression_mask)[0]
        if len(regression_indices) == 0:
            return ""

        # Collect up to max_samples regression windows (chunk around each regression point)
        sample_text = "\n##### REGRESSION SAMPLES\n"
        sample_text += (
            "The following are training samples where the previous rule labeled correctly "
            "but the current rule labels incorrectly. Use these to understand what the "
            "current rule got wrong and fix the regression.\n"
        )

        window_size = min(20, self.chunk_size)
        shown = 0
        shown_ranges = []

        for idx in regression_indices:
            if shown >= max_samples:
                break
            # Skip if this index overlaps with an already-shown window
            start = max(0, idx - window_size // 2)
            end = min(len(eval_df), idx + window_size // 2)
            if any(s <= idx <= e for s, e in shown_ranges):
                continue

            shown_ranges.append((start, end))
            chunk = eval_df.iloc[start:end]

            sample_text += f"\n--- Sample {shown + 1} (indices {start}-{end}) ---\n"
            sample_text += f"Data values: {chunk['value'].values.tolist()}\n"
            sample_text += f"Ground truth labels: {gt_labels[start:end].tolist()}\n"
            sample_text += f"Previous rule labels: {last_labels[start:end].astype(int).tolist()}\n"
            sample_text += f"Current rule labels:  {curr_labels[start:end].astype(int).tolist()}\n"

            # Summarize the specific regression
            fn_count = int(np.sum((curr_labels[start:end] == 0) & (gt_labels[start:end] == 1) & (last_labels[start:end] == 1)))
            fp_count = int(np.sum((curr_labels[start:end] == 1) & (gt_labels[start:end] == 0) & (last_labels[start:end] == 0)))
            if fn_count > 0:
                sample_text += f"  -> {fn_count} false negatives introduced (missed anomalies that previous rule caught)\n"
            if fp_count > 0:
                sample_text += f"  -> {fp_count} false positives introduced (normal points wrongly flagged)\n"

            shown += 1

        sample_text += f"\nTotal regression points: {len(regression_indices)} out of {len(eval_df)} samples\n"
        return sample_text

    def eval_test(self, curr_rule_path, output_full_res=False):
        # TODO: shoud use train or test data to evaluate?
        if self.dataset.get_dataset_mode() == "one-by-one":
            if self.mode == "train-combined-fp" or self.mode == "train-combined-fn":
                return self.combined_eval(curr_rule_path, "test")
            else:
                return self.eval(curr_rule_path, self.dataset.get_test_df(), output_full_res=output_full_res)
        elif self.dataset.get_dataset_mode() == "all-in-one":
            # TODO: support combined evaluation
            if self.mode == "train-combined-fp" or self.mode == "train-combined-fn":
                return self.combined_eval_all_in_one(curr_rule_path, "test")
            else:
                test_dict = self.dataset.get_test_dict()
                return self.eval_all_in_one(curr_rule_path, test_dict)
        else:
            raise ValueError(
                f"Unsupported dataset mode: {self.dataset.get_dataset_mode()}"
            )

    # TODO: add eval metric info in the figure name
    def eval(self, rule_file, eval_df, log=True, output_full_res=False):
        """
        Evaluate the rule on the given dataset

        Args:
            rule_file ([str]): path to the rule file
            eval_df ([pd.DataFrame]): evaluation dataset
        """

        with open(rule_file, "r") as f:
            # BUGFIX: we must only create local env, but not global for parallel evaluation
            rule = f.read()
            local_env = {}
            try:
                exec(rule, local_env)
                # execute_and_cleanup(rule)
                inference_fn = local_env["inference"]
            except Exception as e:
                # cleanup_global_env()
                raise SyntaxException(str(e), rule_file)

        if log:
            logging.info(f"[ReviewAgent] Read following rule from {rule_file}: {rule}")
            logging.info(f"[ReviewAgent] Start to evaluate the rule on test dataset")
        scores = np.zeros(shape=(len(eval_df),), dtype=int)
        # print inference function to see if it is correct
        # logging.info(f"[ReviewAgent] Inference function: {inspect.getsource(inference)}")

        start_time = time.time()
        for i in range(len(eval_df) // self.chunk_size + 1):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, len(eval_df))
            current_data = eval_df[start:end].copy()
            # drop label column
            current_data.drop(columns=["label"], inplace=True)

            # print(inference(current_data.values).shape)
            try:
                raw_labels = run_with_timeout(
                    inference_fn, TIMEOUT_INFERENCE, current_data.values
                )
                # raw_labels = inference(current_data.values)
                format_check(current_data, rule_file, raw_labels)
            except Exception as e:
                # cleanup_global_env()
                raise RuntimeException(str(e), current_data, rule_file)
            scores[start:end] = smooth_labels(raw_labels, window_size=3)
            # labels[start:end] = inference(current_data.values)
            count = np.count_nonzero(scores[start:end])
            # logging.info(f"[ReviewAgent] Number of anomalies in chunk {i}: {count}")
            # labels[start:end] = np.zeros(shape=(self.chunk_size))
        end_time = time.time()
        logging.info(f"[ReviewAgent] Inference time: {end_time - start_time}")

        # cleanup_global_env()

        final_res_dict = self.eval_scores_by_metrics(
            scores, eval_df["label"].values, log
        )
        labels = np.zeros(shape=(len(eval_df),))
        threshold = final_res_dict["event-based f1 under pa with mode squeeze"][
            "threshold"
        ]
        labels[scores >= threshold] = 1
        
        if output_full_res:
            return final_res_dict, labels, scores
        return (
            final_res_dict["event-based f1 under pa with mode squeeze"],
            labels,
            scores,
        )

    def eval_all_in_one(self, rule_file, df_dict):
        # evaluate rule on every df in the dict, report the performance metrics
        scores = []
        gt_labels = []
        for key, df in df_dict.items():
            _, _, curr_scores = self.eval(rule_file=rule_file, eval_df=df, log=False)
            gt_labels.append(df["label"].values)
            scores.append(curr_scores)
        gt_labels = np.concatenate(gt_labels)
        scores = np.concatenate(scores)
        final_res_dict = self.eval_scores_by_metrics(scores, gt_labels)
        labels = np.zeros(shape=(len(gt_labels),))
        threshold = final_res_dict["event-based f1 under pa with mode squeeze"][
            "threshold"
        ]
        labels[scores >= threshold] = 1
        return (
            final_res_dict["event-based f1 under pa with mode squeeze"],
            labels,
            scores,
        )

    def eval_scores_by_metrics(self, scores, gt_labels, log=True):

        report = classification_report(
            gt_labels, scores, labels=[0, 1], zero_division=0
        )
        eval_interface = PointF1()
        eval_res_pf1 = eval_interface.calc(scores, gt_labels, None)

        eval_interface = PointF1PA()
        eval_res_pf1pa = eval_interface.calc(scores, gt_labels, None)

        eval_interface = EventF1PA(mode="squeeze")
        eval_res_ef1pa = eval_interface.calc(scores, gt_labels, None)

        if log:
            logging.info(report)
            logging.info(eval_res_pf1.to_dict())
            logging.info(eval_res_pf1pa.to_dict())
            logging.info(eval_res_ef1pa.to_dict())

        # combine 3 dicts in to a final_res_dict
        final_res_dict = {
            **eval_res_pf1.to_dict(),
            **eval_res_pf1pa.to_dict(),
            **eval_res_ef1pa.to_dict(),
        }
        return final_res_dict

    def combined_eval(self, rule_file, eval_mode="train"):
        """Combined evaluation goes in the following steps: due to necessary preprocessing steps added at EasyTSAD side during training and evaluation

        Step 1: Get model scores and get model labels using model scroes and model threshold. The scores are left aligned with window size (i.e. first sample of window size X goes to score[0], second sample goes to score[1], etc.)
        Step 2: Get rule scores and labels from the rule scores and rule threshold. The label will be postprocessed to align with labels produced on the model size (see preprocess_labels(...) function for detail)
        Step 3: Get ground truth labels from the eval_df, do the same post processing as rule labels.
        Step 4: Combine model labels and rule labels to get combined labels
        Step 5: Calculate performance metrics based on combined labels and ground truth labels. We have made sure our implementation of performance calculation is consistent with EasyTSAD's implementation.

        Args:
            rule_file ([str]): path to the rule file
            eval_df ([pd.DataFrame]): evaluation dataset
            eval_mode ([str]): evaluation mode, either "train" or "test". Neccessary for obtaining model scores and labels. Defaults to "train".
        """

        if eval_mode == "train":
            model_labels = self.dataset.get_model_train_labels()
            eval_df = self.dataset.get_whole_train_df()
        elif eval_mode == "test":
            model_labels = self.dataset.get_model_test_labels()
            eval_df = self.dataset.get_test_df()
        else:
            raise ValueError(f"Unsupported eval mode: {eval_mode}")

        rule_eval_dict, rule_labels, rule_scores = self.eval(rule_file, eval_df)
        # rule_labels = get_rule_labels(rule_eval_dict, rule_scores)

        gt_labels = get_gt_labels(eval_df)
        # model labels may be shorter than rule labels and gt labels
        assert len(rule_labels) == len(gt_labels) and len(model_labels) == len(
            rule_labels
        )
        # rule_labels = rule_labels[:len(model_labels)]
        # gt_labels = gt_labels[:len(model_labels)]
        # assert len(rule_labels) == len(model_labels) == len(gt_labels)
        combined_labels = combine_labels(model_labels, rule_labels, self.mode)
        res_dict = calculate_performance(combined_labels, gt_labels)
        res_dict["rule_path"] = rule_file
        res_dict["threshold"] = rule_eval_dict["threshold"]
        return res_dict, combined_labels, rule_scores

    def combined_eval_all_in_one(self, rule_file, eval_mode="train"):
        # dataset_name -> train_df, test_df
        dataset_dict = self.dataset.get_dataset_dict()
        all_gt_labels = []
        all_combined_labels = []
        all_model_labels = []
        all_rule_scores = []
        for dataset_name, (train_df, test_df, *_rest) in dataset_dict.items():
            if eval_mode == "train":
                model_labels = self.dataset.get_model_train_labels(dataset_name)
                eval_df = train_df
            elif eval_mode == "test":
                model_labels = self.dataset.get_model_test_labels(dataset_name)
                eval_df = test_df
            else:
                raise ValueError(f"Unsupported eval mode: {eval_mode}")
            rule_eval_dict, rule_labels, rule_scores = self.eval(rule_file, eval_df)
            gt_labels = get_gt_labels(eval_df)
            # assert len(rule_labels) == len(gt_labels) and len(model_labels) == len(rule_labels)
            # combined_labels = combine_labels(model_labels, rule_labels, self.mode)
            all_gt_labels.append(gt_labels)
            # all_combined_labels.append(combined_labels)
            all_rule_scores.append(rule_scores)
            all_model_labels.append(model_labels)
        all_gt_labels = np.concatenate(all_gt_labels)
        # all_combined_labels = np.concatenate(all_combined_labels)
        all_rule_scores = np.concatenate(all_rule_scores)
        all_model_labels = np.concatenate(all_model_labels)
        assert len(all_gt_labels) == len(all_model_labels) == len(all_rule_scores)
        final_res_dict = self.eval_scores_by_metrics(all_rule_scores, all_gt_labels)
        all_rule_labels = np.zeros(shape=(len(all_gt_labels),))
        threshold = final_res_dict["event-based f1 under pa with mode squeeze"][
            "threshold"
        ]
        all_rule_labels[all_rule_scores >= threshold] = 1
        all_combined_labels = combine_labels(
            all_model_labels, all_rule_labels, self.mode
        )
        res_dict = calculate_performance(all_combined_labels, all_gt_labels)
        res_dict["rule_path"] = rule_file
        res_dict["threshold"] = threshold
        return res_dict, all_combined_labels, all_rule_scores

    def combined_inference_all_in_one(
        self,
        rule_file_fn=None,
        threshold_fn=0.0,
        rule_file_fp=None,
        threshold_fp=0.0,
        eval_mode="train",
    ):
        # dataset_name -> train_df, test_df
        dataset_dict = self.dataset.get_dataset_dict()
        all_gt_labels = []
        all_model_labels = []
        all_rule_scores_fp = []
        all_rule_scores_fn = []
        for dataset_name, (train_df, test_df, *_rest) in dataset_dict.items():
            if eval_mode == "train":
                model_labels = self.dataset.get_model_train_labels(dataset_name)
                eval_df = train_df
            elif eval_mode == "test":
                model_labels = self.dataset.get_model_test_labels(dataset_name)
                eval_df = test_df
            else:
                raise ValueError(f"Unsupported eval mode: {eval_mode}")
            # rule_eval_dict, rule_labels, rule_scores = self.eval(rule_file, eval_df)
            if rule_file_fp is not None:
                _, _, rule_scores_fp = self.eval(rule_file_fp, eval_df)
                all_rule_scores_fp.append(rule_scores_fp)
            if rule_file_fn is not None:
                _, _, rule_scores_fn = self.eval(rule_file_fn, eval_df)
                all_rule_scores_fn.append(rule_scores_fn)
            gt_labels = get_gt_labels(eval_df)
            all_gt_labels.append(gt_labels)
            all_model_labels.append(model_labels)
        all_gt_labels = np.concatenate(all_gt_labels)
        # all_combined_labels = np.concatenate(all_combined_labels)
        # all_rule_scores_fp = np.concatenate(all_rule_scores_fp)
        # all_rule_scores_fn = np.concatenate(all_rule_scores_fn)
        all_model_labels = np.concatenate(all_model_labels)
        # assert len(all_gt_labels) == len(all_model_labels) == len(all_rule_scores_fp) == len(all_rule_scores_fn)
        all_combined_labels = all_model_labels.copy()
        if rule_file_fp is not None:
            all_rule_scores_fp = np.concatenate(all_rule_scores_fp)
            all_rule_labels_fp = np.zeros(shape=(len(all_gt_labels),))
            final_res_dict = self.eval_scores_by_metrics(
                all_rule_scores_fp, all_gt_labels
            )
            threshold = final_res_dict["event-based f1 under pa with mode squeeze"][
                "threshold"
            ]
            all_rule_labels_fp[all_rule_scores_fp >= threshold] = 1
            all_combined_labels = combine_labels(
                all_combined_labels, all_rule_labels_fp, "train-combined-fp"
            )
        if rule_file_fn is not None:
            all_rule_scores_fn = np.concatenate(all_rule_scores_fn)
            all_rule_labels_fn = np.zeros(shape=(len(all_gt_labels),))
            final_res_dict = self.eval_scores_by_metrics(
                all_rule_scores_fn, all_gt_labels
            )
            threshold = final_res_dict["event-based f1 under pa with mode squeeze"][
                "threshold"
            ]
            all_rule_labels_fn[all_rule_scores_fn >= threshold] = 1
            all_combined_labels = combine_labels(
                all_combined_labels, all_rule_labels_fn, "train-combined-fn"
            )
        res_dict = calculate_performance(all_combined_labels, all_gt_labels)
        res_dict["rule_path_fn"] = rule_file_fn
        # res_dict["threshold_fn"] = threshold_fn
        res_dict["rule_path_fp"] = rule_file_fp
        return res_dict, all_combined_labels

    def get_baseline_performance(self, eval_mode="train"):

        if eval_mode == "train":
            model_labels = self.dataset.get_model_train_labels()
            eval_df = self.dataset.get_whole_train_df()
        elif eval_mode == "test":
            model_labels = self.dataset.get_model_test_labels()
            eval_df = self.dataset.get_test_df()
        else:
            raise ValueError(f"Unsupported eval mode: {eval_mode}")

        gt_labels = get_gt_labels(eval_df)
        assert len(model_labels) == len(gt_labels)
        # np.save("model_labels_left.npy", model_labels)
        # np.save("gt_labels_left.npy", gt_labels)
        # # save eval_df to csv
        # eval_df.to_csv("eval_df_left.csv", index=False)
        res_dict = calculate_performance(model_labels, gt_labels)
        res_dict["rule_path"] = None
        return res_dict, None

    def get_baseline_performance_all_in_one(self, eval_mode="train"):
        dataset_dict = self.dataset.get_dataset_dict()
        all_gt_labels = []
        all_model_labels = []
        for dataset_name, (train_df, test_df, *_rest) in dataset_dict.items():
            if eval_mode == "train":
                model_labels = self.dataset.get_model_train_labels(dataset_name)
                eval_df = train_df
            elif eval_mode == "test":
                model_labels = self.dataset.get_model_test_labels(dataset_name)
                eval_df = test_df
            else:
                raise ValueError(f"Unsupported eval mode: {eval_mode}")
            gt_labels = get_gt_labels(eval_df)
            assert len(model_labels) == len(gt_labels)
            all_gt_labels.append(gt_labels)
            all_model_labels.append(model_labels)
        all_gt_labels = np.concatenate(all_gt_labels)
        all_model_labels = np.concatenate(all_model_labels)
        assert len(all_gt_labels) == len(all_model_labels)
        res_dict = calculate_performance(all_model_labels, all_gt_labels)
        res_dict["rule_path"] = None
        return res_dict, None

    def combined_inference(
        self,
        rule_file_fn=None,
        threshold_fn=0.0,
        rule_file_fp=None,
        threshold_fp=0.0,
        eval_mode="train",
    ):
        if eval_mode == "train":
            model_labels = self.dataset.get_model_train_labels()
            eval_df = self.dataset.get_whole_train_df()
        elif eval_mode == "test":
            model_labels = self.dataset.get_model_test_labels()
            eval_df = self.dataset.get_test_df()
        else:
            raise ValueError(f"Unsupported eval mode: {eval_mode}")

        gt_labels = get_gt_labels(eval_df)

        assert len(model_labels) == len(gt_labels) == len(eval_df)

        # copy model_labels to combined_labels
        combined_labels = model_labels.copy()

        if rule_file_fp is not None:
            rule_eval_dict, rule_labels, _ = self.eval(rule_file_fp, eval_df)
            # rule_labels = get_rule_labels(rule_eval_dict, rule_scores)
            combined_labels = combine_labels(
                combined_labels, rule_labels, "train-combined-fp"
            )
        if rule_file_fn is not None:
            rule_eval_dict, rule_labels, _ = self.eval(rule_file_fn, eval_df)
            # rule_labels = get_rule_labels(rule_eval_dict, rule_scores)
            combined_labels = combine_labels(
                combined_labels, rule_labels, "train-combined-fn"
            )

        res_dict = calculate_performance(combined_labels, gt_labels)
        res_dict["rule_path_fn"] = rule_file_fn
        # res_dict["threshold_fn"] = threshold_fn
        res_dict["rule_path_fp"] = rule_file_fp
        # res_dict["threshold_fp"] = threshold_fp
        return res_dict, combined_labels
