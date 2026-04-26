import json
import logging
import os
import pickle
import pprint
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent.agent import MAX_ITER, TIMEOUT
from agent.detection_agent import DetectionAgentV3
from agent.image_agent import ImageAgent
from agent.mutate_agent import MutateAgent
from agent.repair_agent import RepairAgent
from agent.review_agent import ReviewAgent
from baseline.llmad import LLMAD
from common.common import calculate_performance
from common.exception import (RuntimeException, SyntaxException,
                              TimeoutException)
from datasets.dataset import ArgosDataset
from selector.train_perf_selector import TrainPerfSelector
import threading

DEFAULT_PARALLEL_CORE = 2

class Engine(ABC):
    def __init__(
        self,
        dataset_path,
        chunk_size=1000,
        image_chunk_size=10000,
        image_subplots=(1, 1),
        rule_path="/tmp",
        model_res_path=None,
        mode="train-LLM-only",
        dataset_mode="one-by-one",
        train_test_split=0.7,
        top_k=1,
        llm_engine="gpt-4o",
        timeout=150,
        sample_per_prompt=1,
        parallel_core=DEFAULT_PARALLEL_CORE,
        rule_per_group=3,
        max_iter=None,
    ):
        self.chunk_size = chunk_size
        self.image_chunk_size = image_chunk_size
        self.image_subplots = image_subplots
        self.normal_chunk_size = chunk_size
        self.dataset_path = dataset_path
        self.mode = mode
        self.dataset_mode = dataset_mode
        self.train_test_split = train_test_split
        self.top_k = top_k
        self.llm_engine = llm_engine
        self.sample_per_prompt = sample_per_prompt
        self.parallel_core = parallel_core
        self.rule_per_group = rule_per_group
        # for threading
        self.lock = threading.Lock()

        # set random seed
        np.random.seed(8)

        if mode == "train-combined-fp" or mode == "train-combined-fn":
            assert (
                model_res_path is not None
            ), "Model result path must be provided for combined training mode"
        self.model_res_path = model_res_path

        self.cur_iter = 0
        self.max_iter = max_iter if max_iter is not None else MAX_ITER
        self.max_iter_override = max_iter

        # in the unit of minute
        self.max_time = timeout * 60
        print(f"Timeout: {timeout} minutes")

        self.rule_path = rule_path
        self.update_rule_path_prefix()
        os.makedirs(self.rule_path, exist_ok=True)
        self.rule_file_base = os.path.join(self.rule_path, "rule")

        curve_name = ""

        self.dataset = ArgosDataset(
            dataset_path=dataset_path,
            dataset_mode=dataset_mode,
            engine_mode=mode,
            chunk_size=chunk_size,
            image_chunk_size=image_chunk_size,
            train_test_split=train_test_split,
            model_res_path=model_res_path,
        )
        # save logs to file
        logging.basicConfig(
            filename=os.path.join(self.rule_path, "output.log"),
            level=logging.INFO,
            filemode="a",
        )

        if self.mode == "baseline-LLMAD":
            self.LLMAD = LLMAD(
                chunk_size=self.chunk_size, mode=self.mode, llm_engine=self.llm_engine
            )
            return

        print(f"Using LLM engine: {llm_engine}")

        if self.mode == "train-evolution":
            timeout *= self.parallel_core
        elif self.mode == "train-LLM-only-parallel":
            timeout *= self.top_k

    
        self.detection_agent = DetectionAgentV3(
            chunk_size=self.chunk_size,
            mode=self.mode,
            llm_engine=self.llm_engine,
            timeout=timeout,
        )

        self.repair_agent = RepairAgent(
            chunk_size=self.chunk_size, llm_engine=self.llm_engine,
            timeout=timeout
        )
        self.review_agent = ReviewAgent(
            chunk_size=self.chunk_size,
            dataset=self.dataset,
            dataset_name=curve_name,
            model_res_path=self.model_res_path,
            mode=self.mode,
            llm_engine=self.llm_engine,
            timeout=timeout
        )

        self.mutate_agent = MutateAgent(
            chunk_size=self.chunk_size, 
            llm_engine=self.llm_engine, 
            timeout=timeout
        )

        if self.mode == "train-LLM-only-image":
            self.image_agent = ImageAgent(
                chunk_size=self.image_chunk_size, mode=self.mode, llm_engine="gpt-4o"
            )

    def generate_mapping(self, dir_path: str):
        files = []
        for root, _, filenames in os.walk(dir_path):
            for filename in filenames:
                if "mutated" in filename:
                    files.append(os.path.join(root, filename))
        
        mapping = {}
        for file in files:
            parts = file.split("_mutated_")
            index = int(parts[1].split(".py")[0])
            mapping[index] = file

        return mapping


    def run_mutate_eval_thread(self, 
                               start_time,
                               pipeline_id,
                               best_rule_path,
                               curr_dfs):
        mutated_rule_path = best_rule_path.replace(".py", f"_mutated_{pipeline_id}.py")
        self.repair_agent.run(curr_dfs[0], mutated_rule_path)
        success = False
        while time.time() - start_time < self.max_time:
            try:
                eval_res, labels, _ = self.review_agent.run(mutated_rule_path, best_rule_path)
                success = True
                break  # Exit loop on success
            except Exception as e:
                logging.info(f"[TrainingEngine]: ReviewAgent error: {e}. Attempting repair...")
                if isinstance(e, (SyntaxException, RuntimeException)):
                    self.repair_agent.run(e.df if hasattr(e, "df") else curr_dfs[0], e.rule_path)
                elif isinstance(e, TimeoutException):
                    logging.info("[TrainingEngine]: ReviewAgent timed out, skipping iteration.")
                    success = False
                    break   
                else:
                    raise e

        if self.dataset_mode == "one-by-one":
            train_eval_res, _, _ = self.review_agent.eval(mutated_rule_path, self.dataset.get_train_df())
            test_eval_res, _, _ = self.review_agent.eval_test(mutated_rule_path)
            # Evaluate on validation set for rule selection
            try:
                val_eval_res, _, _ = self.review_agent.eval_val(mutated_rule_path)
            except Exception:
                val_eval_res = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.0}
        else:
            raise ValueError(f"Invalid dataset mode {self.dataset_mode}")
        final_train_res_path = mutated_rule_path.replace(".py", "_eval_res_train.json")
        with open(final_train_res_path, "w") as f:
            json.dump(train_eval_res, f)

        final_val_res_path = mutated_rule_path.replace(".py", "_eval_res_val.json")
        with open(final_val_res_path, "w") as f:
            json.dump(val_eval_res, f)

        final_test_res_path = mutated_rule_path.replace(".py", "_eval_res_test.json")
        with open(final_test_res_path, "w") as f:
            json.dump(test_eval_res, f)

    def run_iteration_thread(
        self, pipeline_id, start_time, curr_dfs, current_iteration_dir_path, rule_perf_pairs, last_rule_path_in=None
    ):
        last_rule_path = last_rule_path_in
        
        rule_file_path = os.path.join(current_iteration_dir_path, f"rule_{pipeline_id}_0.py")
        
        self.detection_agent.run(
            curr_dfs=curr_dfs,
            curr_rule_path=rule_file_path,
            last_rule_path=last_rule_path,
            image_path=None,
        )
        self.repair_agent.run(curr_dfs[0], rule_file_path)
        # save from the current rule_path to the temp_res_path
        # temp_res_path = rule_file_path.replace(".py", "_temp.py")
        # os.system(f"cp {rule_file_path} {temp_res_path}")
        eval_res = None
        while time.time() - start_time < self.max_time:
            try:
                eval_res, _, _ = self.review_agent.run(rule_file_path, last_rule_path)
                break  # Exit loop on success
            except Exception as e:
                logging.info(f"[TrainingEngine]: ReviewAgent error: {e}. Attempting repair...")
                if isinstance(e, (SyntaxException, RuntimeException)):
                    self.repair_agent.run(e.df if hasattr(e, "df") else curr_dfs[0], e.rule_path)
                elif isinstance(e, TimeoutException):
                    logging.info("[TrainingEngine]: ReviewAgent timed out, skipping iteration.")            
                    eval_res = None
                    break
                else:
                    raise e  # Rethrow unknown exceptions

        if eval_res is None:
            # copy back the original rule when available; on the first iteration
            # there may be no previous rule to restore, so keep the current rule.
            # os.system(f"cp {temp_res_path} {rule_file_path}")
            if last_rule_path is not None:
                os.system(f"cp {last_rule_path} {rule_file_path}")
            else:
                logging.info(
                    f"[TrainingEngine]: No previous rule available for fallback; keeping current rule {rule_file_path}."
                )
            try:
                eval_res, _ , _ = self.review_agent.eval_test(rule_file_path, output_full_res=True)
            except Exception as e:
                eval_res = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.0}
            logging.info("ReviewAgent time out for rule", rule_file_path)
        # os.remove(temp_res_path)
        final_res_path = rule_file_path.replace(".py", "_eval_res_test.json")
        last_rule_path = rule_file_path
        with open(final_res_path, "w") as f:
            json.dump(eval_res, f)
        if self.dataset_mode == "one-by-one":
            try:
                train_eval_res, _, _ = self.review_agent.eval(rule_file_path, self.dataset.get_train_df(), output_full_res=True)
            except Exception as e:
                train_eval_res = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.0}
            # Evaluate on validation set for rule selection
            try:
                val_eval_res, _, _ = self.review_agent.eval_val(rule_file_path, output_full_res=True)
            except Exception as e:
                val_eval_res = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.0}
        else:
            raise ValueError(f"Invalid dataset mode {self.dataset_mode}")
        final_train_res_path = rule_file_path.replace(".py", "_eval_res_train.json")
        with open(final_train_res_path, "w") as f:
            json.dump(train_eval_res, f)
        # Save validation results for rule selection
        final_val_res_path = rule_file_path.replace(".py", "_eval_res_val.json")
        with open(final_val_res_path, "w") as f:
            json.dump(val_eval_res, f)
        with self.lock:
            rule_perf_pairs.append((rule_file_path, final_val_res_path))

    def run_evolution_mode(self):
        self.cur_iter = 0
        start_time = time.time()

        if self.max_iter_override is None:
            self.max_iter = 50
        mapping_last_rule_path = {i: None for i in range(self.parallel_core)}
        rule_perf_pairs = []
        
        for agent in [self.detection_agent, self.repair_agent, self.review_agent, self.mutate_agent]:
                agent.set_start_time()

        while self.cur_iter < self.max_iter and time.time() - start_time < self.max_time:
            logging.info(f"[TrainingEngine]: === Rule Iteration {self.cur_iter} ===")

            # Generate random datasets for sampling
            curr_dfs = [self.dataset.get_train_df_by_iter(np.random.randint(0, 1000)) for _ in range(self.sample_per_prompt)]

            iteration_rule_path = os.path.join(self.rule_path, f"iter_{self.cur_iter}")
            if(self.cur_iter == 0):
                os.makedirs(iteration_rule_path, exist_ok=True)

            # Launch parallel threads
            threads = []
            for pipeline_id in range(self.parallel_core):
                thread = threading.Thread(
                    target=self.run_iteration_thread,
                    args=(
                        pipeline_id,
                        start_time,
                        curr_dfs,
                        iteration_rule_path,
                        rule_perf_pairs,
                        mapping_last_rule_path[pipeline_id],
                    ),
                )
                threads.append(thread)
                thread.start()

            # Wait for all threads to finish
            for thread in threads:
                thread.join()

            # Select the best rule based on performance
            topk_selector = TrainPerfSelector(rule_perf_pairs)
            logging.info("Rule performance pairs: %s", rule_perf_pairs)
            logging.info("prev_best_rule_path: %s", mapping_last_rule_path)
            best_rules_path = topk_selector.select_k_top_rule(self.rule_per_group, mapping_last_rule_path)
            rule_perf_pairs.clear()
            # we do mutation between 2 iterations
            if self.cur_iter % 2 == 1: 

                # Mutate the best rule (-1 to keep the original rule)
                # self.mutate_agent.run(best_rule_path, self.parallel_core - 1)
                self.mutate_agent.run_with_top_pick(best_rules_path, self.parallel_core, True)
                self.mutate_agent.populate_missing_mutated_item(best_rules_path, self.parallel_core)

                # Prepare for the next iteration
                mapping_last_rule_path = self.generate_mapping(iteration_rule_path)

                # # evaluate the mutated rules
                # threads = []
                # for i in range(self.parallel_core):
                #     mutate_rule_path = mapping_last_rule_path[i]
                #     original_rule_path = mutate_rule_path.replace(f"_mutated_{i}.py",".py")
                #     # mutated_rule_path = best_rule_path.replace(".py", f"_mutated_{i}.py")
                #     thread = threading.Thread(
                #         target=self.run_mutate_eval_thread,
                #         args=(start_time, i, original_rule_path, curr_dfs),
                #     )

                #     threads.append(thread)
                #     thread.start()

                # for thread in threads:
                #     thread.join()

                next_iteration_rule_path = os.path.join(self.rule_path, f"iter_{self.cur_iter + 1}")
                os.makedirs(next_iteration_rule_path, exist_ok=True)

                # copy all the mutated rules to the next iteration
                for i in range(self.parallel_core):
                    mutated_rule_path = mapping_last_rule_path[i]
                    # mutated_rule_path = best_rule_path.replace(".py", f"_mutated_{i}.py")
                    # eval_res_path_test = mutated_rule_path.replace(".py", "_eval_res_test.json")
                    # eval_res_path_train = mutated_rule_path.replace(".py", "_eval_res_train.json")
                    base_name = f"rule_{i}_{0}.py"
                    # base_name_eval_res_test = f"rule_{i}_{0}_eval_res_test.json"
                    # base_name_eval_res_train = f"rule_{i}_{0}_eval_res_train.json"
                    os.system(f"cp {mutated_rule_path} {os.path.join(next_iteration_rule_path, base_name)}")
                    # os.system(f"cp {eval_res_path_test} {os.path.join(next_iteration_rule_path, base_name_eval_res_test)}")
                    # os.system(f"cp {eval_res_path_train} {os.path.join(next_iteration_rule_path, base_name_eval_res_train)}")
                    # rule_perf_pairs.append((os.path.join(next_iteration_rule_path, base_name), os.path.join(next_iteration_rule_path, base_name_eval_res_train)))
                    mapping_last_rule_path[i] = os.path.join(next_iteration_rule_path, base_name)
            else:
                next_iteration_rule_path = os.path.join(self.rule_path, f"iter_{self.cur_iter + 1}")
                os.makedirs(next_iteration_rule_path, exist_ok=True)
                # this iteration there is no mutation, we just copy the top-k rules to new pipelines
                for i in range(self.parallel_core):
                    index = i % len(best_rules_path)
                    rule_path = best_rules_path[index]
                    # eval_res_path_test = rule_path.replace(".py", "_eval_res_test.json")
                    # eval_res_path_train = rule_path.replace(".py", "_eval_res_train.json")
                    base_name = f"rule_{i}_{0}.py"
                    # base_name_eval_res_test = f"rule_{i}_{0}_eval_res_test.json"
                    # base_name_eval_res_train = f"rule_{i}_{0}_eval_res_train.json"
                    os.system(f"cp {rule_path} {os.path.join(next_iteration_rule_path, base_name)}")
                    # os.system(f"cp {eval_res_path_test} {os.path.join(next_iteration_rule_path, base_name_eval_res_test)}")
                    # os.system(f"cp {eval_res_path_train} {os.path.join(next_iteration_rule_path, base_name_eval_res_train)}")
                    # rule_perf_pairs.append((os.path.join(next_iteration_rule_path, base_name), os.path.join(next_iteration_rule_path, base_name_eval_res_train)))
                    mapping_last_rule_path[i] = os.path.join(next_iteration_rule_path, base_name)
            
            # copy original one
            # base_name = f"rule_{self.parallel_core - 1}_{0}.py"
            # base_name_eval_res_test = f"rule_{self.parallel_core - 1}_{0}_eval_res_test.json"
            # base_name_eval_res_train = f"rule_{self.parallel_core - 1}_{0}_eval_res_train.json"
            # os.system(f"cp {best_rule_path} {os.path.join(next_iteration_rule_path, base_name)}")
            # os.system(f"cp {best_rule_path.replace('.py', '_eval_res_test.json')} {os.path.join(next_iteration_rule_path, base_name_eval_res_test)}")
            # os.system(f"cp {best_rule_path.replace('.py', '_eval_res_train.json')} {os.path.join(next_iteration_rule_path, base_name_eval_res_train)}")
            # rule_perf_pairs.append((os.path.join(next_iteration_rule_path, base_name), os.path.join(next_iteration_rule_path, base_name_eval_res_train)))
            
            logging.info(f"[TrainingEngine]: Best rule path at iteration {self.cur_iter}: {best_rules_path}")

            # Check for overfitting using validation set on the best rule
            try:
                best_val_res, _, _ = self.review_agent.eval_val(best_rules_path[0])
                val_f1 = best_val_res.get("f1", 0.0)
                if self.review_agent.check_overfitting(val_f1):
                    logging.info(
                        f"[TrainingEngine]: Early stopping at iteration {self.cur_iter} "
                        f"due to overfitting (validation F1 decreased for consecutive iterations)"
                    )
                    break
            except Exception as e:
                logging.info(f"[TrainingEngine]: Validation eval failed: {e}, continuing training")

            for rule_path in best_rules_path:
                with open(os.path.join(self.rule_path, "best_rule_path.txt"), "a", buffering=1) as f:
                    f.write(rule_path + "\n")

            self.cur_iter += 1

        self.wrapup_run(start_time, None)

    def run_normal_mode_thread(self, 
                               id,
                               start_time,
                               curr_dfs,
                               last_rule_path,
                               rule_perf_pairs):
        self.detection_agent.run(
            curr_dfs=curr_dfs,
            curr_rule_path=self.get_rule_path(top_k_curr=id),
            last_rule_path=last_rule_path,
            image_path=None
        )

        self.repair_agent.run(curr_dfs[0], self.get_rule_path(top_k_curr=id))

        eval_res = None
        while time.time() - start_time < self.max_time:
            try:
                eval_res, labels, _ = self.review_agent.run(self.get_rule_path(top_k_curr=id), last_rule_path)
                break  # Exit loop on success
            except Exception as e:
                logging.info(f"[TrainingEngine]: ReviewAgent error: {e}. Attempting repair...")

                if isinstance(e, (SyntaxException, RuntimeException)):
                    self.repair_agent.run(e.df if hasattr(e, "df") else curr_dfs[0], e.rule_path)
                elif isinstance(e, TimeoutException):
                    logging.info("[TrainingEngine]: ReviewAgent timed out, skipping iteration.")
                    eval_res = None
                    break
                else:
                    raise e
                
        final_res_path = self.get_rule_path(top_k_curr=id).replace(".py", "_eval_res_test.json")

        if eval_res:
            with open(final_res_path, "w") as f:
                json.dump(eval_res, f)
        else:
            if last_rule_path is not None:
                os.system(f"cp {last_rule_path} {self.get_rule_path(top_k_curr=id)}")
            else:
                logging.info("[TrainingEngine]: No previous rule available for fallback copy; evaluating current rule as-is.")
            eval_res, _, _ = self.review_agent.eval_test(
                self.get_rule_path(top_k_curr=id), output_full_res=True
            )
            with open(final_res_path, "w") as f:
                json.dump(eval_res, f)

        if self.dataset_mode == "one-by-one":
            if self.mode == "train-combined-fn" or self.mode == "train-combined-fp":
                train_eval_res, _, _ = self.review_agent.combined_eval(self.get_rule_path(top_k_curr=id), "train")
            else:
                train_eval_res, _, _ = self.review_agent.eval(self.get_rule_path(top_k_curr=id), self.dataset.get_train_df(), output_full_res=True)
        elif self.dataset_mode == "all-in-one":
            if self.mode == "train-combined-fn" or self.mode == "train-combined-fp":
                train_eval_res, _, _ = self.review_agent.combined_eval_all_in_one(self.get_rule_path(top_k_curr=id), eval_mode="train")
            else:
                train_dict = self.dataset.get_train_dict()
                train_eval_res, _, _ = self.review_agent.eval_all_in_one(self.get_rule_path(top_k_curr=id), train_dict)
        else:
            raise ValueError(f"Invalid dataset mode {self.dataset_mode}")
        final_train_res_path = self.get_rule_path(top_k_curr=id).replace(".py", "_eval_res_train.json")

        with open(final_train_res_path, "w") as f:
            json.dump(train_eval_res, f)

        # Evaluate on validation set for rule selection
        try:
            val_eval_res, _, _ = self.review_agent.eval_val(self.get_rule_path(top_k_curr=id), output_full_res=True)
        except Exception:
            val_eval_res = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.0}
        final_val_res_path = self.get_rule_path(top_k_curr=id).replace(".py", "_eval_res_val.json")
        with open(final_val_res_path, "w") as f:
            json.dump(val_eval_res, f)

        with self.lock:
            rule_perf_pairs.append((self.get_rule_path(top_k_curr=id), final_val_res_path))

    def run(self):
        if self.mode == "train-evolution":
            self.run_evolution_mode()
            return
        if self.mode == "baseline-LLMAD":
            self.run_baseline()
            return
        if self.mode == "train-combined-fn" or self.mode == "train-combined-fp":
            if self.dataset.get_skip_training():
                logging.info(f"[TrainingEngine]: Skip training for {self.mode}")
                return
        if self.mode == "ablation-detection-only":
            self.ablation_detection_only()
            return
        self.update_rule_path_prefix()

        # Step 1: image understanding
        if self.mode == "train-LLM-only-image":
            self.cur_iter = 0
            start_time = time.time()
            num_chunks = len(self.dataset.get_train_dict())
            self.image_agent.set_start_time()
            max_image_iter = num_chunks * 5
            last_anomaly_types = None
            while (
                self.cur_iter < max_image_iter
                and time.time() - start_time < self.max_time
            ):
                logging.info(
                    f"[TrainingEngine]: === Image Iteration {self.cur_iter} ==="
                )
                # start = (self.cur_iter % num_chunks) * self.image_chunk_size
                # end = min(((self.cur_iter % num_chunks) + 1) * self.image_chunk_size, len(self.train_df))
                # curr_df = self.train_df[start:end]
                curr_df = self.dataset.get_train_image_df_by_iter(self.cur_iter)
                figure_path = os.path.join(
                    self.rule_path, f"anomaly_types_figure_{self.cur_iter}.png"
                )
                labels = np.zeros(len(curr_df))
                self.visualize(
                    curr_df[["value", "label", "index"]].values,
                    labels,
                    figure_path,
                    self.image_subplots,
                )
                try:
                    last_anomaly_types = self.image_agent.run(
                        figure_path, last_anomaly_types
                    )
                except Exception as e:
                    if isinstance(e, TimeoutException):
                        logging.info(
                            f"[TrainingEngine]: ImageAgent times out, skip this iteration"
                        )
                        continue
                    else:
                        raise e
                self.cur_iter += 1
            logging.info(
                f"[TrainingEngine]: Image training finishes after {self.cur_iter} iterations and {(time.time() - start_time)/60} minutes"
            )
            logging.info(f"[TrainingEngine]: Final anomaly types: {last_anomaly_types}")
            final_anomaly_types = last_anomaly_types

        # Step 2: rule training

        self.cur_iter = 0
        last_rule_paths = [None for _ in range(self.top_k)]

        start_time = time.time()

        self.detection_agent.set_start_time()
        self.repair_agent.set_start_time()
        self.review_agent.set_start_time()

        # terminated = False

        while (
            self.cur_iter < self.max_iter and time.time() - start_time < self.max_time
        ):
            logging.info(f"[TrainingEngine]: === Rule Iteration {self.cur_iter} ===")
            skip = False

            # start = (self.cur_iter % num_chunks) * self.chunk_size
            # end = min(((self.cur_iter % num_chunks) + 1) * self.chunk_size, len(self.train_df))

            # curr_df = self.train_df[start:end]

            # pick a random int
            curr_dfs = []

            for i in range(self.sample_per_prompt):
                rand_num = np.random.randint(0, 1000)
                curr_df = self.dataset.get_train_df_by_iter(rand_num)
                curr_dfs.append(curr_df)

            rule_perf_pairs = []
            threads = []
            for top_k_curr in range(self.top_k):
                if self.mode == "train-combined-fn":
                    # if self.normal_df is not None:
                    #     normal_chunks = len(self.normal_df) // self.normal_chunk_size
                    #     normal_start = (self.cur_iter % normal_chunks) * self.normal_chunk_size
                    #     normal_end = min(((self.cur_iter % normal_chunks) + 1) * self.normal_chunk_size, len(self.normal_df))
                    #     normal_df = self.normal_df[normal_start:normal_end]
                    # else:
                    #     normal_df = None
                    normal_dfs = []
                    for i in range(len(curr_dfs)):
                        if self.dataset_mode == "one-by-one":
                            normal_df = self.dataset.get_normal_df_by_iter(
                                self.cur_iter
                            )
                        elif self.dataset_mode == "all-in-one":
                            normal_df = self.dataset.get_closest_normal_df(curr_dfs[i])
                        normal_dfs.append(normal_df)
                    self.detection_agent.run(
                        curr_dfs,
                        self.get_rule_path(top_k_curr=top_k_curr),
                        last_rule_paths[top_k_curr],
                        normal_dfs,
                    )
                elif self.mode == "train-combined-fp":
                    # if self.abnormal_df is not None:
                    #     abnormal_chunks = len(self.abnormal_df) // self.chunk_size
                    #     abnormal_start = (self.cur_iter % abnormal_chunks) * self.chunk_size
                    #     abnormal_end = min(((self.cur_iter % abnormal_chunks) + 1) * self.chunk_size, len(self.abnormal_df))
                    #     abnormal_df = self.abnormal_df[abnormal_start:abnormal_end]
                    # else:
                    #     abnormal_df = None
                    abnormal_dfs = []
                    for i in range(len(curr_dfs)):
                        if self.dataset_mode == "one-by-one":
                            abnormal_df = self.dataset.get_abnormal_df_by_iter(
                                self.cur_iter
                            )
                        elif self.dataset_mode == "all-in-one":
                            abnormal_df = self.dataset.get_closest_abnormal_df(
                                curr_dfs[i]
                            )
                        abnormal_dfs.append(abnormal_df)
                    self.detection_agent.run(
                        curr_dfs,
                        self.get_rule_path(top_k_curr=top_k_curr),
                        last_rule_paths[top_k_curr],
                        abnormal_dfs,
                    )
                elif self.mode == "train-LLM-only-image":
                    self.detection_agent.run(
                        curr_dfs=curr_dfs[0],
                        curr_rule_path=self.get_rule_path(top_k_curr=top_k_curr),
                        last_rule_path=last_rule_paths[top_k_curr],
                        anomaly_types=final_anomaly_types,
                    )
                elif self.mode == "train-LLM-only-parallel":
                    thread = threading.Thread(
                    target=self.run_normal_mode_thread,
                    args=(
                        top_k_curr,
                        start_time,
                        curr_dfs,
                        last_rule_paths[top_k_curr],
                        rule_perf_pairs,
                    ),
                    )
                    threads.append(thread)
                    thread.start()
                    continue
                else:
                    # training_figure_path = os.path.join(self.rule_path, f"training_figure_{self.cur_iter}.png")
                    # labels = np.zeros(len(curr_df))
                    # self.visualize(curr_df[["value", "label", "index"]].values, labels, training_figure_path)
                    self.detection_agent.run(
                        curr_dfs=curr_dfs,
                        curr_rule_path=self.get_rule_path(top_k_curr=top_k_curr),
                        last_rule_path=last_rule_paths[top_k_curr],
                        image_path=None,
                    )
                print("Detection done HOHOHOHOHO")
                self.repair_agent.run(
                    curr_dfs[0], self.get_rule_path(top_k_curr=top_k_curr)
                )
                while True and time.time() - start_time < self.max_time:
                    try:
                        eval_res, labels, _ = self.review_agent.run(
                            self.get_rule_path(top_k_curr=top_k_curr), last_rule_paths[top_k_curr]
                        )
                        break
                    except Exception as e:
                        logging.info(
                            f"[TrainingEngine]: ReviewAgent returns with following exception {e}, trying to fix..."
                        )
                        if isinstance(e, SyntaxException):
                            self.repair_agent.run(curr_dfs[0], e.rule_path)
                        elif isinstance(e, RuntimeException):
                            self.repair_agent.run(e.df, e.rule_path)
                        elif isinstance(e, TimeoutException):
                            logging.info(
                                f"[TrainingEngine]: ReviewAgent times out, skip this iteration"
                            )
                            skip = True
                            break
                        else:
                            raise e
                if skip:
                    break
                final_res_path = self.get_rule_path(top_k_curr=top_k_curr).replace(
                    ".py", "_eval_res_test.json"
                )
                with open(final_res_path, "w") as f:
                    json.dump(eval_res, f)

                # Also get train_df performance
                if self.dataset_mode == "one-by-one":
                    if (
                        self.mode == "train-combined-fn"
                        or self.mode == "train-combined-fp"
                    ):
                        train_eval_res, _, _ = self.review_agent.combined_eval(
                            self.get_rule_path(top_k_curr=top_k_curr), "train"
                        )
                    else:
                        train_eval_res, _, _ = self.review_agent.eval(
                            self.get_rule_path(top_k_curr=top_k_curr),
                            self.dataset.get_train_df(),
                            output_full_res=True,
                        )
                elif self.dataset_mode == "all-in-one":
                    if (
                        self.mode == "train-combined-fn"
                        or self.mode == "train-combined-fp"
                    ):
                        train_eval_res, _, _ = (
                            self.review_agent.combined_eval_all_in_one(
                                self.get_rule_path(top_k_curr=top_k_curr),
                                eval_mode="train",
                            )
                        )
                    else:
                        train_dict = self.dataset.get_train_dict()
                        train_eval_res, _, _ = self.review_agent.eval_all_in_one(
                            self.get_rule_path(top_k_curr=top_k_curr), train_dict
                        )
                else:
                    raise ValueError(f"Invalid dataset mode {self.dataset_mode}")
                final_train_res_path = self.get_rule_path(
                    top_k_curr=top_k_curr
                ).replace(".py", "_eval_res_train.json")
                with open(final_train_res_path, "w") as f:
                    json.dump(train_eval_res, f)

                # Evaluate on validation set for rule selection
                try:
                    val_eval_res, _, _ = self.review_agent.eval_val(
                        self.get_rule_path(top_k_curr=top_k_curr), output_full_res=True
                    )
                except Exception:
                    val_eval_res = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.0}
                final_val_res_path = self.get_rule_path(
                    top_k_curr=top_k_curr
                ).replace(".py", "_eval_res_val.json")
                with open(final_val_res_path, "w") as f:
                    json.dump(val_eval_res, f)

                # Use validation performance for rule selection
                rule_perf_pairs.append(
                    (self.get_rule_path(top_k_curr=top_k_curr), final_val_res_path)
                )

            if self.mode == "train-LLM-only-parallel":
                for thread in threads:
                    thread.join()

            if skip:
                logging.info(
                    f"[TrainingEngine]: Review Agent timeout, restart iteration {self.cur_iter} and pick another data"
                )
                continue

            topk_selector = TrainPerfSelector(rule_perf_pairs)
            best_rule_paths = topk_selector.select_k_top_rule(self.rule_per_group)
            rule_perf_pairs.clear()

            logging.info(
                f"[TrainingEngine]: Best rule path at iteration {self.cur_iter}: {best_rule_paths}"
            )

            # Check for overfitting using validation set on the best rule
            try:
                best_val_res, _, _ = self.review_agent.eval_val(best_rule_paths[0])
                val_f1 = best_val_res.get("f1", 0.0)
                if self.review_agent.check_overfitting(val_f1):
                    logging.info(
                        f"[TrainingEngine]: Early stopping at iteration {self.cur_iter} "
                        f"due to overfitting (validation F1 decreased for consecutive iterations)"
                    )
                    break
            except Exception as e:
                logging.info(f"[TrainingEngine]: Validation eval failed: {e}, continuing training")

            for rule_path in best_rule_paths:
                with open(os.path.join(self.rule_path, "best_rule_path.txt"), "a", buffering=1) as f:
                    f.write(rule_path + "\n")

            last_rule_paths = []
            # do a round-robin mapping
            for i in range(self.top_k):
                last_rule_paths.append(best_rule_paths[i % len(best_rule_paths)])

            # if self.dataset.get_dataset_mode() == "one-by-one":
            #     if self.mode == "train-combined-fn" or self.mode == "train-combined-fp":
            #         eval_res, labels, _ = self.review_agent.combined_eval(
            #             best_rule_path, eval_mode="test"
            #         )
            #     else:
            #         eval_res, labels, _ = self.review_agent.eval(
            #             best_rule_path, self.dataset.get_test_df()
            #         )

            #     prefix = best_rule_path.split("/")[-1].split(".")[0]
            #     figure_path = os.path.join(self.rule_path, f"eval_figure_{prefix}.png")
            #     self.visualize(
            #         self.dataset.get_test_df()[["value", "label", "index"]].values,
            #         labels,
            #         figure_path,
            #     )
            # elif self.dataset.get_dataset_mode() == "all-in-one":
            #     # TODO: implement combine eval for all-in-one mode

            #     prefix = best_rule_path.split("/")[-1].split(".")[0]
            #     figure_path = os.path.join(self.rule_path, f"eval_figure_{prefix}.png")
            #     if self.mode == "train-combined-fn" or self.mode == "train-combined-fp":
            #         eval_res, labels, _ = self.review_agent.combined_eval_all_in_one(
            #             best_rule_path, eval_mode="test"
            #         )
            #         dataset_dict = self.dataset.get_dataset_dict()
            #         test_df = []
            #         for _, (_, df) in dataset_dict.items():
            #             test_df.append(df)
            #         test_df = pd.concat(test_df)
            #     else:
            #         test_dict = self.dataset.get_test_dict()
            #         eval_res, labels, _ = self.review_agent.eval_all_in_one(
            #             best_rule_path, test_dict
            #         )

            #         # create test_df from test_dict by concatenating all the dataframes
            #         test_df = []
            #         for k, df in test_dict.items():
            #             test_df.append(df)
            #         test_df = pd.concat(test_df)
            #     # remove index column
            #     test_df = test_df.drop(columns=["index"])
            #     # re-add index column
            #     test_df["index"] = np.arange(len(test_df))
            #     self.visualize(
            #         test_df[["value", "label", "index"]].values, labels, figure_path
            #     )
            # else:
            #     raise ValueError(
            #         f"Invalid dataset mode {self.dataset.get_dataset_mode()}"
            #     )
            self.cur_iter += 1

        self.wrapup_run(start_time, last_rule_paths)

    def wrapup_run(self, start_time, last_rule_path=None): 
        stats_path = os.path.join(self.rule_path, "stats.json")
        stats = {
            "cur_iter": self.cur_iter,
            "max_iter": self.max_iter,
            "max_time": self.max_time,
            "time_elapsed": time.time() - start_time,
            "best_rule_paths": last_rule_path,
            "mode": self.mode,
            "token_count": {},
        }
        for agent in [self.detection_agent, self.repair_agent, self.review_agent]:
            stats["token_count"][agent.name] = agent.get_token_count()
        with open(stats_path, "w") as f:
            json.dump(stats, f)
        logging.info(
            f"[TrainingEngine]: Rule training finished after {self.cur_iter} iterations and {(time.time() - start_time)/60} minutes"
        )

    def run_baseline(self):
        logging.info(f"[TrainingEngine]: Running baseline {self.mode}")
        self.cur_iter = 0

        start_time = time.time()

        # num_chunks = len(self.train_df) // self.chunk_size + 1
        test_dict = self.dataset.get_test_dict()
        total_iterations = len(test_dict)

        self.LLMAD.set_start_time()

        all_pred_labels = []
        all_gt_labels = []

        completed_indices = 0

        while time.time() - start_time < self.max_time:
            logging.info(f"[TrainingEngine]: === LLMAD Iteration {self.cur_iter} ===")

            # TODO: support all-in-one mode
            curr_df = self.dataset.get_test_df_by_iter(self.cur_iter)

            # remove label
            # curr_df = curr_df.drop(columns=["label"])

            curr_json_path = os.path.join(self.rule_path, f"llmad_{self.cur_iter}.json")

            # input curr_df but drop labels
            json_obj = self.LLMAD.run(
                curr_df=curr_df[["value", "index"]], curr_json_path=curr_json_path
            )

            if json_obj is None:
                logging.info(f"[TrainingEngine]: LLMAD returns None, timing out")
                break
            pred_labels = np.zeros(len(curr_df))
            start_index = curr_df["index"].iloc[0]
            # set scores to 1 for anomalies
            for idx in json_obj["anomalies"]:
                pred_labels[idx - start_index] = 1
            all_pred_labels.append(pred_labels)
            all_gt_labels.append(curr_df["label"].values)

            completed_indices += len(curr_df)
            self.cur_iter += 1

            if self.cur_iter >= total_iterations:
                break

        all_pred_labels = np.concatenate(all_pred_labels)
        all_gt_labels = np.concatenate(all_gt_labels)
        assert len(all_pred_labels) == len(
            all_gt_labels
        ), "Length of all_pred_labels and all_gt_labels do not match"
        logging.info(
            f"[TrainingEngine]: LLMAD finished after {self.cur_iter} iterations and {(time.time() - start_time)/60} minutes"
        )
        logging.info(
            f"[TrainingEngine]: Completed {self.cur_iter}/{total_iterations} iterations"
        )
        final_pred_path = os.path.join(self.rule_path, "llmad_scores_test.npy")
        np.save(final_pred_path, all_pred_labels)
        final_gt_path = os.path.join(self.rule_path, "llmad_gt_test.npy")
        np.save(final_gt_path, all_gt_labels)
        final_res_path = os.path.join(self.rule_path, "llmad_eval_res_test.json")
        final_res_dict = calculate_performance(all_pred_labels, all_gt_labels)
        with open(final_res_path, "w") as f:
            json.dump(final_res_dict, f)

    def ablation_detection_only(self):
        logging.info(f"[TrainingEngine]: Running ablation {self.mode}")
        self.update_rule_path_prefix()

        self.cur_iter = 0
        last_rule_path = None

        start_time = time.time()

        self.detection_agent.set_start_time()

        while (
            self.cur_iter < self.max_iter and time.time() - start_time < self.max_time
        ):
            logging.info(f"[TrainingEngine]: === Rule Iteration {self.cur_iter} ===")

            # pick a random int
            curr_dfs = []

            for i in range(self.sample_per_prompt):
                rand_num = np.random.randint(0, 1000)
                curr_df = self.dataset.get_train_df_by_iter(rand_num)
                curr_dfs.append(curr_df)

            self.detection_agent.run(
                curr_dfs=curr_dfs,
                curr_rule_path=self.get_rule_path(),
                last_rule_path=last_rule_path,
                image_path=None,
            )

            best_rule_path = self.get_rule_path()

            try:
                if self.dataset.get_dataset_mode() == "one-by-one":
                    if (
                        self.mode == "train-combined-fn"
                        or self.mode == "train-combined-fp"
                    ):
                        eval_res, labels, _ = self.review_agent.combined_eval(
                            best_rule_path, eval_mode="test"
                        )
                    else:
                        eval_res, labels, _ = self.review_agent.eval(
                            best_rule_path, self.dataset.get_test_df()
                        )

                    prefix = best_rule_path.split("/")[-1].split(".")[0]
                    figure_path = os.path.join(
                        self.rule_path, f"eval_figure_{prefix}.png"
                    )
                    self.visualize(
                        self.dataset.get_test_df()[["value", "label", "index"]].values,
                        labels,
                        figure_path,
                    )
                elif self.dataset.get_dataset_mode() == "all-in-one":
                    # TODO: implement combine eval for all-in-one mode

                    prefix = best_rule_path.split("/")[-1].split(".")[0]
                    figure_path = os.path.join(
                        self.rule_path, f"eval_figure_{prefix}.png"
                    )
                    if (
                        self.mode == "train-combined-fn"
                        or self.mode == "train-combined-fp"
                    ):
                        eval_res, labels, _ = (
                            self.review_agent.combined_eval_all_in_one(
                                best_rule_path, eval_mode="test"
                            )
                        )
                        dataset_dict = self.dataset.get_dataset_dict()
                        test_df = []
                        for _, (_, df, *_rest) in dataset_dict.items():
                            test_df.append(df)
                        test_df = pd.concat(test_df)
                    else:
                        test_dict = self.dataset.get_test_dict()
                        eval_res, labels, _ = self.review_agent.eval_all_in_one(
                            best_rule_path, test_dict
                        )

                        # create test_df from test_dict by concatenating all the dataframes
                        test_df = []
                        for k, df in test_dict.items():
                            test_df.append(df)
                        test_df = pd.concat(test_df)
                    # remove index column
                    test_df = test_df.drop(columns=["index"])
                    # re-add index column
                    test_df["index"] = np.arange(len(test_df))
                    self.visualize(
                        test_df[["value", "label", "index"]].values, labels, figure_path
                    )
                else:
                    raise ValueError(
                        f"Invalid dataset mode {self.dataset.get_dataset_mode()}"
                    )
                eval_res["error"] = None
            except Exception as e:
                eval_res = {"f1": -1, "precision": -1, "recall": -1, "error": str(e)}
            final_res_path = best_rule_path.replace(".py", "_eval_res_test.json")
            with open(final_res_path, "w") as f:
                json.dump(eval_res, f)

            last_rule_path = best_rule_path
            self.cur_iter += 1

        # output stats to stats.json
        stats_path = os.path.join(self.rule_path, "stats.json")
        stats = {
            "cur_iter": self.cur_iter,
            "max_iter": self.max_iter,
            "max_time": self.max_time,
            "time_elapsed": time.time() - start_time,
            "best_rule_path": last_rule_path,
            "mode": self.mode,
            "token_count": {},
        }
        for agent in [self.detection_agent]:
            stats["token_count"][agent.name] = agent.get_token_count()
        with open(stats_path, "w") as f:
            json.dump(stats, f)
        logging.info(
            f"[TrainingEngine]: Rule training finished after {self.cur_iter} iterations and {(time.time() - start_time)/60} minutes"
        )

    def update_rule_path_prefix(self) -> None:
        postfix = self.dataset_path.split("/")[-1].split(".")[0]
        self.rule_path_prefix = os.path.join(
            self.rule_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + postfix
        )

    def visualize(self, input_data, labels, save_path, image_subplots=(1, 1)):
        """
        test_data: np.ndarray (shape=(n, 3), (value, label, index))
        labels: np.ndarray (shape=(n,), 0 for normal or 1 for abnormal)
        """
        # shift the index to start from 0
        # test_data[:, 2] = test_data[:, 2] - test_data[0, 2]
        assert len(input_data) == len(
            labels
        ), "Length of input_data and labels do not match"
        # print(image_subplots)
        num_row = int(image_subplots[0])
        num_col = int(image_subplots[1])

        # scale with number of points in the df
        fig, axes = plt.subplots(
            num_row,
            num_col,
            figsize=(min(30 * num_col * len(input_data) // 2500, 150), 15 * num_row),
        )
        # fig, axes = plt.subplots(num_row, num_col, figsize=(15, 5))

        fig.patch.set_facecolor("white")
        # print(f"figure size: {min(30*num_col*len(input_data)//2500, 600)}, {20*num_row}")

        partition_size = len(input_data) // (num_row * num_col)

        for row in range(num_row):
            for col in range(num_col):
                if num_row == 1 and num_col == 1:
                    ax = axes
                elif num_row == 1:
                    ax = axes[col]
                elif num_col == 1:
                    ax = axes[row]
                else:
                    ax = axes[row, col]
                partition_start = (row * num_col + col) * partition_size
                partition_end = min(
                    (row * num_col + col + 1) * partition_size, len(input_data)
                )
                data_partition = input_data[partition_start:partition_end]
                label_partition = labels[partition_start:partition_end]
                # first plot value vs index
                ax.plot(data_partition[:, 2], data_partition[:, 0])
                # now highlight the anomaly point with red dot
                for i in range(len(data_partition)):
                    if data_partition[i, 1] == 1:
                        ax.plot(
                            data_partition[i, 2],
                            data_partition[i, 0],
                            "ro",
                            markersize=5,
                        )

                # count anomaly segment
                ano_seg = []
                ano_flag = 0
                start, end = 0, 0
                for i in range(len(label_partition)):
                    if label_partition[i] == 1 and ano_flag == 0:
                        start = i
                        ano_flag = 1
                    elif label_partition[i] == 0 and ano_flag == 1:
                        end = i
                        ano_flag = 0
                        ano_seg.append((start, end))

                    if i == len(label_partition) - 1 and label_partition[i] == 1:
                        end = i
                        ano_seg.append((start, end))
                # do a vertical span on every label=1 point in labels
                for seg in ano_seg:
                    ax.axvspan(
                        data_partition[seg[0], 2],
                        data_partition[seg[1], 2],
                        alpha=0.5,
                        color="pink",
                    )

                # set font for x and y axis and ticks
                # ax.set_xlabel("Index", fontsize=20)
                # ax.set_ylabel("Value", fontsize=20)
                # ax.tick_params(axis='x', labelsize=20)
                # ax.tick_params(axis='y', labelsize=20)
                ax.set_xlabel("Index", fontsize=40)
                ax.set_ylabel("Value", fontsize=40)
                ax.tick_params(axis="x", labelsize=40)
                ax.tick_params(axis="y", labelsize=40)
                ax.set_facecolor("white")
        # tight layout
        plt.tight_layout()
        # enable grid
        plt.grid()
        plt.savefig(save_path, facecolor="white")
        # plt.show()

    def get_train_df(self):
        return self.dataset.get_train_df()

    def get_whole_train_df(self):
        return self.dataset.get_whole_train_df()

    def get_test_df(self):
        return self.dataset.get_test_df()

    def get_rule_path(self, top_k_curr=1):
        rule_path = self.rule_file_base + f"_iter{self.cur_iter}_{top_k_curr}.py"
        # rule_path = self.rule_file_base + ".py"
        return rule_path

    def eval(self, rule_file, mode="test"):
        """
        For debug only
        """
        if mode == "test":
            if self.dataset_mode == "one-by-one":
                return self.review_agent.eval(rule_file, self.dataset.get_test_df())
            elif self.dataset_mode == "all-in-one":
                return self.review_agent.eval_all_in_one(
                    rule_file, self.dataset.get_test_dict()
                )
        elif mode == "train":
            if self.dataset_mode == "one-by-one":
                return self.review_agent.eval(
                    rule_file, self.dataset.get_whole_train_df()
                )
            elif self.dataset_mode == "all-in-one":
                return self.review_agent.eval_all_in_one(
                    rule_file, self.dataset.get_train_dict()
                )
        else:
            raise ValueError(f"Invalid mode {mode}")

    def combined_eval(self, rule_file, eval_mode="train"):
        """
        For debug only
        """
        return self.review_agent.combined_eval(rule_file, eval_mode)

    def combined_inference(
        self,
        rule_file_fn=None,
        threshold_fn=0.0,
        rule_file_fp=None,
        threshold_fp=0.0,
        eval_mode="train",
    ):
        return self.review_agent.combined_inference(
            rule_file_fn, threshold_fn, rule_file_fp, threshold_fp, eval_mode
        )

    def combined_inference_all_in_one(
        self,
        rule_file_fn=None,
        threshold_fn=0.0,
        rule_file_fp=None,
        threshold_fp=0.0,
        eval_mode="train",
    ):
        return self.review_agent.combined_inference_all_in_one(
            rule_file_fn, threshold_fn, rule_file_fp, threshold_fp, eval_mode
        )

    def get_detection_agent(self):
        return self.detection_agent

    def get_repair_agent(self):
        return self.repair_agent

    def get_review_agent(self):
        return self.review_agent