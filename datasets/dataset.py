import json
import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ArgosDataset(ABC):
    def __init__(
        self,
        dataset_path,
        dataset_mode="one-by-one",
        engine_mode="train-LLM-only",
        chunk_size=1000,
        image_chunk_size=None,
        train_test_split=0.7,
        model_res_path=None,
    ) -> None:
        """
        Args:
            dataset_path (str): The path to the dataset.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_mode = dataset_mode
        self.engine_mode = engine_mode
        self.chunk_size = chunk_size
        self.image_chunk_size = image_chunk_size
        self.train_test_split = train_test_split
        self.model_res_path = model_res_path
        self.preprocess()

    def preprocess(self):
        """
        Preprocess the dataset.
        """
        if self.dataset_mode == "one-by-one":
            # assume the dataset is a csv file
            assert self.dataset_path.endswith(".csv"), "Dataset should be a csv file"

            curve_name = self.dataset_path.split("/")[-1].split(".")[0]

            df = pd.read_csv(self.dataset_path)
            # for every value, chunk to 3 digits after the decimal point
            # df["value"] = df["value"].apply(lambda x: round(x, 3))

            self.train_df = df[: int(len(df) * self.train_test_split)]

            # also consider validation portion
            # split_index = int(len(self.train_df) * 0.2)
            # self.train_df = self.train_df[:-split_index]

            self.whole_train_df = self.train_df.copy()

            self.test_df = df[int(len(df) * self.train_test_split) :]

            if (
                self.engine_mode == "train-combined-fp"
                or self.engine_mode == "train-combined-fn"
            ):
                assert (
                    self.model_res_path is not None
                ), "Model result path must be provided for combined mode"
                (
                    self.abnormal_dict,
                    self.normal_dict,
                    self.abnormal_df,
                    self.normal_df,
                ) = self.derive_abnormal_normal(self.train_df)
                self.train_df, self.skip_training = self.derive_train_df_from_indices(
                    self.train_df, curve_name
                )
                self.model_test_label_dict = {}
                self.model_test_label_dict[curve_name] = self.load_model_test_labels(
                    curve_name
                )
                self.model_train_label_dict = {}
                self.model_train_label_dict[curve_name] = self.load_model_train_labels(
                    curve_name
                )

            self.train_dict = self.split_df_by_chunk(self.train_df, self.chunk_size)
            self.test_dict = self.split_df_by_chunk(self.test_df, self.chunk_size)
            if self.image_chunk_size:
                self.train_dict_image = self.split_df_by_chunk(
                    self.train_df, self.image_chunk_size
                )

        elif self.dataset_mode == "all-in-one":
            # assert dataset_path is a directory
            assert os.path.isdir(self.dataset_path), "Dataset should be a directory"
            dataset_files = os.listdir(self.dataset_path)

            # sort the dataset_files
            dataset_files = sorted(dataset_files)

            # Step 1: split the dataset into train and test
            # dataset_name -> train_df, test_df
            dataset_dict = {}
            for dataset_file in dataset_files:
                if not dataset_file.endswith(".csv"):
                    continue
                df = pd.read_csv(os.path.join(self.dataset_path, dataset_file))
                dataset_name = dataset_file.split(".")[0]
                train_df = df[: int(len(df) * self.train_test_split)]
                test_df = df[int(len(df) * self.train_test_split) :]
                dataset_dict[dataset_name] = (train_df, test_df)
            self.dataset_dict = dataset_dict
            # Optional: derive abnormal and normal dict
            if (
                self.engine_mode == "train-combined-fp"
                or self.engine_mode == "train-combined-fn"
            ):
                # self.abnormal_dict = {}
                # self.normal_dict = {}
                self.abnormal_list = []
                self.normal_list = []
                self.abnormal_avg_std = []
                self.normal_avg_std = []

                self.model_train_label_dict = {}
                self.model_test_label_dict = {}
                for dataset_name, (train_df, test_df) in dataset_dict.items():
                    self.model_test_label_dict[dataset_name] = (
                        self.load_model_test_labels(dataset_name)
                    )
                    self.model_train_label_dict[dataset_name] = (
                        self.load_model_train_labels(dataset_name)
                    )
                    abnormal_dict, normal_dict, abnormal_df, normal_df = (
                        self.derive_abnormal_normal(train_df)
                    )
                    if abnormal_dict is not None:
                        self.abnormal_list.extend(list(abnormal_dict.values()))
                    if normal_dict is not None:
                        self.normal_list.extend(list(normal_dict.values()))
                # calculate the average and std of each chunk, for both abnormal and normal
                for chunk_id in range(len(self.abnormal_list)):
                    chunk = self.abnormal_list[chunk_id]
                    self.abnormal_avg_std.append(
                        (chunk["value"].mean(), chunk["value"].std())
                    )
                for chunk_id in range(len(self.normal_list)):
                    chunk = self.normal_list[chunk_id]
                    self.normal_avg_std.append(
                        (chunk["value"].mean(), chunk["value"].std())
                    )

                # derive new train_df from each dataset
                new_dataset_dict = {}
                self.skip_training = True
                for dataset_name, (train_df, test_df) in dataset_dict.items():
                    train_df, skip_training = self.derive_train_df_from_indices(
                        train_df, dataset_name
                    )
                    if not skip_training:
                        new_dataset_dict[dataset_name] = (train_df, test_df)
                    self.skip_training = self.skip_training and skip_training
                dataset_dict = new_dataset_dict

            # Step 2: split the train_df into chunks
            train_list = []
            test_list = []
            if self.image_chunk_size:
                train_list_image = []
            for dataset_name, (train_df, test_df) in dataset_dict.items():
                train_dict = self.split_df_by_chunk(train_df, self.chunk_size)
                test_dict = self.split_df_by_chunk(test_df, self.chunk_size)

                for chunk_id in range(len(train_dict)):
                    chunk = train_dict[chunk_id]
                    if self.engine_mode == "train-combined-fn":
                        assert (
                            chunk["label"].sum() > 0
                        ), "There should be anomaly in the train_df"
                    elif self.engine_mode == "train-combined-fp":
                        assert (
                            chunk["label"].sum() == 0
                        ), "There should be no anomaly in the train_df"

                # append the values to the list
                train_list.extend(list(train_dict.values()))
                test_list.extend(list(test_dict.values()))
                if self.image_chunk_size:
                    train_dict_image = self.split_df_by_chunk(
                        train_df, self.image_chunk_size
                    )
                    train_list_image.extend(list(train_dict_image.values()))
            # for each list, assign index to each chunk and create dict
            self.train_dict = {i: train_list[i] for i in range(len(train_list))}
            self.test_dict = {i: test_list[i] for i in range(len(test_list))}
            if self.image_chunk_size:
                self.train_dict_image = {
                    i: train_list_image[i] for i in range(len(train_list_image))
                }

            # TODO: support combined mode for all-in-one dataset
        else:
            raise NotImplementedError("Dataset mode not implemented")

    def derive_abnormal_normal(self, df):
        """
        Derive abnormal and normal segments from the dataset.

        Args:
            df (pd.DataFrame): The dataset.

        Returns:
            abnormal_dict (dict): The abnormal dict, iter->df by chunk.
            normal_dict (dict): The normal dict, iter->df by chunk.
            abnormal_df (pd.DataFrame): The abnormal df.
            normal_df (pd.DataFrame): The normal df.
        """

        # derive anomaly segments
        flag = 0
        start = 0
        anomaly_segments = []
        for i in range(len(df)):
            if df.iloc[i]["label"] == 1 and flag == 0:
                start = i
                flag = 1
            elif df.iloc[i]["label"] == 0 and flag == 1:
                anomaly_segments.append((start, i))
                flag = 0

        # create self.abnormal_df
        abnormal_chunks = []
        for seg in anomaly_segments:
            start = seg[0]
            end = seg[1]
            start_chunk = start // self.chunk_size
            end_chunk = end // self.chunk_size
            for i in range(start_chunk, end_chunk + 1):
                abnormal_chunks.append(
                    df[i * self.chunk_size : (i + 1) * self.chunk_size]
                )
        if len(abnormal_chunks) > 0:
            abnormal_df = pd.concat(abnormal_chunks)
        else:
            abnormal_df = None

        normal_segments = []
        # any segment out of the anomaly_segments is normal
        # add the first segment
        if len(anomaly_segments) == 0:
            normal_segments.append((0, len(df)))
        else:
            normal_segments.append((0, anomaly_segments[0][0]))
            for i in range(1, len(anomaly_segments)):
                normal_segments.append(
                    (anomaly_segments[i - 1][1], anomaly_segments[i][0])
                )
            # add the last segment
            normal_segments.append((anomaly_segments[-1][1], len(df)))
        # for each normal segment, split it into chunks
        normal_chunks = []
        for seg in normal_segments:
            start = seg[0]
            end = seg[1]
            if end - start < self.chunk_size:
                continue
            while end - start > self.chunk_size:
                normal_chunks.append(df[start : start + self.chunk_size])
                start += self.chunk_size
        if len(normal_chunks) > 0:
            normal_df = pd.concat(normal_chunks)
        else:
            normal_df = None

        if abnormal_df is not None:
            abnormal_dict = self.split_df_by_chunk(abnormal_df, self.chunk_size)
        else:
            abnormal_dict = None

        if normal_df is not None:
            normal_dict = self.split_df_by_chunk(normal_df, self.chunk_size)
        else:
            normal_dict = None
        return abnormal_dict, normal_dict, abnormal_df, normal_df

    def derive_train_df_from_indices(self, train_df, curve_name):
        """
        Derive the train_df from the incorrect indices.

        Args:
            train_df (pd.DataFrame): The original train_df.

        Returns:
            train_df (pd.DataFrame): The new train_df with only the incorrect indices chunks kept.
        """
        # Yile: this is only for combined mode, and currently only for KPI dataset
        # curve_name = None

        train_df = train_df.copy()

        skip_training = False
        if self.engine_mode == "train-combined-fn":
            # curve_name = self.dataset_path.split("/")[-1].split(".")[0]
            with open(
                os.path.join(self.model_res_path, "IncorrectIndices/train.json"), "r"
            ) as f:
                incorrect_segments = json.load(f)
            fn_segments = incorrect_segments["FN"]

            # if curve_name not in fn_segments:
            #     exit(0)
            # if len(fn_segments[curve_name]) == 0:
            #     exit(0)

            if curve_name in fn_segments and len(fn_segments[curve_name]) > 0:
                chunk_ids = []
                for item in fn_segments[curve_name]:
                    start = item[0]
                    end = item[1]
                    # sanity check: this should be a anomaly segment
                    # assert train_df.iloc[start:end]["label"].sum() > 0, f"FN segment {start} to {end} is not anomaly"
                    if train_df.iloc[start:end]["label"].sum() == 0:
                        logging.info(
                            f"Warning: {curve_name}: FN segment {start} to {end} is not anomaly"
                        )
                        continue
                    start_chunk = start // self.chunk_size
                    end_chunk = end // self.chunk_size
                    chunk_ids.extend(list(range(start_chunk, end_chunk + 1)))
                chunk_ids = list(set(chunk_ids))
                # now only keep the chunks in chunk_ids in the train_df
                train_df = train_df[
                    train_df["index"].apply(lambda x: x // self.chunk_size in chunk_ids)
                ]
            else:
                skip_training = True

        elif self.engine_mode == "train-combined-fp":
            # curve_name = self.dataset_path.split("/")[-1].split(".")[0]
            with open(
                os.path.join(self.model_res_path, "IncorrectIndices/train.json"), "r"
            ) as f:
                incorrect_segments = json.load(f)
            fp_segments = incorrect_segments["FP"]

            # if curve_name not in fp_segments:
            #     exit(0)
            # if len(fp_segments[curve_name]) == 0:
            #     exit(0)
            if curve_name in fp_segments and len(fp_segments[curve_name]) > 0:
                chunk_ids = []
                for item in fp_segments[curve_name]:
                    start = item[0]
                    end = item[1]
                    # sanity check: this should be a anomaly segment
                    # assert self.train_df.iloc[start:end]["label"].sum() == 0, f"FP segment {start} to {end} is anomaly"
                    if train_df.iloc[start:end]["label"].sum() > 0:
                        logging.info(
                            f"Warning: {curve_name}: FP segment {start} to {end} is anomaly"
                        )
                        continue

                    start_chunk = start // self.chunk_size
                    end_chunk = end // self.chunk_size

                    # if this start_chunk to end_chunk has anomaly, skip
                    data = train_df[
                        train_df["index"].apply(
                            lambda x: x // self.chunk_size
                            in list(range(start_chunk, end_chunk + 1))
                        )
                    ]
                    if data["label"].sum() > 0:
                        continue

                    chunk_ids.extend(list(range(start_chunk, end_chunk + 1)))
                chunk_ids = list(set(chunk_ids))
                # now only keep the chunks in chunk_ids in the train_df
                train_df = train_df[
                    train_df["index"].apply(lambda x: x // self.chunk_size in chunk_ids)
                ]
                assert (
                    train_df["label"].sum() == 0
                ), "There should be no anomaly in the train_df"
            else:
                skip_training = True

        return train_df, skip_training

    def load_model_test_labels(self, curve_name):
        test_label_path = os.path.join(
            self.model_res_path, f"TestLabels/{curve_name}.npy"
        )
        return np.load(test_label_path)

    def load_model_train_labels(self, curve_name):
        train_label_path = os.path.join(
            self.model_res_path, f"TrainLabels/{curve_name}.npy"
        )
        return np.load(train_label_path)

    def split_df_by_chunk(self, df, chunk_size):
        # generate a dict: chunk_id -> df
        chunk_dict = {}
        num_chunks = len(df) // chunk_size
        if len(df) % chunk_size != 0:
            num_chunks += 1
        for i in range(num_chunks):
            end_index = min((i + 1) * chunk_size, len(df))
            chunk_dict[i] = df[i * chunk_size : end_index]
        # sort the dict by key
        chunk_dict = dict(sorted(chunk_dict.items(), key=lambda x: x[0]))
        return chunk_dict

    def get_train_df_by_iter(self, iter_num):
        chunk_id = iter_num % len(self.train_dict)
        return self.train_dict[chunk_id]

    def get_test_df_by_iter(self, iter_num):
        chunk_id = iter_num % len(self.test_dict)
        return self.test_dict[chunk_id]

    def get_train_image_df_by_iter(self, iter_num):
        assert self.image_chunk_size is not None, "Image chunk size is not set"
        chunk_id = iter_num % len(self.train_dict_image)
        return self.train_dict_image[chunk_id]

    def get_normal_df_by_iter(self, iter_num):
        if self.normal_dict is None:
            return None
        chunk_id = iter_num % len(self.normal_dict)
        return self.normal_dict[chunk_id]

    def get_abnormal_df_by_iter(self, iter_num):
        if self.abnormal_dict is None:
            return None
        chunk_id = iter_num % len(self.abnormal_dict)
        return self.abnormal_dict[chunk_id]

    def get_closest_abnormal_df(self, df):
        """
        Get the closest abnormal df to the given df.
        Normalize mean and std differences before summing.
        """
        if len(self.abnormal_list) == 0:
            return None

        mean = df["value"].mean()
        std = df["value"].std()

        # Define expected ranges for normalization
        mean_range = max(
            abs(a[0]) for a in self.abnormal_avg_std
        )  # max mean value in abnormal_avg_std
        std_range = max(
            abs(a[1]) for a in self.abnormal_avg_std
        )  # max std value in abnormal_avg_std

        min_diff = float("inf")
        closest_chunk_id = None
        for chunk_id, (abnormal_mean, abnormal_std) in enumerate(self.abnormal_avg_std):
            diff = (
                abs(mean - abnormal_mean) / mean_range
                + abs(std - abnormal_std) / std_range
            )
            if diff < min_diff:
                min_diff = diff
                closest_chunk_id = chunk_id

        return self.abnormal_list[closest_chunk_id]

    def get_closest_normal_df(self, df):
        """
        Get the closest normal df to the given df.
        Normalize mean and std differences before summing.
        """
        if len(self.normal_list) == 0:
            return None

        mean = df["value"].mean()
        std = df["value"].std()

        # Define expected ranges for normalization
        mean_range = max(abs(a[0]) for a in self.normal_avg_std)
        std_range = max(abs(a[1]) for a in self.normal_avg_std)

        min_diff = float("inf")
        closest_chunk_id = None
        for chunk_id, (normal_mean, normal_std) in enumerate(self.normal_avg_std):
            diff = (
                abs(mean - normal_mean) / mean_range + abs(std - normal_std) / std_range
            )
            if diff < min_diff:
                min_diff = diff
                closest_chunk_id = chunk_id

        return self.normal_list[closest_chunk_id]

    def get_train_df(self):
        assert (
            self.dataset_mode == "one-by-one"
        ), "This method is only for one-by-one dataset"
        return self.train_df

    def get_test_df(self):
        assert (
            self.dataset_mode == "one-by-one"
        ), "This method is only for one-by-one dataset"
        return self.test_df

    def get_whole_train_df(self):
        assert (
            self.dataset_mode == "one-by-one"
        ), "This method is only for one-by-one dataset"
        return self.whole_train_df

    def get_train_dict(self):
        return self.train_dict

    def get_test_dict(self):
        return self.test_dict

    def get_train_dict_image(self):
        assert self.image_chunk_size is not None, "Image chunk size is not set"
        return self.train_dict_image

    def get_dataset_mode(self):
        return self.dataset_mode

    def get_skip_training(self):
        return self.skip_training

    def get_model_test_labels(self, curve_name=None):
        assert (
            self.engine_mode == "train-combined-fn"
            or self.engine_mode == "train-combined-fp"
        ), "This method is only for combined mode"
        assert self.model_test_label_dict is not None, "Model test labels are not set"
        if curve_name is not None:
            return self.model_test_label_dict[curve_name]
        else:
            assert len(self.model_test_label_dict) == 1
            return self.model_test_label_dict.popitem()[1]

    def get_model_train_labels(self, curve_name=None):
        assert (
            self.engine_mode == "train-combined-fn"
            or self.engine_mode == "train-combined-fp"
        ), "This method is only for combined mode"
        assert self.model_train_label_dict is not None, "Model train labels are not set"
        if curve_name is not None:
            return self.model_train_label_dict[curve_name]
        else:
            assert len(self.model_train_label_dict) == 1
            return self.model_train_label_dict.popitem()[1]

    def get_dataset_dict(self):
        assert (
            self.dataset_mode == "all-in-one"
        ), "This method is only for all-in-one dataset"
        return self.dataset_dict
