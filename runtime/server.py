# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import numpy as np
import pandas as pd
from agentsdk.agent import AgentApiServer
from agentsdk.common import logger
from agentsdk.infracontroller import ControllerClient

from agent import DetectionAgent


class InParameters:
    def __init__(self, action, rule_path, data_file, node_list):
        self.action = action
        self.rule_path = rule_path
        self.data_file = data_file
        self.node_list = node_list


class OutParameters:
    def __init__(self, messages):
        self.messages = messages


class Server(AgentApiServer):
    def __init__(self):
        super().__init__()
        self._controller_client = ControllerClient()

        self.interval = 10
        self.rule_filename = "rule.py"
        self.counter_url_format_str = "http://{node}:9999/counter.pkl"
        self.ib_anomaly_format_str = "IB {nic} in {node}"

    def build_in_parameters(self, data):
        return InParameters(
            data.data.action,
            os.path.expandvars(data.data.rule_path),
            os.path.expandvars(getattr(data.data, "data_file", "")),
            getattr(data.data, "node_list", []),
        )

    def gen_sliding_window(self, metrics):
        if len(metrics) < self.interval:
            return [[(0, 0) for _ in range(self.interval - len(metrics))] + metrics]
        return [
            metrics[i : i + self.interval]
            for i in range(len(metrics) - self.interval + 1)
        ]

    def load_data(self, data_file):
        try:
            data = np.load(data_file)
            assert len(data["feature"]) == len(
                data["label"]
            ), "Length of feature and label do not match"
            return data
        except Exception as e:
            logger.exception(f"Failed to load data file from {data_file}: {e}")
            raise e

    def load_rule(self, rule_path):
        try:
            if os.path.exists(os.path.join(rule_path, "best.py")):
                self.rule_filename = "best.py"
            with open(os.path.join(rule_path, self.rule_filename), "r") as f:
                rule = f.read()
            exec(rule, globals())
        except Exception as e:
            logger.exception(f"Failed to load rule file from {rule_path}: {e}")
            raise e
        if not is_negative:
            raise "Failed to load rule function"

    def load_counter(self, node):
        counter_url = self.counter_url_format_str.format(node=node)
        try:
            df = pd.read_pickle(counter_url)
            return df
        except Exception as e:
            logger.warning(
                f"Cannot read counter file on {node} from {counter_url}: {e}"
            )
            return None

    def detect_with_rule(self, metrics):
        for sample in self.gen_sliding_window(metrics):
            if is_negative(np.array([x for pair in sample for x in pair])):
                logger.info(f"Detected anomaly on sample {sample}")
                return True
        return False

    def perform_operation(self, params):
        if params.action == "train":
            return self.train(params.data_file, params.rule_path)
        elif params.action == "inference":
            return self.inference(params.rule_path, params.node_list)
        else:
            raise ValueError(f"Unknown action {params.action}")

    def train(self, data_file, rule_path):
        logger.info(
            f"Accept request to train rules using data file {data_file} and output to {rule_path}"
        )

        data = self.load_data(data_file)
        agent = DetectionAgent(data, rule_path)
        ret, fn, msg = agent.run()
        if ret:
            results = f"Generated model/rule in {rule_path} and accuracy figure in {fn} successfully, {msg}"
        else:
            results = f"Generated model/rule in {rule_path} and accuracy figure in {fn}, but precision regresses a little bit. Please consider to retry for a better result."
        return OutParameters(
            [
                {
                    "role": "detection agent",
                    "content": results,
                }
            ]
        )

    def inference(self, rule_path, node_list):
        logger.info(
            f"Accept request to detect anomaly using rule {rule_path} on nodes {', '.join(node_list)}"
        )

        self.load_rule(rule_path)
        anomalies = []
        for node in node_list:
            df = self.load_counter(node)
            if not df:
                continue
            for col in df.columns:
                if "mlx" not in col:
                    continue
                logger.info(
                    f"Checking {self.ib_anomaly_format_str.format(nic=col, node=node)} ..."
                )
                metrics = df.tail(self.interval * 2)[col].to_list()
                if len(metrics) < self.interval:
                    metrics = [
                        (0, 0) for _ in range(self.interval - len(metrics))
                    ] + metrics
                if self.detect_with_rule(metrics):
                    anomalies.append(
                        self.ib_anomaly_format_str.format(nic=col, node=node)
                    )

        if len(anomalies) > 1:
            anomalies[-1] = "and " + anomalies[-1]
        results = (
            f"Anomaly detected in {', '.join(anomalies)}"
            if anomalies
            else "No anomaly detected."
        )
        return OutParameters(
            [
                {
                    "role": "detection agent",
                    "content": results,
                }
            ]
        )


if __name__ == "__main__":
    server = Server()
    server.run()
