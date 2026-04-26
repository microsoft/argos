import json
from abc import ABC, abstractmethod
from typing import Dict, List


class TrainPerfSelector(ABC):
    def __init__(self, rule_perf_pairs) -> None:
        """
        Args:
            rule_perf_pairs (List[Tuple[str, str]]): A list of rule path - performance path pairs.
        """
        super().__init__()
        self.rule_perf_pairs = rule_perf_pairs

    def select(self) -> str:
        """
        Select the best rule from the given rule path - performance path pairs.

        Returns:
            str: The best rule path.
        """

        best_rule_path = None
        best_f1 = 0

        for rule_path, perf_path in self.rule_perf_pairs:
            with open(perf_path, "r") as f:
                perf = json.load(f)["event-based f1 under pa with mode squeeze"]

            if perf["f1"] > best_f1:
                best_f1 = perf["f1"]
                best_rule_path = rule_path

        return best_rule_path
    
    def extract_pipeline_id(self, rule_path: str) -> int:
        """
        Extract the pipeline ID from the rule path.

        Args:
            rule_path (str): The path of the rule.

        Returns:
            int: The pipeline ID.

        Example: rule_1_1.py -> pipeline_id = 1
        Example: rule_1_2.py -> pipeline_id = 1
        Example: rule_2_1.py -> pipeline_id = 2
        Example: rule_2_2.py -> pipeline_id = 2
        """
        return int(rule_path.split("/")[-1].split("_")[1])
    

    def extract_score_path(self, rule_path: str) -> str:
        """
        Extract the score path from the rule path.
        Uses validation results if available, otherwise falls back to train results.

        Args:
            rule_path (str): The path of the rule.

        Returns:
            str: The score path.
        """
        import os
        val_path = rule_path.replace(".py", "_eval_res_val.json")
        if os.path.exists(val_path):
            return val_path
        return rule_path.replace(".py", "_eval_res_train.json")
    

    def select_k_top_rule(self, k: int, prev_best_rule=None) -> List[str]:
        """
        Select the top k rules from the given rule path - performance path pairs.

        Args:
            k (int): The number of top rules to select.

        Returns:
            List[str]: A list of top k rule paths.
        """
        
        if prev_best_rule is not None:
            prev_scores = []

            for pipeline, rule_path in prev_best_rule.items():
                if rule_path is None:
                    prev_scores.append((0, pipeline))
                    continue
                score_path = self.extract_score_path(rule_path)
                with open(score_path, "r") as f:
                    perf = json.load(f)["event-based f1 under pa with mode squeeze"]
                prev_scores.append((perf["f1"], pipeline))
            
            # Sort by F1 score
            prev_scores.sort(key=lambda pair: pair[0], reverse=True)
            prev_rank: Dict[int, int] = {pipeline: rank for rank, (_, pipeline) in enumerate(prev_scores)}

        perf_list = []
        for rule_path, perf_path in self.rule_perf_pairs:
            with open(perf_path, "r") as f:
                perf = json.load(f)["event-based f1 under pa with mode squeeze"]
            perf_list.append((rule_path, perf["f1"]))

        if prev_best_rule is not None:
            # Sort by F1 score descending; on ties, prefer the pipeline that
            # was the previous best (smallest prev_rank). Since reverse=True
            # sorts the whole tuple descending, negate prev_rank so that
            # smaller ranks come first within equal F1.
            perf_list.sort(
                key=lambda x: (
                x[1],  # Sort by F1 score
                -prev_rank.get(self.extract_pipeline_id(x[0]), float("inf")),
                ),
                reverse=True
            )
        else:
            # Sort by F1 score
            perf_list.sort(key=lambda x: x[1], reverse=True)

        # Select top k rules
        return [rule_path for rule_path, _ in perf_list[:k]]
