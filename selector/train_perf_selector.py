import json
from abc import ABC, abstractmethod


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
                perf = json.load(f)

            if perf["f1"] > best_f1:
                best_f1 = perf["f1"]
                best_rule_path = rule_path

        return best_rule_path
