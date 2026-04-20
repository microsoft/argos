import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baseline.openevolve.evaluator import evaluate


def _write_dummy_rule(rule_path: str) -> None:
    with open(rule_path, "w") as f:
        f.write(
            "\n".join(
                [
                    "import numpy as np",
                    "",
                    "def inference(sample: np.ndarray) -> np.ndarray:",
                    "    # Always predict normal",
                    "    return np.zeros(shape=(len(sample),), dtype=int)",
                    "",
                ]
            )
        )


def _write_dummy_csv(csv_path: str, rows: int = 50) -> None:
    df = pd.DataFrame(
        {
            "value": np.random.randn(rows),
            "label": np.zeros(rows, dtype=int),
            "index": np.arange(rows),
        }
    )
    df.to_csv(csv_path, index=False)


class OpenEvolveEvaluatorTests(unittest.TestCase):
    def test_evaluate_one_by_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "data.csv")
            rule_path = os.path.join(tmpdir, "rule.py")
            _write_dummy_csv(csv_path)
            _write_dummy_rule(rule_path)

            os.environ["ARGOS_DATASET_PATH"] = csv_path
            os.environ["ARGOS_DATASET_MODE"] = "one-by-one"
            os.environ["ARGOS_ENGINE_MODE"] = "train-LLM-only"
            os.environ["ARGOS_CHUNK_SIZE"] = "10"
            os.environ["ARGOS_TRAIN_TEST_SPLIT"] = "0.7"
            os.environ["ARGOS_EVAL_SPLIT"] = "test"

            result = evaluate(rule_path)

            # Support both dict (legacy) and EvaluationResult (with artifacts)
            if hasattr(result, "to_dict"):
                metrics = result.to_dict()
            else:
                metrics = result

            self.assertIn("combined_score", metrics)
            self.assertIn("f1", metrics)


if __name__ == "__main__":
    unittest.main()
