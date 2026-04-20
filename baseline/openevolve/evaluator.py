import os
import sys
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.review_agent import ReviewAgent
from datasets.dataset import ArgosDataset
from openevolve.evaluation_result import EvaluationResult

logger = logging.getLogger(__name__)

EVAL_WITH_TRAIN_SAMPLE_MODE = "train-LLM-only-with-train-sample"


class _EvalOnlyReviewAgent(ReviewAgent):
    """
    Lightweight ReviewAgent that skips LLM initialization.
    Only supports evaluation helpers (eval + eval_scores_by_metrics).
    """

    def __init__(self, chunk_size: int, mode: str = "train-LLM-only") -> None:
        self.LLM = None
        self.name = "ReviewAgentEvalOnly"
        self.dataset = None
        self.dataset_name = None
        self.chunk_size = chunk_size
        self.mode = "train-LLM-only"
        self.model_res_path = None
        self.max_time = 0


class OpenEvolveEvaluator:
    """
    Evaluator wrapper for OpenEvolve.
    Initialized with dataset and evaluation split info.
    """

    def __init__(
        self,
        dataset: ArgosDataset,
        eval_split: str,
        chunk_size: int = 1000,
        engine_mode: str = "train-LLM-only",
    ) -> None:
        self.dataset = dataset
        self.eval_split = eval_split
        self.chunk_size = chunk_size
        self.engine_mode = engine_mode
        self.agent = _EvalOnlyReviewAgent(chunk_size=chunk_size, mode=engine_mode)

    def _get_random_train_sample(self) -> Dict:
        """
        Return one random train chunk so OpenEvolve can align with
        Argos engine's random chunked training behavior.
        """
        train_dict = self.dataset.get_train_dict()
        if len(train_dict) == 0:
            return {
                "chunk_size": self.chunk_size,
                "sample_size": 0,
                "records": [],
            }
        rand_num = int(np.random.randint(0, len(train_dict)))
        train_df = self.dataset.get_train_df_by_iter(rand_num)
        return {
            "chunk_size": self.chunk_size,
            "sample_size": int(len(train_df)),
            "records": train_df.to_dict(orient="records"),
        }

    def evaluate(self, program_path: str) -> EvaluationResult:
        """
        Evaluate the program and return metrics.

        If OpenEvolve's EvaluationResult is available, return an EvaluationResult
        with large data samples stored in artifacts to avoid bloating metrics.
        Otherwise, fall back to the original dict-based interface.
        """
        try:
            if self.dataset.get_dataset_mode() == "one-by-one":
                eval_df = _get_eval_df(self.dataset, self.eval_split)
                metrics_dict, _, _ = self.agent.eval(
                    rule_file=program_path,
                    eval_df=eval_df,
                    log=False,
                )
            elif self.dataset.get_dataset_mode() == "all-in-one":
                eval_dict = _get_eval_df_dict(self.dataset, self.eval_split)
                logger.info(f"Eval dict: {eval_dict}")
                metrics_dict, _, _ = self.agent.eval_all_in_one(
                    rule_file=program_path,
                    df_dict=eval_dict,
                )
            else:
                raise ValueError(
                    f"Unsupported dataset mode: {self.dataset.get_dataset_mode()}"
                )

            combined_score = float(metrics_dict.get("f1", 0.0))
            base_metrics: Dict[str, float] = {
                "combined_score": combined_score,
                "f1": float(metrics_dict.get("f1", 0.0)),
            }

            # When enabled, attach a random train sample as an artifact instead of
            # putting it directly into metrics (which would massively bloat prompts).
            artifacts: Dict[str, object] = {}
            if self.engine_mode == EVAL_WITH_TRAIN_SAMPLE_MODE:
                artifacts["train_sample"] = self._get_random_train_sample()

            return EvaluationResult(metrics=base_metrics, artifacts=artifacts)
        except Exception as e:
            logger.exception(f"Error during evaluation of program {program_path}: {e}")
            return EvaluationResult(metrics={"combined_score": 0.0, "error": 0.0})


_DEFAULT_EVALUATOR: Optional[OpenEvolveEvaluator] = None


def _get_eval_df(dataset: ArgosDataset, eval_split: str) -> pd.DataFrame:
    if eval_split == "train":
        return dataset.get_train_df()
    if eval_split == "test":
        return dataset.get_test_df()
    if eval_split == "whole_train":
        return dataset.get_whole_train_df()
    raise ValueError(f"Unsupported eval split: {eval_split}")


def _get_eval_df_dict(dataset: ArgosDataset, eval_split: str) -> Dict[int, pd.DataFrame]:
    if eval_split == "train":
        return dataset.get_train_dict()
    if eval_split == "test":
        return dataset.get_test_dict()
    raise ValueError(f"Unsupported eval split for all-in-one: {eval_split}")


def _load_dataset_from_env() -> ArgosDataset:
    dataset_path = os.environ.get("ARGOS_DATASET_PATH")
    if not dataset_path:
        raise RuntimeError("Missing env var: ARGOS_DATASET_PATH")

    dataset_mode = os.environ.get("ARGOS_DATASET_MODE", "one-by-one")
    engine_mode = os.environ.get("ARGOS_ENGINE_MODE", "train-LLM-only")
    chunk_size = int(os.environ.get("ARGOS_CHUNK_SIZE", "1000"))
    train_test_split = float(os.environ.get("ARGOS_TRAIN_TEST_SPLIT", "0.7"))

    logger.info(f"Loading dataset from {dataset_path}")
    logger.info(f"Dataset mode: {dataset_mode}")
    logger.info(f"Engine mode: {engine_mode}")
    logger.info(f"Chunk size: {chunk_size}")
    logger.info(f"Train test split: {train_test_split}")
    return ArgosDataset(
        dataset_path=dataset_path,
        dataset_mode=dataset_mode,
        engine_mode=engine_mode,
        chunk_size=chunk_size,
        train_test_split=train_test_split,
    )


def configure_evaluator_from_env() -> None:
    """
    Configure a module-level evaluator instance from environment variables.
    """
    global _DEFAULT_EVALUATOR
    dataset = _load_dataset_from_env()
    # train or test
    eval_split = os.environ.get("ARGOS_EVAL_SPLIT", "train")
    chunk_size = int(os.environ.get("ARGOS_CHUNK_SIZE", "1000"))
    engine_mode = os.environ.get("ARGOS_ENGINE_MODE", "train-LLM-only")
    _DEFAULT_EVALUATOR = OpenEvolveEvaluator(
        dataset=dataset,
        eval_split=eval_split,
        chunk_size=chunk_size,
        engine_mode=engine_mode,
    )


def evaluate(program_path: str) -> EvaluationResult:
    """
    OpenEvolve entrypoint. Uses env config on first call.
    """
    if _DEFAULT_EVALUATOR is None:
        configure_evaluator_from_env()
    return _DEFAULT_EVALUATOR.evaluate(program_path)
