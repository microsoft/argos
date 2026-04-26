from typing import Type

import numpy as np
import sklearn.metrics

from eval_metrics.base import EvalInterface, MetricInterface
from eval_metrics.metrics import F1Class


class PointF1(EvalInterface):
    """
    Using Traditional F1 score to evaluate the models.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "point-wise f1"

    def calc(self, scores, labels, margins) -> type[MetricInterface]:
        """
        Returns:
         A F1class (Evaluations.Metrics.F1class), including:\n
            best_f1: the value of best f1 score;\n
            precision: corresponding precision value;\n
            recall: corresponding recall value;
        """
        prec, recall, _ = sklearn.metrics.precision_recall_curve(
            y_true=labels, y_score=scores
        )

        denom = prec + recall
        f1_all = np.divide(
            2 * prec * recall,
            denom,
            out=np.zeros_like(denom, dtype=float),
            where=denom != 0,
        )
        max_idx = np.nanargmax(f1_all)

        return F1Class(
            name=self.name, p=prec[max_idx], r=recall[max_idx], f1=f1_all[max_idx]
        )
