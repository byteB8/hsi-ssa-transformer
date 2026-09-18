"""Classification metrics reported for hyperspectral benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


@dataclass
class Scores:
    overall_accuracy: float
    average_accuracy: float
    kappa: float
    per_class: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "overall_accuracy": self.overall_accuracy,
            "average_accuracy": self.average_accuracy,
            "kappa": self.kappa,
            "per_class": self.per_class,
        }

    def __str__(self) -> str:
        return (
            f"OA {self.overall_accuracy * 100:.2f}  "
            f"AA {self.average_accuracy * 100:.2f}  "
            f"Kappa {self.kappa * 100:.2f}"
        )


def score(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Scores:
    labels = list(range(n_classes))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    support = matrix.sum(axis=1)
    # A class absent from the test split has no recall; leave it out of the mean
    # rather than scoring it zero, which would silently depress AA.
    present = support > 0
    recall = np.zeros(n_classes, dtype=float)
    recall[present] = np.diag(matrix)[present] / support[present]
    return Scores(
        overall_accuracy=float((y_true == y_pred).mean()),
        average_accuracy=float(recall[present].mean()),
        kappa=float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        per_class=[float(v) for v in recall],
    )
