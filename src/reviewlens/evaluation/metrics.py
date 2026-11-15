"""Classification metrics.

``compute_metrics`` returns one flat dict per run, so results from many runs stack directly
into a table (one row per method x fraction x seed) and a CSV file.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, labels: Sequence[str]
) -> dict[str, float]:
    """Macro-F1 (primary metric), accuracy, and per-class precision / recall / F1.

    Macro-F1 averages the F1 of each class with equal weight, so a model cannot score well by
    ignoring a small class (accuracy can).
    """
    class_ids = list(range(len(labels)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_ids, zero_division=0
    )
    metrics = {
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    for i, name in enumerate(labels):
        metrics[f"precision_{name}"] = float(precision[i])
        metrics[f"recall_{name}"] = float(recall[i])
        metrics[f"f1_{name}"] = float(f1[i])
    return metrics


def compute_confusion(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, n_labels: int
) -> npt.NDArray[np.int_]:
    """Confusion matrix: row = true class, column = predicted class."""
    return np.asarray(confusion_matrix(y_true, y_pred, labels=list(range(n_labels))), dtype=int)
