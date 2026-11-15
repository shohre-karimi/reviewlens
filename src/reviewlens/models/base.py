"""The common interface every model in the project follows.

All models, supervised and semi-supervised, are trained the same way:

    model.fit(X_train, y_masked)

where ``y_masked`` holds a class id for labeled rows and ``UNLABELED`` (-1) for the rest.
Supervised models train on the labeled rows only; semi-supervised models also use the
unlabeled ones. Because the call is identical, the experiment runner can treat every method
the same way and the comparison stays fair: same rows, same labels, same features.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from reviewlens.data.split import UNLABELED

IntArray = npt.NDArray[np.int_]
FloatArray = npt.NDArray[np.float64]

__all__ = ["UNLABELED", "Classifier", "FloatArray", "IntArray", "labeled_rows"]


@runtime_checkable
class Classifier(Protocol):
    """Anything with these three methods can be used by the experiment runner."""

    def fit(self, X: Any, y: npt.ArrayLike) -> Classifier:
        """Train on features ``X`` and labels ``y`` (-1 marks unlabeled rows)."""
        ...

    def predict(self, X: Any) -> IntArray:
        """Predicted class id for each row."""
        ...

    def predict_proba(self, X: Any) -> FloatArray:
        """Class probabilities, shape ``(n_rows, n_classes)``; each row sums to 1."""
        ...


def labeled_rows(y: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Boolean mask of labeled rows; raises if there are none to learn from."""
    mask: npt.NDArray[np.bool_] = np.asarray(y) != UNLABELED
    if not mask.any():
        raise ValueError("No labeled rows: every label is UNLABELED (-1).")
    return mask
