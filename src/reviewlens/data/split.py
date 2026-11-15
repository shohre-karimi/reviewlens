"""Simulate a partly labeled training set by hiding labels.

The training texts are all kept (they become the *unlabeled* pool), but only a fraction of
their labels stays visible; the rest are replaced by ``UNLABELED`` (-1), the convention
scikit-learn's semi-supervised estimators use.

Masking is **stratified**: each class keeps the same fraction of labels, so a 1 % labeled set
has the same class balance as the full training set.

Masking is **nested** for a fixed seed: each class is shuffled once with the seed and the first
``n`` items are kept, so the labeled samples at 1 % are a subset of those at 5 %, which are a
subset of those at 10 %, and so on. Differences between label fractions therefore come from
*more* labels, not from *different* labels.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

UNLABELED = -1

IntArray = npt.NDArray[np.int_]


def mask_labels(labels: npt.ArrayLike, fraction: float, seed: int) -> IntArray:
    """Return a copy of ``labels`` where all but ``fraction`` of each class is set to -1.

    Every class keeps at least one label, so even tiny fractions see all classes.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    y = np.asarray(labels, dtype=int)
    if (y == UNLABELED).any():
        raise ValueError("labels already contain UNLABELED (-1) values")

    rng = np.random.default_rng(seed)
    masked = np.full_like(y, UNLABELED)
    for cls in np.unique(y):
        class_indices = np.flatnonzero(y == cls)
        shuffled = rng.permutation(class_indices)
        n_keep = max(1, round(fraction * len(class_indices)))
        masked[shuffled[:n_keep]] = cls
    return masked


def labeled_mask(masked_labels: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Boolean array: True where the label is visible."""
    visible: npt.NDArray[np.bool_] = np.asarray(masked_labels) != UNLABELED
    return visible
