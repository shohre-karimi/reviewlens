from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from reviewlens.data.split import UNLABELED, labeled_mask, mask_labels

# 600 negatives and 400 positives: an imbalanced set makes stratification visible.
LABELS = np.array([0] * 600 + [1] * 400)


def test_mask_keeps_fraction_per_class() -> None:
    masked = mask_labels(LABELS, fraction=0.05, seed=0)

    assert (masked == 0).sum() == 30  # 5 % of 600
    assert (masked == 1).sum() == 20  # 5 % of 400
    assert (masked == UNLABELED).sum() == 950


def test_visible_labels_are_the_true_labels() -> None:
    masked = mask_labels(LABELS, fraction=0.1, seed=3)
    visible = labeled_mask(masked)
    assert np.array_equal(masked[visible], LABELS[visible])


def test_input_is_not_modified() -> None:
    labels = LABELS.copy()
    mask_labels(labels, fraction=0.1, seed=0)
    assert np.array_equal(labels, LABELS)


def test_full_fraction_keeps_everything() -> None:
    assert np.array_equal(mask_labels(LABELS, fraction=1.0, seed=0), LABELS)


def test_every_class_keeps_at_least_one_label() -> None:
    masked = mask_labels(np.array([0] * 50 + [1] * 5), fraction=0.01, seed=0)
    assert (masked == 0).sum() == 1
    assert (masked == 1).sum() == 1


def test_same_seed_is_deterministic_and_different_seeds_differ() -> None:
    first = mask_labels(LABELS, fraction=0.05, seed=7)
    again = mask_labels(LABELS, fraction=0.05, seed=7)
    other = mask_labels(LABELS, fraction=0.05, seed=8)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_labeled_sets_are_nested_across_fractions() -> None:
    fractions = [0.01, 0.05, 0.1, 0.2, 1.0]
    masks = [labeled_mask(mask_labels(LABELS, fraction=f, seed=1)) for f in fractions]

    for smaller, larger in pairwise(masks):
        assert np.all(larger[smaller]), "labeled set must grow, never swap samples"


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_invalid_fraction_raises(fraction: float) -> None:
    with pytest.raises(ValueError, match="fraction"):
        mask_labels(LABELS, fraction=fraction, seed=0)


def test_already_masked_input_raises() -> None:
    with pytest.raises(ValueError, match="UNLABELED"):
        mask_labels(np.array([0, 1, -1]), fraction=0.5, seed=0)
