from __future__ import annotations

import pytest

from reviewlens.evaluation.metrics import compute_confusion, compute_metrics

LABELS = ["negative", "positive"]


def test_perfect_predictions() -> None:
    metrics = compute_metrics([0, 1, 1, 0], [0, 1, 1, 0], LABELS)
    assert metrics["macro_f1"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_known_values() -> None:
    # true:  0 0 0 0 1 1    pred: 0 0 0 1 1 0
    # negative: precision 3/4, recall 3/4 ; positive: precision 1/2, recall 1/2
    metrics = compute_metrics([0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 0], LABELS)

    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["precision_negative"] == pytest.approx(0.75)
    assert metrics["recall_positive"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx((0.75 + 0.5) / 2)


def test_macro_f1_punishes_ignoring_a_class_more_than_accuracy() -> None:
    y_true = [0] * 9 + [1]
    y_pred = [0] * 10  # always predicts the majority class
    metrics = compute_metrics(y_true, y_pred, LABELS)

    assert metrics["accuracy"] == pytest.approx(0.9)
    assert metrics["macro_f1"] < 0.5
    assert metrics["recall_positive"] == 0.0


def test_confusion_matrix_rows_are_true_classes() -> None:
    matrix = compute_confusion([0, 0, 1, 1], [0, 1, 1, 1], n_labels=2)
    assert matrix.tolist() == [[1, 1], [0, 2]]
