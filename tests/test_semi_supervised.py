from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from reviewlens.config import FeaturesConfig
from reviewlens.data.split import UNLABELED, mask_labels
from reviewlens.features import FeatureKind, build_features
from reviewlens.models.base import Classifier
from reviewlens.models.registry import create_model, get_spec
from reviewlens.models.semi_supervised import (
    LabelSpreadingModel,
    PseudoLabelingModel,
    PseudoLabelStep,
    ReportsPseudoLabels,
    SymmetricKnnKernel,
    steps_from_iterations,
)

from .conftest import ToyData

SEMI = ["self_training", "label_spreading", "pseudo_labeling"]


def _features(toy: ToyData, kind: FeatureKind) -> tuple[Any, Any]:
    cfg = FeaturesConfig(min_df=1, lsa_components=5)
    features = build_features(toy.train_texts, toy.test_texts, cfg, seed=0, with_lsa=kind == "lsa")
    return features.get(kind)


# ------------------------------------------------------------------ pseudo-label history


def test_steps_from_iterations_counts_additions_and_correct_ones() -> None:
    labeled_iter = [0, 0, 1, 1, 2, -1]
    assigned = [0, 1, 1, 0, 1, -1]
    y_true = [0, 1, 1, 1, 0, 0]

    steps = steps_from_iterations(labeled_iter, assigned, y_true)

    assert steps == [PseudoLabelStep(1, 2, 1), PseudoLabelStep(2, 1, 0)]
    assert steps[0].accuracy == 0.5


def test_step_without_additions_has_nan_accuracy() -> None:
    assert math.isnan(PseudoLabelStep(1, 0, 0).accuracy)


def test_no_pseudo_labels_gives_empty_history() -> None:
    assert steps_from_iterations([0, 0, -1], [0, 1, -1], [0, 1, 1]) == []


# ------------------------------------------------------------------ all three methods


def test_registry_marks_semi_supervised_methods() -> None:
    assert all(get_spec(name).semi_supervised for name in SEMI)
    assert get_spec("label_spreading").features == "lsa"


@pytest.mark.parametrize("name", SEMI)
def test_semi_supervised_model_learns_from_few_labels(name: str, toy_data: ToyData) -> None:
    X_train, X_test = _features(toy_data, get_spec(name).features)
    y_masked = mask_labels(toy_data.y_train, fraction=0.1, seed=0)  # 12 of 120 labeled

    model = create_model(name, seed=0)
    assert isinstance(model, Classifier)
    assert isinstance(model, ReportsPseudoLabels)
    model.fit(X_train, y_masked)

    proba = model.predict_proba(X_test)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert (model.predict(X_test) == toy_data.y_test).mean() >= 0.8

    history = model.pseudo_label_history(toy_data.y_train)
    n_unlabeled = int((y_masked == UNLABELED).sum())
    assert sum(step.n_added for step in history) <= n_unlabeled
    assert all(0 <= step.n_correct <= step.n_added for step in history)


@pytest.mark.parametrize("name", SEMI)
def test_semi_supervised_model_needs_at_least_one_label(name: str, toy_data: ToyData) -> None:
    X_train, _ = _features(toy_data, get_spec(name).features)
    with pytest.raises(ValueError, match="No labeled rows"):
        create_model(name, seed=0).fit(X_train, np.full(len(toy_data.y_train), UNLABELED))


# ------------------------------------------------------------------ Pseudo-Labeling


def test_pseudo_labeling_adds_the_same_number_per_class(toy_data: ToyData) -> None:
    X_train, _ = _features(toy_data, "tfidf")
    y_masked = mask_labels(toy_data.y_train, fraction=0.1, seed=0)

    model = create_model("pseudo_labeling", seed=0, params={"threshold": 0.5}).fit(
        X_train, y_masked
    )

    assert isinstance(model, PseudoLabelingModel)
    assert model.n_iter_ >= 1
    for k in range(1, model.n_iter_ + 1):
        added = model.labeled_iter_ == k
        per_class = np.bincount(model.transduction_[added], minlength=2)
        assert per_class[0] == per_class[1]
        assert per_class[0] <= 6  # growth cap: ceil(1.0 * 6 labeled rows per class)


class _AlwaysPositive:
    """Fake base model that is confident about class 1 only."""

    classes_ = np.array([0, 1])

    def fit(self, X: Any, y: Any) -> _AlwaysPositive:
        return self

    def predict(self, X: Any) -> Any:
        return np.ones(X.shape[0], dtype=int)

    def predict_proba(self, X: Any) -> Any:
        return np.tile([0.01, 0.99], (X.shape[0], 1))


def test_pseudo_labeling_stops_when_a_class_has_no_candidates() -> None:
    X = np.zeros((10, 2))
    y = np.array([0, 1] + [UNLABELED] * 8)

    model = PseudoLabelingModel(_AlwaysPositive, threshold=0.9).fit(X, y)

    assert model.n_iter_ == 0  # adding only positives would unbalance the labeled set
    assert model.pseudo_label_history(np.zeros(10, dtype=int)) == []


@pytest.mark.parametrize(("threshold", "growth"), [(0.0, 1.0), (1.5, 1.0), (0.9, 0.0)])
def test_pseudo_labeling_rejects_invalid_settings(threshold: float, growth: float) -> None:
    with pytest.raises(ValueError):
        PseudoLabelingModel(_AlwaysPositive, threshold=threshold, growth=growth)


# ------------------------------------------------------------------ Label Spreading


def test_symmetric_knn_kernel_builds_symmetric_graph() -> None:
    rng = np.random.default_rng(0)
    X_train, X_new = rng.normal(size=(30, 4)), rng.normal(size=(5, 4))
    kernel = SymmetricKnnKernel(n_neighbors=3, n_jobs=1)

    graph = kernel(X_train, X_train)
    assert graph.shape == (30, 30)
    assert (graph != graph.T).nnz == 0

    links = kernel(X_train, X_new)
    assert links.shape == (30, 5)
    np.testing.assert_array_equal(np.asarray(links.sum(axis=0)).ravel(), 3)


def test_symmetric_knn_kernel_requires_training_graph_first() -> None:
    kernel = SymmetricKnnKernel(n_neighbors=3, n_jobs=1)
    with pytest.raises(RuntimeError):
        kernel(np.zeros((5, 2)), np.zeros((2, 2)))


def test_label_spreading_replaces_nan_probabilities_with_prior(
    toy_data: ToyData, monkeypatch: pytest.MonkeyPatch
) -> None:
    X_train, X_test = _features(toy_data, "lsa")
    y_masked = mask_labels(toy_data.y_train, fraction=0.1, seed=0)
    model = create_model("label_spreading", seed=0).fit(X_train, y_masked)
    assert isinstance(model, LabelSpreadingModel)

    def nan_rows(X: Any) -> Any:
        proba = np.full((X.shape[0], 2), 0.5)
        proba[0] = np.nan
        return proba

    monkeypatch.setattr(model.estimator, "predict_proba", nan_rows)
    proba = model.predict_proba(X_test)

    assert model.n_fallback_ == 1
    np.testing.assert_allclose(proba[0], [0.5, 0.5])  # balanced labeled set -> prior 0.5 / 0.5
    assert not np.isnan(proba).any()
