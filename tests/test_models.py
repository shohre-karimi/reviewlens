from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from reviewlens.config import FeaturesConfig
from reviewlens.data.split import mask_labels
from reviewlens.features import build_features
from reviewlens.models.base import Classifier
from reviewlens.models.registry import available_models, create_model, get_spec
from reviewlens.models.supervised import SupervisedModel

from .conftest import ToyData

SUPERVISED = ["naive_bayes", "logistic_regression", "linear_svm", "random_forest"]
FAST_PARAMS = {"random_forest": {"n_estimators": 20, "n_jobs": 1}}


def _tfidf(toy: ToyData) -> tuple[Any, Any]:
    features = build_features(toy.train_texts, toy.test_texts, FeaturesConfig(min_df=1), seed=0)
    return features.get("tfidf")


def test_registry_contains_the_supervised_baselines() -> None:
    assert set(SUPERVISED) <= set(available_models())
    assert all(not get_spec(name).semi_supervised for name in SUPERVISED)


def test_unknown_model_lists_available_ones() -> None:
    with pytest.raises(ValueError, match=r"Available: .*logistic_regression"):
        create_model("magic_model", seed=0)


def test_misspelled_hyperparameter_gives_clear_error() -> None:
    with pytest.raises(ValueError, match="Invalid parameters for model 'logistic_regression'"):
        create_model("logistic_regression", seed=0, params={"c": 1.0})


@pytest.mark.parametrize("name", SUPERVISED)
def test_supervised_model_follows_interface_and_learns(name: str, toy_data: ToyData) -> None:
    X_train, X_test = _tfidf(toy_data)
    model = create_model(name, seed=0, params=FAST_PARAMS.get(name))

    assert isinstance(model, Classifier)
    model.fit(X_train, toy_data.y_train)

    proba = model.predict_proba(X_test)
    assert proba.shape == (len(toy_data.y_test), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    accuracy = (model.predict(X_test) == toy_data.y_test).mean()
    assert accuracy >= 0.9  # the toy corpus is trivially separable


@pytest.mark.parametrize("name", SUPERVISED)
def test_supervised_model_trains_on_labeled_rows_only(name: str, toy_data: ToyData) -> None:
    X_train, _ = _tfidf(toy_data)
    y_masked = mask_labels(toy_data.y_train, fraction=0.1, seed=0)  # 6 + 6 labeled rows

    model = create_model(name, seed=0, params=FAST_PARAMS.get(name))
    model.fit(X_train, y_masked)

    assert isinstance(model, SupervisedModel)
    assert model.n_train_ == 12


def test_fit_without_any_labels_raises(toy_data: ToyData) -> None:
    X_train, _ = _tfidf(toy_data)
    model = create_model("logistic_regression", seed=0)
    with pytest.raises(ValueError, match="No labeled rows"):
        model.fit(X_train, np.full(len(toy_data.y_train), -1))


def test_same_seed_gives_identical_predictions(toy_data: ToyData) -> None:
    X_train, X_test = _tfidf(toy_data)
    params = FAST_PARAMS["random_forest"]
    first = create_model("random_forest", seed=3, params=params).fit(X_train, toy_data.y_train)
    second = create_model("random_forest", seed=3, params=params).fit(X_train, toy_data.y_train)
    np.testing.assert_array_equal(first.predict_proba(X_test), second.predict_proba(X_test))
