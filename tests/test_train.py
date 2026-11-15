from __future__ import annotations

from pathlib import Path

import pytest

from reviewlens.config import AppConfig
from reviewlens.experiments.train import build_experiment_features, train_and_evaluate
from reviewlens.models.text_classifier import (
    TextClassifier,
    load_text_classifier,
    save_text_classifier,
)

from .conftest import ToyData


def test_train_and_evaluate_returns_a_complete_result(
    toy_data: ToyData, toy_config: AppConfig
) -> None:
    features = build_experiment_features(
        toy_data.train_texts, toy_data.test_texts, ["logistic_regression"], toy_config
    )
    _, result = train_and_evaluate(
        "logistic_regression", features, toy_data.y_train, toy_data.y_test, 0.1, 0, toy_config
    )

    assert result.n_labeled == 12
    assert result.metrics["macro_f1"] > 0.8
    row = result.as_row()
    assert {"method", "label_fraction", "seed", "macro_f1", "recall_negative"} <= set(row)
    assert sum(map(sum, result.confusion)) == len(toy_data.y_test)


def test_lsa_is_built_only_when_a_method_needs_it(toy_data: ToyData, toy_config: AppConfig) -> None:
    features = build_experiment_features(
        toy_data.train_texts, toy_data.test_texts, ["naive_bayes"], toy_config
    )
    assert features.lsa is None


def test_text_classifier_roundtrip_predicts_raw_text(
    toy_data: ToyData, toy_config: AppConfig, tmp_path: Path
) -> None:
    features = build_experiment_features(
        toy_data.train_texts, toy_data.test_texts, ["logistic_regression"], toy_config
    )
    model, _ = train_and_evaluate(
        "logistic_regression", features, toy_data.y_train, toy_data.y_test, 1.0, 0, toy_config
    )
    classifier = TextClassifier(
        "logistic_regression", ["negative", "positive"], features.vectorizer, model
    )

    path = save_text_classifier(classifier, tmp_path / "model.joblib")
    loaded = load_text_classifier(path)

    # Raw, unnormalized input (Arabic yeh in "عالي") goes through the same normalization.
    assert loaded.predict_labels(["غذا عالي و خوشمزه بود", "غذا سرد و افتضاح بود"]) == [
        "positive",
        "negative",
    ]
    assert loaded.predict_proba(["غذا عالی"]).shape == (1, 2)


def test_loading_missing_model_gives_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="reviewlens train"):
        load_text_classifier(tmp_path / "missing.joblib")
