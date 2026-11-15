"""Model registry: one table from a method name (as written in the config) to how it is built.

Adding a model means writing a factory function and adding one line to ``MODEL_REGISTRY``;
the experiment runner, the CLI and the config validation pick it up from here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from reviewlens.features import FeatureKind
from reviewlens.models import semi_supervised, supervised
from reviewlens.models.base import Classifier


@dataclass(frozen=True)
class ModelSpec:
    """How to build a model and what it needs."""

    factory: Callable[..., Classifier]
    features: FeatureKind  # which feature matrix the model is trained on
    semi_supervised: bool  # does it learn from unlabeled rows?
    description: str


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "naive_bayes": ModelSpec(
        supervised.make_naive_bayes, "tfidf", False, "Multinomial Naive Bayes"
    ),
    "logistic_regression": ModelSpec(
        supervised.make_logistic_regression, "tfidf", False, "Logistic Regression"
    ),
    "linear_svm": ModelSpec(supervised.make_linear_svm, "tfidf", False, "Linear SVM (calibrated)"),
    "random_forest": ModelSpec(supervised.make_random_forest, "tfidf", False, "Random Forest"),
    "self_training": ModelSpec(
        semi_supervised.make_self_training, "tfidf", True, "Self-Training (Logistic Regression)"
    ),
    "label_spreading": ModelSpec(
        semi_supervised.make_label_spreading, "lsa", True, "Label Spreading (kNN graph on LSA)"
    ),
    "pseudo_labeling": ModelSpec(
        semi_supervised.make_pseudo_labeling, "tfidf", True, "Pseudo-Labeling (from scratch)"
    ),
}


def available_models() -> list[str]:
    return list(MODEL_REGISTRY)


def get_spec(name: str) -> ModelSpec:
    """Look up a model by name; unknown names list the valid ones."""
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown model '{name}'. Available: {', '.join(available_models())}"
        ) from None


def create_model(name: str, seed: int, params: Mapping[str, Any] | None = None) -> Classifier:
    """Build a fresh, unfitted model with the given seed and hyperparameters."""
    spec = get_spec(name)
    try:
        return spec.factory(seed=seed, **dict(params or {}))
    except TypeError as error:  # e.g. a misspelled hyperparameter in the config
        raise ValueError(f"Invalid parameters for model '{name}': {error}") from error
