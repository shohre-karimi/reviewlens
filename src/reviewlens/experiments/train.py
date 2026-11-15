"""One training run: one method, one label fraction, one seed.

    hide labels (mask_labels) -> build model -> fit -> predict test -> metrics

Features are passed in rather than built here: TF-IDF and LSA use no labels, so they are the
same for every method, fraction and seed and are computed once per experiment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt

from reviewlens.config import AppConfig
from reviewlens.data.split import labeled_mask, mask_labels
from reviewlens.evaluation.metrics import compute_confusion, compute_metrics
from reviewlens.features import FeatureSet, build_features
from reviewlens.models.base import Classifier
from reviewlens.models.registry import create_model, get_spec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunResult:
    method: str
    label_fraction: float
    seed: int
    n_labeled: int
    fit_seconds: float
    metrics: dict[str, float]
    confusion: list[list[int]]

    def as_row(self) -> dict[str, Any]:
        """Flat dict for one row of a results table."""
        return {
            "method": self.method,
            "label_fraction": self.label_fraction,
            "seed": self.seed,
            "n_labeled": self.n_labeled,
            "fit_seconds": round(self.fit_seconds, 3),
            **self.metrics,
        }


def train_and_evaluate(
    method: str,
    features: FeatureSet,
    y_train: npt.ArrayLike,
    y_test: npt.ArrayLike,
    label_fraction: float,
    seed: int,
    cfg: AppConfig,
) -> tuple[Classifier, RunResult]:
    """Train ``method`` on ``label_fraction`` of the training labels and evaluate on test."""
    spec = get_spec(method)
    X_train, X_test = features.get(spec.features)
    y_masked = mask_labels(y_train, label_fraction, seed)

    model = create_model(method, seed, cfg.models.get(method))
    start = perf_counter()
    model.fit(X_train, y_masked)
    fit_seconds = perf_counter() - start

    y_pred = model.predict(X_test)
    result = RunResult(
        method=method,
        label_fraction=label_fraction,
        seed=seed,
        n_labeled=int(labeled_mask(y_masked).sum()),
        fit_seconds=fit_seconds,
        metrics=compute_metrics(y_test, y_pred, cfg.task.labels),
        confusion=compute_confusion(y_test, y_pred, len(cfg.task.labels)).tolist(),
    )
    logger.info(
        "Run done method=%s fraction=%s seed=%d n_labeled=%d macro_f1=%.4f fit_seconds=%.1f",
        method,
        label_fraction,
        seed,
        result.n_labeled,
        result.metrics["macro_f1"],
        fit_seconds,
    )
    return model, result


def build_experiment_features(
    train_texts: npt.ArrayLike, test_texts: npt.ArrayLike, methods: list[str], cfg: AppConfig
) -> FeatureSet:
    """Build the features once; LSA only if one of ``methods`` needs it (it is slow)."""
    with_lsa = any(get_spec(method).features == "lsa" for method in methods)
    return build_features(
        list(np.asarray(train_texts)),
        list(np.asarray(test_texts)),
        cfg.features,
        seed=cfg.seed,
        with_lsa=with_lsa,
    )
