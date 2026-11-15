"""Supervised baselines: they learn from the labeled rows only.

Each ``make_*`` factory takes the run's ``seed`` plus hyperparameters (from the ``models``
section of the config) and returns a ``SupervisedModel``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from reviewlens.models.base import FloatArray, IntArray, labeled_rows


class SupervisedModel:
    """Wrap a scikit-learn classifier so it follows the project's ``Classifier`` interface.

    ``fit`` drops the unlabeled rows (label -1) before training, so a supervised model sees
    exactly the labeled fraction and nothing else.
    """

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator
        self.n_train_: int = 0  # number of labeled rows used in the last fit

    def fit(self, X: Any, y: npt.ArrayLike) -> SupervisedModel:
        y_arr = np.asarray(y)
        mask = labeled_rows(y_arr)
        self.estimator.fit(X[mask], y_arr[mask])
        self.n_train_ = int(mask.sum())
        return self

    def predict(self, X: Any) -> IntArray:
        return np.asarray(self.estimator.predict(X), dtype=int)

    def predict_proba(self, X: Any) -> FloatArray:
        return np.asarray(self.estimator.predict_proba(X), dtype=float)

    def __repr__(self) -> str:
        return f"SupervisedModel({self.estimator!r})"


def make_naive_bayes(seed: int, alpha: float = 1.0) -> SupervisedModel:
    """Multinomial Naive Bayes. Deterministic, so ``seed`` is unused."""
    del seed
    return SupervisedModel(MultinomialNB(alpha=alpha))


def make_logistic_regression(seed: int, C: float = 1.0, max_iter: int = 1000) -> SupervisedModel:
    """L2-regularized logistic regression; probabilities come from the model itself."""
    return SupervisedModel(LogisticRegression(C=C, max_iter=max_iter, random_state=seed))


def make_linear_svm(seed: int, C: float = 1.0, calibration_folds: int = 3) -> SupervisedModel:
    """Linear SVM with sigmoid (Platt) calibration.

    A plain SVM gives only distances to the decision boundary, not probabilities.
    ``CalibratedClassifierCV`` fits a sigmoid on held-out folds to turn those distances
    into probabilities, which Self-Training and the agent's confidence both need.
    """
    svm = LinearSVC(C=C, random_state=seed)
    return SupervisedModel(
        CalibratedClassifierCV(estimator=svm, method="sigmoid", cv=calibration_folds)
    )


def make_random_forest(
    seed: int, n_estimators: int = 200, max_features: str = "sqrt", n_jobs: int = -1
) -> SupervisedModel:
    """Random forest; ``n_jobs=-1`` trains trees on all CPU cores."""
    return SupervisedModel(
        RandomForestClassifier(
            n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs, random_state=seed
        )
    )
