"""Semi-supervised methods: they learn from labeled *and* unlabeled rows.

All three follow the same ``Classifier`` interface as the supervised models (``fit(X, y)``
with -1 for unlabeled rows), and all three can report their pseudo-labels:

    model.pseudo_label_history(y_true) -> one PseudoLabelStep per iteration

``y_true`` holds the real labels of the training rows, which the model never saw. They are
used *only* for this diagnostic, never for training: it shows how many pseudo-labels each
iteration added and how many of them were correct, which explains why a method helps or not.

Methods:
- Self-Training (scikit-learn): repeatedly add confident predictions as labels.
- Label Spreading (scikit-learn): spread labels over a k-nearest-neighbour graph of LSA vectors.
- Pseudo-Labeling (written from scratch here): like Self-Training, but class-balanced and with
  a cap on how fast the labeled set may grow.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier

from reviewlens.models.base import FloatArray, IntArray, labeled_rows

NEVER_LABELED = -1  # value in ``labeled_iter_`` for rows that never received a label


@dataclass(frozen=True)
class PseudoLabelStep:
    """Pseudo-labels added in one iteration, checked against the hidden true labels."""

    iteration: int
    n_added: int
    n_correct: int

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_added if self.n_added else math.nan

    def as_dict(self) -> dict[str, float | int]:
        return {**asdict(self), "accuracy": self.accuracy}


@runtime_checkable
class ReportsPseudoLabels(Protocol):
    def pseudo_label_history(self, y_true: npt.ArrayLike) -> list[PseudoLabelStep]: ...


def steps_from_iterations(
    labeled_iter: npt.ArrayLike, assigned: npt.ArrayLike, y_true: npt.ArrayLike
) -> list[PseudoLabelStep]:
    """Build the history from ``labeled_iter`` (scikit-learn's Self-Training format).

    ``labeled_iter[i] == k`` means row ``i`` got its label in iteration ``k``:
    0 = it was labeled from the start, -1 = it never got a label.
    """
    iters, labels, truth = np.asarray(labeled_iter), np.asarray(assigned), np.asarray(y_true)
    steps = []
    for k in range(1, int(iters.max(initial=0)) + 1):
        added = iters == k
        steps.append(
            PseudoLabelStep(k, int(added.sum()), int((labels[added] == truth[added]).sum()))
        )
    return steps


# --------------------------------------------------------------------------- Self-Training


class SelfTrainingModel:
    """scikit-learn ``SelfTrainingClassifier``.

    Each iteration: fit the base model on the labeled rows, predict the unlabeled rows, and
    give every prediction with probability >= ``threshold`` its predicted label. Stop when no
    prediction passes the threshold or after ``max_iter`` iterations.
    """

    def __init__(self, base_estimator: Any, threshold: float, max_iter: int) -> None:
        self.estimator = SelfTrainingClassifier(
            estimator=base_estimator, threshold=threshold, max_iter=max_iter
        )

    def fit(self, X: Any, y: npt.ArrayLike) -> SelfTrainingModel:
        y_arr = np.asarray(y)
        mask = labeled_rows(y_arr)  # fail early if nothing is labeled
        with warnings.catch_warnings():
            if mask.all():  # 100 % labels: nothing to self-train on, which is expected here
                warnings.filterwarnings("ignore", message="y contains no unlabeled samples")
            self.estimator.fit(X, y_arr)
        return self

    def predict(self, X: Any) -> IntArray:
        return np.asarray(self.estimator.predict(X), dtype=int)

    def predict_proba(self, X: Any) -> FloatArray:
        return np.asarray(self.estimator.predict_proba(X), dtype=float)

    def pseudo_label_history(self, y_true: npt.ArrayLike) -> list[PseudoLabelStep]:
        return steps_from_iterations(
            self.estimator.labeled_iter_, self.estimator.transduction_, y_true
        )


# --------------------------------------------------------------------------- Label Spreading


class SymmetricKnnKernel:
    """k-nearest-neighbour graph where an edge exists if *either* row is a neighbour of the other.

    scikit-learn's built-in ``knn`` kernel is directed: row i only receives label information
    from its own k neighbours. A group of near-duplicate reviews that are all each other's
    nearest neighbours then forms a closed loop no label can enter (on this data ~3,000
    training rows never received any label mass, and 400-600 test rows got 0/0 = NaN
    probabilities). Making the graph symmetric opens those loops.

    Called by ``LabelSpreading`` as ``kernel(X_train, X_train)`` while fitting (returns the
    n_train x n_train graph) and ``kernel(X_train, X_new)`` while predicting (returns an
    n_train x n_new matrix linking each new row to its k nearest training rows).
    """

    def __init__(self, n_neighbors: int, n_jobs: int) -> None:
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self._nn: NearestNeighbors | None = None

    def __call__(self, X: Any, Y: Any) -> Any:
        if Y is X:  # building the training graph
            self._nn = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=self.n_jobs).fit(X)
            graph = self._nn.kneighbors_graph(X, mode="connectivity")
            return graph.maximum(graph.T).tocsr()
        if self._nn is None:
            raise RuntimeError("The kernel must build the training graph before predicting.")
        return self._nn.kneighbors_graph(Y, mode="connectivity").T.tocsr()


class LabelSpreadingModel:
    """scikit-learn ``LabelSpreading`` on a symmetric k-nearest-neighbour graph.

    Every training row becomes a node connected to its ``n_neighbors`` most similar rows
    (dense LSA vectors). Label information flows along the edges until it settles; ``alpha``
    is how much a node may move away from its initial label (0 = labels are fixed).

    A kNN graph keeps the problem sparse. The ``rbf`` kernel would need a dense
    n x n matrix (60k x 60k ~ 29 GB), which does not fit in memory.
    """

    def __init__(self, n_neighbors: int, alpha: float, max_iter: int, n_jobs: int) -> None:
        self.estimator = LabelSpreading(
            kernel=SymmetricKnnKernel(n_neighbors, n_jobs), alpha=alpha, max_iter=max_iter
        )
        self._unlabeled: npt.NDArray[np.bool_] | None = None
        self._prior: FloatArray | None = None
        self.n_fallback_ = 0  # predictions in the last predict_proba call that used the prior

    def fit(self, X: Any, y: npt.ArrayLike) -> LabelSpreadingModel:
        y_arr = np.asarray(y)
        mask = labeled_rows(y_arr)
        self.estimator.fit(X, y_arr)
        self._unlabeled = ~mask
        counts = np.bincount(y_arr[mask], minlength=len(self.estimator.classes_))
        self._prior = counts / counts.sum()
        return self

    def predict_proba(self, X: Any) -> FloatArray:
        """Class probabilities; a row that received no label mass gets the labeled class prior.

        Without this fallback such rows would be NaN, and ``argmax`` of NaN silently picks
        class 0, which biases predictions towards one class.
        """
        if self._prior is None:
            raise RuntimeError("Call fit() first.")
        with np.errstate(invalid="ignore", divide="ignore"):
            proba = np.asarray(self.estimator.predict_proba(X), dtype=float)
        missing = np.isnan(proba).any(axis=1)
        proba[missing] = self._prior
        self.n_fallback_ = int(missing.sum())
        return proba

    def predict(self, X: Any) -> IntArray:
        return np.asarray(self.estimator.classes_[self.predict_proba(X).argmax(axis=1)], dtype=int)

    def pseudo_label_history(self, y_true: npt.ArrayLike) -> list[PseudoLabelStep]:
        """Label Spreading labels every unlabeled row at once: one step."""
        if self._unlabeled is None:
            raise RuntimeError("Call fit() first.")
        labeled_iter = np.where(self._unlabeled, 1, 0)
        return steps_from_iterations(labeled_iter, self.estimator.transduction_, y_true)


# --------------------------------------------------------------------------- Pseudo-Labeling


class PseudoLabelingModel:
    """Iterative pseudo-labeling, implemented from scratch.

    Each iteration:
      1. fit a fresh base model on all rows labeled so far (real + pseudo labels);
      2. predict the still-unlabeled rows;
      3. for each class, find the candidates: rows predicted as that class with
         confidence >= ``threshold``;
      4. add the same number ``k`` of pseudo-labels to *every* class (the most confident
         candidates first), where ``k = min(cap, candidates of the weakest class)`` and
         ``cap = ceil(growth * smallest initial class count)``.
    Stop when some class has no candidate, nothing is left unlabeled, or after ``max_iter``.

    Two differences from scikit-learn's Self-Training, both meant to limit error build-up:
    - **strict class balance**: every class gains exactly ``k`` labels per iteration. A base
      model that is more confident about one class (Logistic Regression with 1 % labels is
      much more confident about positive reviews) cannot flood the labeled set with that
      class, which is what makes Self-Training collapse on this data;
    - **growth cap**: with ``growth=1`` the labeled set at most doubles per iteration, so the
      model is retrained before it labels the next, harder batch.
    """

    def __init__(
        self,
        base_factory: Callable[[], Any],
        threshold: float = 0.7,
        max_iter: int = 10,
        growth: float = 1.0,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        if growth <= 0:
            raise ValueError(f"growth must be > 0, got {growth}")
        self.base_factory = base_factory
        self.threshold = threshold
        self.max_iter = max_iter
        self.growth = growth

    def fit(self, X: Any, y: npt.ArrayLike) -> PseudoLabelingModel:
        y_work = np.asarray(y).copy()
        labeled = labeled_rows(y_work).copy()
        self.labeled_iter_ = np.where(labeled, 0, NEVER_LABELED)
        classes, counts = np.unique(y_work[labeled], return_counts=True)
        cap = max(1, math.ceil(self.growth * int(min(counts))))

        self.n_iter_ = 0
        for iteration in range(1, self.max_iter + 1):
            unlabeled_idx = np.flatnonzero(~labeled)
            if unlabeled_idx.size == 0:
                break
            model = self.base_factory().fit(X[labeled], y_work[labeled])
            proba = model.predict_proba(X[unlabeled_idx])
            confidence = proba.max(axis=1)
            predicted = model.classes_[proba.argmax(axis=1)]

            chosen = self._select(confidence, predicted, classes, cap)
            if chosen.size == 0:
                break
            rows = unlabeled_idx[chosen]
            y_work[rows] = predicted[chosen]
            labeled[rows] = True
            self.labeled_iter_[rows] = iteration
            self.n_iter_ = iteration

        self.transduction_ = y_work
        self.model_ = self.base_factory().fit(X[labeled], y_work[labeled])
        return self

    def _select(
        self, confidence: FloatArray, predicted: IntArray, classes: IntArray, cap: int
    ) -> IntArray:
        """Positions (in the unlabeled batch) of the pseudo-labels to add this iteration.

        Returns an empty array when any class has no candidate: adding labels to only some
        classes would unbalance the labeled set.
        """
        ranked: list[IntArray] = []
        for cls in classes:
            candidates = np.flatnonzero((predicted == cls) & (confidence >= self.threshold))
            ranked.append(candidates[np.argsort(-confidence[candidates], kind="stable")])
        k = min(cap, *(len(r) for r in ranked))
        if k == 0:
            return np.array([], dtype=int)
        return np.concatenate([r[:k] for r in ranked])

    def predict(self, X: Any) -> IntArray:
        return np.asarray(self.model_.predict(X), dtype=int)

    def predict_proba(self, X: Any) -> FloatArray:
        return np.asarray(self.model_.predict_proba(X), dtype=float)

    def pseudo_label_history(self, y_true: npt.ArrayLike) -> list[PseudoLabelStep]:
        return steps_from_iterations(self.labeled_iter_, self.transduction_, y_true)


# --------------------------------------------------------------------------- factories


def _logistic_regression(seed: int, C: float) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=1000, random_state=seed)


def make_self_training(
    seed: int, threshold: float = 0.75, max_iter: int = 10, C: float = 1.0
) -> SelfTrainingModel:
    """Self-Training around Logistic Regression (the strongest supervised baseline)."""
    return SelfTrainingModel(_logistic_regression(seed, C), threshold, max_iter)


def make_label_spreading(
    seed: int, n_neighbors: int = 7, alpha: float = 0.2, max_iter: int = 30, n_jobs: int = -1
) -> LabelSpreadingModel:
    """Label Spreading on LSA features. Deterministic, so ``seed`` is unused."""
    del seed
    return LabelSpreadingModel(n_neighbors, alpha, max_iter, n_jobs)


def make_pseudo_labeling(
    seed: int, threshold: float = 0.7, max_iter: int = 10, growth: float = 1.0, C: float = 1.0
) -> PseudoLabelingModel:
    """From-scratch pseudo-labeling around Logistic Regression."""
    return PseudoLabelingModel(
        lambda: _logistic_regression(seed, C), threshold=threshold, max_iter=max_iter, growth=growth
    )
