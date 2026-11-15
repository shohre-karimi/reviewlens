"""Text features: sparse TF-IDF and dense LSA.

Leakage rule: the vectorizer and the SVD are *fit* on training texts only (labeled and
unlabeled together, since fitting them uses no labels) and only *applied* to the test texts.
A word that appears only in the test set therefore has no feature.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Normalizer

from reviewlens.config import FeaturesConfig

logger = logging.getLogger(__name__)

FeatureKind = Literal["tfidf", "lsa"]

# A token is two or more word characters; half-spaces (ZWNJ, U+200C) are allowed *inside* it.
# scikit-learn's default pattern (\b\w\w+\b) treats the half-space as a separator and would
# split "بی‌کیفیت" into "بی" + "کیفیت".
TOKEN_PATTERN = r"(?u)\w[\w\u200c]*\w"


def build_vectorizer(cfg: FeaturesConfig) -> TfidfVectorizer:
    """Create an unfitted TF-IDF vectorizer from the config."""
    kwargs: dict[str, Any] = {
        "analyzer": cfg.analyzer,
        "ngram_range": (cfg.ngram_min, cfg.ngram_max),
        "min_df": cfg.min_df,
        "max_df": cfg.max_df,
        "max_features": cfg.max_features,
        "sublinear_tf": cfg.sublinear_tf,
        "lowercase": False,  # texts are already normalized (and lowercased) in preprocessing
        "dtype": np.float32,  # half the memory of float64, same results for linear models
    }
    if cfg.analyzer == "word":
        kwargs["token_pattern"] = TOKEN_PATTERN
    return TfidfVectorizer(**kwargs)


def build_lsa(n_components: int, seed: int) -> Pipeline:
    """LSA = truncated SVD of the TF-IDF matrix, then L2-normalize each row.

    Normalizing makes the dot product of two rows their cosine similarity, which is what
    graph-based methods such as Label Spreading compare.
    """
    return make_pipeline(
        TruncatedSVD(n_components=n_components, random_state=seed), Normalizer(copy=False)
    )


@dataclass
class FeatureSet:
    """Train/test matrices for each feature kind, plus the fitted transformers."""

    vectorizer: TfidfVectorizer
    train_tfidf: Any  # scipy.sparse CSR matrix, shape (n_train, n_terms)
    test_tfidf: Any
    lsa: Pipeline | None = None
    train_lsa: np.ndarray | None = None  # dense, shape (n_train, lsa_components)
    test_lsa: np.ndarray | None = None

    def get(self, kind: FeatureKind) -> tuple[Any, Any]:
        """Return ``(X_train, X_test)`` for one feature kind."""
        if kind == "tfidf":
            return self.train_tfidf, self.test_tfidf
        if self.train_lsa is None or self.test_lsa is None:
            raise ValueError("LSA features were not built; call build_features(with_lsa=True)")
        return self.train_lsa, self.test_lsa


def build_features(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    cfg: FeaturesConfig,
    seed: int,
    with_lsa: bool = False,
) -> FeatureSet:
    """Fit TF-IDF (and optionally LSA) on the training texts and transform both splits."""
    vectorizer = build_vectorizer(cfg)
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)
    logger.info(
        "TF-IDF fitted analyzer=%s n_terms=%d n_train=%d",
        cfg.analyzer,
        train_tfidf.shape[1],
        train_tfidf.shape[0],
    )
    features = FeatureSet(vectorizer=vectorizer, train_tfidf=train_tfidf, test_tfidf=test_tfidf)

    if with_lsa:
        n_components = min(cfg.lsa_components, train_tfidf.shape[1] - 1)
        lsa = build_lsa(n_components, seed)
        features.lsa = lsa
        features.train_lsa = lsa.fit_transform(train_tfidf)
        features.test_lsa = lsa.transform(test_tfidf)
        explained = lsa.named_steps["truncatedsvd"].explained_variance_ratio_.sum()
        logger.info("LSA fitted components=%d explained_variance=%.3f", n_components, explained)
    return features
