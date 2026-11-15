"""Feature extraction: TF-IDF, LSA and optional transformer embeddings."""

from reviewlens.features.text import (
    TOKEN_PATTERN,
    FeatureKind,
    FeatureSet,
    build_features,
    build_lsa,
    build_vectorizer,
)

__all__ = [
    "TOKEN_PATTERN",
    "FeatureKind",
    "FeatureSet",
    "build_features",
    "build_lsa",
    "build_vectorizer",
]
