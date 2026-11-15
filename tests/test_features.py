from __future__ import annotations

import numpy as np
import pytest

from reviewlens.config import FeaturesConfig
from reviewlens.features import build_features, build_vectorizer

ZWNJ = "‌"
WORD_CFG = FeaturesConfig(min_df=1, ngram_max=1, lsa_components=3)


def _vocabulary(texts: list[str], cfg: FeaturesConfig = WORD_CFG) -> set[str]:
    vectorizer = build_vectorizer(cfg)
    vectorizer.fit(texts)
    return set(vectorizer.vocabulary_)


def test_half_space_words_stay_one_token() -> None:
    vocab = _vocabulary([f"غذا بی{ZWNJ}کیفیت بود", "خیلی خوب"])
    assert f"بی{ZWNJ}کیفیت" in vocab
    assert "کیفیت" not in vocab


def test_single_characters_and_punctuation_are_not_tokens() -> None:
    vocab = _vocabulary(["افتضاح. و بد!", "خوب ،"])
    assert vocab == {"افتضاح", "بد", "خوب"}


def test_bigrams_are_built_when_configured() -> None:
    vocab = _vocabulary(["غذا سرد بود", "پیک دیر رسید"], FeaturesConfig(min_df=1, ngram_max=2))
    assert "غذا سرد" in vocab


def test_char_analyzer_builds_character_ngrams() -> None:
    cfg = FeaturesConfig(analyzer="char_wb", ngram_min=2, ngram_max=3, min_df=1)
    vocab = _vocabulary(["سرد", "گرم"], cfg)
    assert " سر" in vocab  # char_wb pads each word with spaces
    assert "سرد" in vocab


def test_vectorizer_is_fit_on_train_only() -> None:
    features = build_features(["غذا خوب بود", "غذا بد بود"], ["پیتزا عالی"], WORD_CFG, seed=0)

    vocab = set(features.vectorizer.vocabulary_)
    assert "پیتزا" not in vocab and "عالی" not in vocab  # test-only words get no feature
    assert features.test_tfidf.shape == (1, len(vocab))
    assert features.test_tfidf.nnz == 0


def test_lsa_features_are_dense_and_row_normalized() -> None:
    train = ["غذا خوب بود", "غذا بد بود", "پیک سریع بود", "پیک دیر رسید", "سفارش عالی"]
    features = build_features(train, ["غذا خوب"], WORD_CFG, seed=0, with_lsa=True)

    X_train, X_test = features.get("lsa")
    assert isinstance(X_train, np.ndarray)
    assert X_train.shape == (5, 3)
    np.testing.assert_allclose(np.linalg.norm(X_train, axis=1), 1.0, rtol=1e-5)
    assert X_test.shape == (1, 3)


def test_requesting_lsa_without_building_it_raises() -> None:
    features = build_features(["غذا خوب", "غذا بد"], ["خوب"], WORD_CFG, seed=0)
    with pytest.raises(ValueError, match="LSA"):
        features.get("lsa")
