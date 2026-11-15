"""End-to-end preparation on the tiny fixture dataset (see tests/fixtures/README.md)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reviewlens.config import AppConfig
from reviewlens.data.prepare import (
    load_prepared,
    prepare_dataset,
    remove_test_overlap,
    save_prepared,
)


def test_prepare_dataset_reports_every_cleaning_step(fixture_config: AppConfig) -> None:
    prepared = prepare_dataset(fixture_config)
    report = prepared.train_report

    assert report.n_raw == 14  # 11 in train.csv + 3 in validation.csv
    assert report.n_unknown_label == 1
    assert report.n_too_short == 2
    assert report.n_duplicates == 1
    assert report.n_conflicting == 2
    assert report.n_test_overlap == 1
    assert report.n_final == 7
    assert report.label_counts == {"negative": 3, "positive": 4}

    assert prepared.test_report.n_final == 4
    assert prepared.test_report.label_counts == {"negative": 2, "positive": 2}


def test_prepared_train_and_test_do_not_overlap(fixture_config: AppConfig) -> None:
    prepared = prepare_dataset(fixture_config)
    shared = set(prepared.train["text"]) & set(prepared.test["text"])
    assert shared == set()


def test_prepared_texts_are_normalized(fixture_config: AppConfig) -> None:
    train = prepare_dataset(fixture_config).train
    assert "غذا عالی بود" in set(train["text"])  # Arabic ي was normalized, then deduplicated
    assert not train["text"].str.contains("ي").any()


def test_save_and_load_roundtrip(fixture_config: AppConfig) -> None:
    prepared = prepare_dataset(fixture_config)
    save_prepared(prepared, fixture_config.paths.processed_dir)

    train, test = load_prepared(fixture_config)

    pd.testing.assert_frame_equal(train, prepared.train, check_dtype=False)
    pd.testing.assert_frame_equal(test, prepared.test, check_dtype=False)
    assert train["label"].dtype.kind == "i"


def test_load_prepared_without_files_gives_clear_error(fixture_config: AppConfig) -> None:
    with pytest.raises(FileNotFoundError, match="reviewlens data prepare"):
        load_prepared(fixture_config)


def test_missing_raw_file_gives_clear_error(tmp_path: Path) -> None:
    cfg = AppConfig.model_validate({"paths": {"raw_dir": str(tmp_path)}})
    with pytest.raises(FileNotFoundError, match="reviewlens data download"):
        prepare_dataset(cfg)


def test_remove_test_overlap_only_touches_train() -> None:
    train = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 1, 0]})
    test = pd.DataFrame({"text": ["b"], "label": [1]})

    cleaned, n_removed = remove_test_overlap(train, test, "text")

    assert cleaned["text"].tolist() == ["a", "c"]
    assert n_removed == 1
    assert test["text"].tolist() == ["b"]
