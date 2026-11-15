"""Turn the raw files into clean, validated train and test sets.

Pipeline for each split:
    read raw CSVs -> encode labels -> normalize text -> drop short texts -> drop duplicates

Then any training text that also appears in the test set is removed from *train*
(the test set is never modified), and both splits are checked and saved to
``paths.processed_dir`` as ``train.csv`` and ``test.csv``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from reviewlens.config import AppConfig
from reviewlens.data.load import RAW_LABEL, read_raw_files
from reviewlens.data.validate import (
    check_prepared,
    drop_duplicates,
    drop_short_texts,
    encode_labels,
)
from reviewlens.preprocessing import normalize_texts

logger = logging.getLogger(__name__)

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"


@dataclass(frozen=True)
class SplitReport:
    """How many rows each cleaning step removed from one split."""

    n_raw: int
    n_unknown_label: int
    n_too_short: int
    n_duplicates: int
    n_conflicting: int
    n_test_overlap: int
    n_final: int
    label_counts: dict[str, int]


@dataclass(frozen=True)
class PreparedData:
    train: pd.DataFrame
    test: pd.DataFrame
    train_report: SplitReport
    test_report: SplitReport


def _clean_split(raw: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    """Run the cleaning steps on one split and count what each step removed."""
    task, data = cfg.task, cfg.data
    frame, n_unknown = encode_labels(raw, RAW_LABEL, task.label_column, data.label_map, task.labels)
    frame[task.text_column] = normalize_texts(frame[task.text_column])
    frame, n_short = drop_short_texts(frame, task.text_column, data.min_text_chars)
    frame, n_dup, n_conflict = drop_duplicates(frame, task.text_column, task.label_column)
    counts = {
        "n_raw": len(raw),
        "n_unknown_label": n_unknown,
        "n_too_short": n_short,
        "n_duplicates": n_dup,
        "n_conflicting": n_conflict,
    }
    return frame, counts


def remove_test_overlap(
    train: pd.DataFrame, test: pd.DataFrame, text_column: str
) -> tuple[pd.DataFrame, int]:
    """Drop training rows whose (normalized) text also appears in the test set."""
    in_test = train[text_column].isin(set(test[text_column]))
    return train.loc[~in_test].reset_index(drop=True), int(in_test.sum())


def _report(
    frame: pd.DataFrame, counts: dict[str, int], n_overlap: int, cfg: AppConfig
) -> SplitReport:
    value_counts = frame[cfg.task.label_column].value_counts()
    label_counts = {name: int(value_counts.get(i, 0)) for i, name in enumerate(cfg.task.labels)}
    return SplitReport(
        **counts, n_test_overlap=n_overlap, n_final=len(frame), label_counts=label_counts
    )


def prepare_dataset(cfg: AppConfig) -> PreparedData:
    """Build the clean train and test sets from the raw files in ``paths.raw_dir``."""
    raw_dir, task = cfg.paths.raw_dir, cfg.task
    raw_train = read_raw_files([raw_dir / name for name in cfg.data.train_files], cfg.data, task)
    raw_test = read_raw_files([raw_dir / cfg.data.test_file], cfg.data, task)

    train, train_counts = _clean_split(raw_train, cfg)
    test, test_counts = _clean_split(raw_test, cfg)
    train, n_overlap = remove_test_overlap(train, test, task.text_column)

    for frame in (train, test):
        check_prepared(frame, task.text_column, task.label_column, len(task.labels))

    prepared = PreparedData(
        train=train,
        test=test,
        train_report=_report(train, train_counts, n_overlap, cfg),
        test_report=_report(test, test_counts, 0, cfg),
    )
    logger.info("Prepared train %s", asdict(prepared.train_report))
    logger.info("Prepared test %s", asdict(prepared.test_report))
    return prepared


def save_prepared(prepared: PreparedData, processed_dir: Path) -> tuple[Path, Path]:
    """Write the prepared splits as UTF-8 CSV files and return their paths."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path, test_path = processed_dir / TRAIN_FILE, processed_dir / TEST_FILE
    prepared.train.to_csv(train_path, index=False, encoding="utf-8")
    prepared.test.to_csv(test_path, index=False, encoding="utf-8")
    logger.info("Saved processed data train=%s test=%s", train_path, test_path)
    return train_path, test_path


def load_prepared(cfg: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the processed train and test sets written by ``save_prepared``."""
    processed_dir = cfg.paths.processed_dir
    paths = [processed_dir / TRAIN_FILE, processed_dir / TEST_FILE]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Processed data not found: {path}. Run `reviewlens data prepare`."
            )
    dtypes = {cfg.task.text_column: str, cfg.task.label_column: int}
    train, test = (pd.read_csv(path, encoding="utf-8", dtype=dtypes) for path in paths)
    return train, test
