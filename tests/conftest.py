"""Shared test fixtures."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reviewlens.config import AppConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


POSITIVE_WORDS = ["عالی", "خوشمزه", "ممنون", "سریع", "تازه", "داغ"]
NEGATIVE_WORDS = ["افتضاح", "سرد", "دیر", "بد", "کثیف", "بی‌کیفیت"]
FILLER_WORDS = ["غذا", "پیک", "بود", "رستوران", "سفارش", "امروز", "خیلی"]


def make_toy_corpus(n: int, seed: int) -> tuple[list[str], np.ndarray]:
    """Synthetic, easily learnable reviews: 2 sentiment words + 3 neutral words, shuffled.

    Labels alternate 0 (negative) / 1 (positive), so classes are exactly balanced.
    """
    rng = random.Random(seed)
    texts, labels = [], []
    for i in range(n):
        label = i % 2
        words = rng.sample(POSITIVE_WORDS if label else NEGATIVE_WORDS, 2)
        words += rng.sample(FILLER_WORDS, 3)
        rng.shuffle(words)
        texts.append(" ".join(words))
        labels.append(label)
    return texts, np.array(labels)


@dataclass
class ToyData:
    train_texts: list[str]
    y_train: np.ndarray
    test_texts: list[str]
    y_test: np.ndarray


@pytest.fixture
def toy_data() -> ToyData:
    train_texts, y_train = make_toy_corpus(120, seed=0)
    test_texts, y_test = make_toy_corpus(40, seed=1)
    return ToyData(train_texts, y_train, test_texts, y_test)


@pytest.fixture
def toy_config(tmp_path: Path, toy_data: ToyData) -> AppConfig:
    """Config whose processed data is the toy corpus; small settings so tests run fast."""
    processed = tmp_path / "processed"
    processed.mkdir()
    for name, texts, labels in [
        ("train.csv", toy_data.train_texts, toy_data.y_train),
        ("test.csv", toy_data.test_texts, toy_data.y_test),
    ]:
        pd.DataFrame({"text": texts, "label": labels}).to_csv(processed / name, index=False)
    return AppConfig.model_validate(
        {
            "paths": {
                "processed_dir": str(processed),
                "models_dir": str(tmp_path / "models"),
                "results_dir": str(tmp_path / "results"),
            },
            "features": {"min_df": 1, "lsa_components": 5},
            "models": {"random_forest": {"n_estimators": 20, "n_jobs": 1}},
        }
    )


@pytest.fixture
def fixture_config(tmp_path: Path) -> AppConfig:
    """Default config pointed at the tiny fixture dataset, writing into a temp directory."""
    return AppConfig.model_validate(
        {
            "paths": {
                "raw_dir": str(FIXTURES_DIR / "raw"),
                "processed_dir": str(tmp_path / "processed"),
            }
        }
    )


@pytest.fixture
def default_config_path() -> Path:
    return PROJECT_ROOT / "configs" / "default.yaml"


@pytest.fixture
def quick_experiment_path() -> Path:
    return PROJECT_ROOT / "configs" / "experiment" / "quick.yaml"


@pytest.fixture
def write_yaml(tmp_path: Path) -> Callable[..., Path]:
    """Return a helper that writes YAML text to a temp file and returns its path."""

    def _write(text: str, name: str = "config.yaml") -> Path:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path

    return _write
