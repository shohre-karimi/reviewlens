"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from reviewlens.config import AppConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


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
