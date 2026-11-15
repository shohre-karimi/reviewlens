"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
