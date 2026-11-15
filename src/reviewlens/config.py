"""Typed project configuration loaded from YAML files.

YAML holds the values; the pydantic models below define which keys exist, their types,
their defaults and their allowed ranges. An invalid config fails at load time with a clear
message instead of causing a confusing error deep inside an experiment.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


class _StrictModel(BaseModel):
    """Base for all config sections: unknown keys are errors and values are read-only."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_unique(values: Sequence[Any], field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates: {list(values)}")


class PathsConfig(_StrictModel):
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("artifacts/models")
    results_dir: Path = Path("artifacts/results")
    figures_dir: Path = Path("artifacts/figures")


class TaskConfig(_StrictModel):
    """Label schema. Label ``labels[i]`` is encoded as the integer ``i``."""

    name: str = Field(default="sentiment", min_length=1)
    labels: list[str] = Field(default_factory=lambda: ["negative", "positive"], min_length=2)
    text_column: str = Field(default="text", min_length=1)
    label_column: str = Field(default="label", min_length=1)

    @field_validator("labels")
    @classmethod
    def _labels_unique(cls, labels: list[str]) -> list[str]:
        _check_unique(labels, "labels")
        return labels


class ExperimentConfig(_StrictModel):
    name: str = Field(default="default", min_length=1)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    label_fractions: list[float] = Field(
        default_factory=lambda: [0.01, 0.05, 0.10, 0.20, 1.0], min_length=1
    )
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4], min_length=1)
    # Method names are checked against the model registry (phase 3), not here.
    methods: list[str] = Field(default_factory=lambda: ["logistic_regression"], min_length=1)

    @field_validator("label_fractions")
    @classmethod
    def _fractions_valid(cls, fractions: list[float]) -> list[float]:
        for fraction in fractions:
            if not 0.0 < fraction <= 1.0:
                raise ValueError(f"label fractions must be in (0, 1], got {fraction}")
        _check_unique(fractions, "label_fractions")
        return sorted(fractions)

    @field_validator("seeds", "methods")
    @classmethod
    def _no_duplicates(cls, values: list[Any]) -> list[Any]:
        _check_unique(values, "seeds/methods")
        return values


class AgentConfig(_StrictModel):
    """Agent settings. Model names and API keys come from environment variables, not here."""

    provider: Literal["anthropic", "openai_compatible"] = "anthropic"
    max_steps: int = Field(default=5, ge=1, le=20)
    max_tokens: int = Field(default=1024, ge=1)
    openai_base_url: str | None = None


class LoggingConfig(_StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class AppConfig(_StrictModel):
    """Root of the configuration tree."""

    seed: int = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file whose top level is a mapping. An empty file gives an empty dict."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Top level of {path} must be a mapping, got {type(data).__name__}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict: ``override`` on top of ``base``, merging nested dicts key by key.

    Lists and scalars in ``override`` replace the base value entirely.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path = DEFAULT_CONFIG_PATH, overrides: Sequence[Path] = ()) -> AppConfig:
    """Load the main config, apply override files in order, and validate the result."""
    data = read_yaml(path)
    for override_path in overrides:
        data = deep_merge(data, read_yaml(override_path))
    return AppConfig.model_validate(data)
