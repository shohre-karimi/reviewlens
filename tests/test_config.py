from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from reviewlens.config import AppConfig, deep_merge, load_config

WriteYaml = Callable[..., Path]


def test_default_config_loads(default_config_path: Path) -> None:
    cfg = load_config(default_config_path)

    assert isinstance(cfg, AppConfig)
    assert cfg.task.labels == ["negative", "positive"]
    assert cfg.experiment.label_fractions == [0.01, 0.05, 0.10, 0.20, 1.0]
    assert len(cfg.experiment.seeds) == 5
    assert cfg.agent.provider == "anthropic"
    assert cfg.agent.max_steps == 5


def test_experiment_override_merges_into_default(
    default_config_path: Path, quick_experiment_path: Path
) -> None:
    cfg = load_config(default_config_path, [quick_experiment_path])

    assert cfg.experiment.name == "quick"
    assert cfg.experiment.seeds == [0, 1]
    # Keys not mentioned in the override keep their default values.
    assert cfg.experiment.methods == ["logistic_regression", "self_training"]
    assert cfg.data.test_file == "test.csv"
    assert cfg.task.name == "sentiment"


def test_data_file_url_is_pinned_to_revision(default_config_path: Path) -> None:
    data = load_config(default_config_path).data

    url = data.file_url("train.csv")

    assert url.startswith("https://huggingface.co/datasets/")
    assert f"/resolve/{data.revision}/train.csv" in url
    assert data.all_files == ["train.csv", "validation.csv", "test.csv"]


def test_deep_merge_merges_nested_dicts_and_replaces_lists() -> None:
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 1}
    override = {"a": {"y": [3]}, "c": 2}

    merged = deep_merge(base, override)

    assert merged == {"a": {"x": 1, "y": [3]}, "b": 1, "c": 2}
    assert base == {"a": {"x": 1, "y": [1, 2]}, "b": 1}  # input is not mutated


def test_empty_file_gives_all_defaults(write_yaml: WriteYaml) -> None:
    cfg = load_config(write_yaml(""))
    assert cfg == AppConfig()


def test_label_fractions_are_sorted(write_yaml: WriteYaml) -> None:
    cfg = load_config(write_yaml("experiment:\n  label_fractions: [1.0, 0.1, 0.05]\n"))
    assert cfg.experiment.label_fractions == [0.05, 0.1, 1.0]


@pytest.mark.parametrize(
    "yaml_text",
    [
        "experiment:\n  label_fractions: [0.0]\n",  # zero labels makes no sense
        "experiment:\n  label_fractions: [1.5]\n",  # more than 100 %
        "experiment:\n  label_fractions: [0.1, 0.1]\n",  # duplicates
        "data:\n  min_text_chars: 0\n",
        "data:\n  label_map: {GOOD: excellent}\n",  # target label not in task.labels
        "experiment:\n  seeds: []\n",
        "task:\n  labels: [positive, positive]\n",
        "task:\n  labels: [positive]\n",  # need at least two classes
        "agent:\n  provider: gemini\n",
        "agent:\n  max_steps: 0\n",
        "logging:\n  level: LOUD\n",
        "experimnt:\n  name: typo\n",  # unknown top-level key
        "agent:\n  max_step: 3\n",  # unknown nested key
    ],
)
def test_invalid_config_is_rejected(write_yaml: WriteYaml, yaml_text: str) -> None:
    with pytest.raises(ValidationError):
        load_config(write_yaml(yaml_text))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_non_mapping_yaml_raises(write_yaml: WriteYaml) -> None:
    with pytest.raises(ValueError, match="mapping"):
        load_config(write_yaml("- just\n- a list\n"))


def test_config_is_read_only(default_config_path: Path) -> None:
    cfg = load_config(default_config_path)
    with pytest.raises(ValidationError):
        cfg.seed = 7  # type: ignore[misc]
