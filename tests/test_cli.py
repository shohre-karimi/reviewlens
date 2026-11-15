from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from reviewlens import __version__
from reviewlens.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_show_config_prints_valid_yaml(default_config_path: Path) -> None:
    result = runner.invoke(app, ["--config", str(default_config_path), "show-config"])

    assert result.exit_code == 0, result.output
    data = yaml.safe_load(result.stdout)
    assert data["task"]["labels"] == ["negative", "positive"]


def test_experiment_option_overrides(
    default_config_path: Path, quick_experiment_path: Path
) -> None:
    args = ["-c", str(default_config_path), "-e", str(quick_experiment_path), "show-config"]
    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert yaml.safe_load(result.stdout)["experiment"]["name"] == "quick"


def test_missing_config_exits_with_clear_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["--config", str(tmp_path / "nope.yaml"), "show-config"])

    assert result.exit_code == 2
    assert "Config file not found" in result.output


def test_invalid_config_exits_with_clear_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("agent:\n  max_steps: 0\n", encoding="utf-8")

    result = runner.invoke(app, ["--config", str(bad), "show-config"])

    assert result.exit_code == 2
    assert "Invalid configuration" in result.output


def test_unimplemented_command_reports_its_phase(default_config_path: Path) -> None:
    result = runner.invoke(app, ["-c", str(default_config_path), "run-experiments"])

    assert result.exit_code == 1
    assert "phase 4" in result.output
