from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from reviewlens import __version__
from reviewlens.cli import app
from reviewlens.config import AppConfig

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


def test_data_prepare_command_writes_processed_files(tmp_path: Path) -> None:
    fixtures = Path(__file__).resolve().parent / "fixtures" / "raw"
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {"paths": {"raw_dir": str(fixtures), "processed_dir": str(tmp_path / "processed")}}
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["--config", str(config), "data", "prepare"])

    assert result.exit_code == 0, result.output
    assert "n_final" in result.stdout
    assert (tmp_path / "processed" / "train.csv").is_file()


def test_data_prepare_without_raw_files_fails_cleanly(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"paths": {"raw_dir": str(tmp_path)}}), encoding="utf-8")

    result = runner.invoke(app, ["--config", str(config), "data", "prepare"])

    assert result.exit_code == 1
    assert "reviewlens data download" in result.output


def _write_config(cfg: AppConfig, path: Path) -> Path:
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    return path


def test_train_command_prints_metrics_and_saves_model(
    toy_config: AppConfig, tmp_path: Path
) -> None:
    config = _write_config(toy_config, tmp_path / "config.yaml")

    args = ["-c", str(config), "train", "-m", "logistic_regression", "-f", "0.5", "--save"]
    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert "macro_f1" in result.stdout
    assert (toy_config.paths.models_dir / "logistic_regression.joblib").is_file()


def test_train_command_rejects_unknown_method(toy_config: AppConfig, tmp_path: Path) -> None:
    config = _write_config(toy_config, tmp_path / "config.yaml")

    result = runner.invoke(app, ["-c", str(config), "train", "-m", "magic"])

    assert result.exit_code == 1
    assert "Unknown model 'magic'" in result.output


def test_baselines_command_compares_selected_methods(toy_config: AppConfig, tmp_path: Path) -> None:
    config = _write_config(toy_config, tmp_path / "config.yaml")

    args = ["-c", str(config), "baselines", "-m", "naive_bayes", "-m", "linear_svm"]
    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert "naive_bayes" in result.stdout and "linear_svm" in result.stdout
    assert "random_forest" not in result.stdout
    assert list(toy_config.paths.results_dir.glob("baselines_*.csv"))


def test_run_experiments_command_prints_summary(toy_config: AppConfig, tmp_path: Path) -> None:
    data = toy_config.model_dump(mode="json")
    data["experiment"] = {
        "name": "cli",
        "methods": ["naive_bayes", "self_training"],
        "label_fractions": [0.1, 1.0],
        "seeds": [0],
    }
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = runner.invoke(app, ["-c", str(config), "run-experiments"])

    assert result.exit_code == 0, result.output
    assert "self_training" in result.stdout and "±" in result.stdout
    assert (toy_config.paths.results_dir / "cli" / "runs.csv").is_file()

    result = runner.invoke(app, ["-c", str(config), "evaluate"])

    assert result.exit_code == 0, result.output
    assert "best_supervised" in result.stdout
    assert (toy_config.paths.results_dir / "cli" / "report.md").is_file()
    assert (toy_config.paths.figures_dir / "cli" / "macro_f1_vs_fraction.png").is_file()


def test_evaluate_without_results_fails_cleanly(toy_config: AppConfig, tmp_path: Path) -> None:
    config = _write_config(toy_config, tmp_path / "config.yaml")

    result = runner.invoke(app, ["-c", str(config), "evaluate"])

    assert result.exit_code == 1
    assert "reviewlens run-experiments" in result.output


def test_unimplemented_command_reports_its_phase(default_config_path: Path) -> None:
    result = runner.invoke(app, ["-c", str(default_config_path), "agent", "eval"])

    assert result.exit_code == 1
    assert "phase 6" in result.output
