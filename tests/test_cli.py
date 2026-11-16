from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from reviewlens import __version__, cli
from reviewlens.cli import app
from reviewlens.config import AppConfig
from tests.test_agent_loop import FakeClient, answer, call

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


def test_agent_local_test_passes_without_any_model(agent_config: AppConfig, tmp_path: Path) -> None:
    config = _write_config(agent_config, tmp_path / "config.yaml")

    result = runner.invoke(app, ["-c", str(config), "agent", "local-test"])

    assert result.exit_code == 0, result.output
    assert "12/12 tool checks passed" in result.stdout


def test_agent_local_test_fails_without_artifacts(toy_config: AppConfig, tmp_path: Path) -> None:
    config = _write_config(toy_config, tmp_path / "config.yaml")  # no model, no results

    result = runner.invoke(app, ["-c", str(config), "agent", "local-test"])

    assert result.exit_code == 1
    assert "[FAIL] analyze_review" in result.stdout


def test_agent_chat_prints_trace_and_answer(
    agent_config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeClient([call(("analyze_review", {"text": "غذا عالی بود"})), answer("مثبت است.")])
    monkeypatch.setattr(cli, "create_client", lambda cfg, key=None: fake)
    config = _write_config(agent_config, tmp_path / "config.yaml")

    result = runner.invoke(app, ["-c", str(config), "agent", "chat", "تحلیل کن"])

    assert result.exit_code == 0, result.output
    assert "[step 1] analyze_review" in result.stdout
    assert "مثبت است." in result.stdout


def test_agent_chat_without_model_setup_explains_in_persian(
    agent_config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    data = agent_config.model_dump(mode="json")
    data["agent"]["provider"] = "openai_compatible"
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = runner.invoke(app, ["-c", str(config), "agent", "chat", "سلام"])

    assert result.exit_code == 1
    assert "تنظیمات مدل زبانی کامل نیست" in result.output
    assert "Traceback" not in result.output


def test_agent_eval_writes_the_table(
    agent_config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "create_client", lambda cfg, key=None: FakeClient([answer("x")] * 99))
    config = _write_config(agent_config, tmp_path / "config.yaml")
    output = tmp_path / "agent_tests.md"

    args = ["-c", str(config), "agent", "eval", "--runs", "1", "-o", str(output)]
    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert "Tool-selection accuracy" in result.stdout
    assert "| 1 | `analyze_positive`" in output.read_text(encoding="utf-8")
