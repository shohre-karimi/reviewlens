from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from reviewlens.config import AppConfig, ExperimentConfig
from reviewlens.data.prepare import load_prepared
from reviewlens.experiments import runner
from reviewlens.experiments.runner import (
    CONFIG_FILE,
    PSEUDO_FILE,
    RUNS_FILE,
    SUMMARY_FILE,
    format_summary_table,
    run_experiments,
)
from reviewlens.experiments.train import train_and_evaluate


def _with_grid(
    cfg: AppConfig, methods: list[str], fractions: list[float], seeds: list[int]
) -> AppConfig:
    experiment = ExperimentConfig(
        name="test", methods=methods, label_fractions=fractions, seeds=seeds
    )
    return cfg.model_copy(update={"experiment": experiment})


def _count_training_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    calls: list[tuple[Any, ...]] = []
    original = train_and_evaluate

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[:1] + args[4:6])  # (method, fraction, seed)
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "train_and_evaluate", counting)
    return calls


def test_runs_full_grid_and_saves_every_output(toy_config: AppConfig) -> None:
    cfg = _with_grid(toy_config, ["naive_bayes", "pseudo_labeling"], [0.1, 1.0], [0, 1])
    train, test = load_prepared(cfg)

    output = run_experiments(cfg, train, test)

    assert len(output.runs) == 8  # 2 methods x 2 fractions x 2 seeds
    assert len(output.summary) == 4
    assert set(output.pseudo_labels["method"]) == {"pseudo_labeling"}  # supervised: no history
    for name in (RUNS_FILE, PSEUDO_FILE, SUMMARY_FILE, CONFIG_FILE):
        assert (output.output_dir / name).is_file()
    assert len(pd.read_csv(output.output_dir / RUNS_FILE)) == 8


def test_resume_only_trains_missing_runs(
    toy_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, test = load_prepared(toy_config)
    run_experiments(_with_grid(toy_config, ["naive_bayes"], [0.5], [0]), train, test)

    calls = _count_training_calls(monkeypatch)
    output = run_experiments(_with_grid(toy_config, ["naive_bayes"], [0.5], [0, 1]), train, test)

    assert calls == [("naive_bayes", 0.5, 1)]  # seed 0 was already saved
    assert len(output.runs) == 2


def test_fresh_run_ignores_saved_results(
    toy_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _with_grid(toy_config, ["naive_bayes"], [0.5], [0])
    train, test = load_prepared(cfg)
    run_experiments(cfg, train, test)

    calls = _count_training_calls(monkeypatch)
    run_experiments(cfg, train, test, resume=False)

    assert len(calls) == 1


def test_results_of_methods_removed_from_config_are_not_summarized(toy_config: AppConfig) -> None:
    train, test = load_prepared(toy_config)
    run_experiments(_with_grid(toy_config, ["naive_bayes", "linear_svm"], [0.5], [0]), train, test)

    output = run_experiments(_with_grid(toy_config, ["naive_bayes"], [0.5], [0]), train, test)

    assert set(output.summary["method"]) == {"naive_bayes"}


def test_unknown_method_fails_before_any_training(
    toy_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _count_training_calls(monkeypatch)
    train, test = load_prepared(toy_config)

    with pytest.raises(ValueError, match="Unknown model 'magic'"):
        run_experiments(_with_grid(toy_config, ["naive_bayes", "magic"], [0.5], [0]), train, test)
    assert calls == []


def test_summary_table_has_mean_and_std_per_fraction() -> None:
    summary = pd.DataFrame(
        {
            "method": ["a", "a"],
            "label_fraction": [0.01, 1.0],
            "macro_f1_mean": [0.81234, 0.9],
            "macro_f1_std": [0.004, float("nan")],
        }
    )

    table = format_summary_table(summary)

    assert list(table.columns) == ["0.01", "1"]
    assert table.loc["a", "0.01"] == "0.812 ± 0.004"
    assert table.loc["a", "1"] == "0.900 ± 0.000"
