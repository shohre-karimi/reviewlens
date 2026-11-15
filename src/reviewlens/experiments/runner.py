"""Experiment runner: every method x every label fraction x every seed.

Outputs (in ``paths.results_dir / experiment.name``):
- ``runs.csv``           one row per run: method, fraction, seed, metrics, fit time
- ``pseudo_labels.csv``  one row per semi-supervised iteration: pseudo-labels added + accuracy
- ``summary.csv``        mean and std over seeds for each method x fraction
- ``config.yaml``        the exact configuration used, for reproducibility

Results are written after *every* run. If a long experiment stops (crash, laptop closed),
running it again with ``resume=True`` skips the runs already in ``runs.csv``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from reviewlens.config import AppConfig
from reviewlens.experiments.train import build_experiment_features, train_and_evaluate
from reviewlens.models.registry import get_spec
from reviewlens.models.semi_supervised import ReportsPseudoLabels

logger = logging.getLogger(__name__)

RUNS_FILE = "runs.csv"
PSEUDO_FILE = "pseudo_labels.csv"
SUMMARY_FILE = "summary.csv"
CONFIG_FILE = "config.yaml"
RUN_KEY = ("method", "label_fraction", "seed")
PSEUDO_COLUMNS = [*RUN_KEY, "iteration", "n_added", "n_correct", "accuracy"]

RunKey = tuple[str, float, int]


@dataclass(frozen=True)
class ExperimentOutput:
    runs: pd.DataFrame
    pseudo_labels: pd.DataFrame
    summary: pd.DataFrame
    output_dir: Path


def experiment_dir(cfg: AppConfig) -> Path:
    return cfg.paths.results_dir / cfg.experiment.name


def planned_runs(cfg: AppConfig) -> list[RunKey]:
    """The full grid, in the order it is run (method by method)."""
    exp = cfg.experiment
    return [(m, f, s) for m in exp.methods for f in exp.label_fractions for s in exp.seeds]


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds for each method x label fraction."""
    grouped = runs.groupby(["method", "label_fraction"], sort=True)
    return grouped.agg(
        n_labeled=("n_labeled", "first"),
        n_seeds=("seed", "nunique"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        fit_seconds_mean=("fit_seconds", "mean"),
    ).reset_index()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write via a temporary file so an interrupted write never corrupts the results."""
    partial = path.with_name(path.name + ".part")
    frame.to_csv(partial, index=False)
    partial.replace(path)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:  # a file written by an older version with no columns
        return []
    return [{str(key): value for key, value in row.items()} for row in frame.to_dict("records")]


def _key(row: dict[str, Any]) -> RunKey:
    return (str(row["method"]), float(row["label_fraction"]), int(row["seed"]))


def run_experiments(
    cfg: AppConfig, train_df: pd.DataFrame, test_df: pd.DataFrame, resume: bool = True
) -> ExperimentOutput:
    """Run the configured grid, saving results after every run."""
    for method in cfg.experiment.methods:
        get_spec(method)  # fail before any work if a method name is wrong

    out_dir = experiment_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / CONFIG_FILE).write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    run_rows = _read_rows(out_dir / RUNS_FILE) if resume else []
    pseudo_rows = _read_rows(out_dir / PSEUDO_FILE) if resume else []
    done = {_key(row) for row in run_rows}
    grid = planned_runs(cfg)
    todo = [key for key in grid if key not in done]
    logger.info(
        "Experiment=%s planned=%d done=%d todo=%d", out_dir.name, len(grid), len(done), len(todo)
    )

    text, label = cfg.task.text_column, cfg.task.label_column
    y_train, y_test = train_df[label].to_numpy(), test_df[label].to_numpy()
    if todo:
        features = build_experiment_features(
            train_df[text], test_df[text], cfg.experiment.methods, cfg
        )

    for i, (method, fraction, seed) in enumerate(todo, start=1):
        model, result = train_and_evaluate(method, features, y_train, y_test, fraction, seed, cfg)
        run_rows.append(result.as_row())
        if isinstance(model, ReportsPseudoLabels):
            for step in model.pseudo_label_history(y_train):
                pseudo_rows.append(
                    {"method": method, "label_fraction": fraction, "seed": seed, **step.as_dict()}
                )
        _write_csv(pd.DataFrame(run_rows), out_dir / RUNS_FILE)
        # Explicit columns: with only supervised methods there are no rows, and a CSV without
        # a header cannot be read back when resuming.
        _write_csv(pd.DataFrame(pseudo_rows, columns=PSEUDO_COLUMNS), out_dir / PSEUDO_FILE)
        logger.info("Progress %d/%d", i, len(todo))

    # Keep only the runs of the current grid (an older config may have had other methods).
    grid_keys = set(grid)
    runs = pd.DataFrame([row for row in run_rows if _key(row) in grid_keys])
    pseudo = pd.DataFrame(
        [row for row in pseudo_rows if _key(row) in grid_keys], columns=PSEUDO_COLUMNS
    )
    summary = summarize(runs)
    _write_csv(summary, out_dir / SUMMARY_FILE)
    return ExperimentOutput(runs=runs, pseudo_labels=pseudo, summary=summary, output_dir=out_dir)


def format_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Methods as rows, label fractions as columns, cells "mean ± std" of macro-F1."""
    cells = summary.assign(
        cell=summary["macro_f1_mean"].map("{:.3f}".format)
        + " ± "
        + summary["macro_f1_std"].fillna(0.0).map("{:.3f}".format)
    )
    table = cells.pivot(index="method", columns="label_fraction", values="cell")
    table.columns = [f"{fraction:g}" for fraction in table.columns]
    return table
