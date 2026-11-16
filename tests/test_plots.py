"""Figures are checked for structure (lines, panels, files), not for pixels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reviewlens.config import AppConfig, ExperimentConfig
from reviewlens.data.prepare import load_prepared
from reviewlens.evaluation import plots, report
from reviewlens.evaluation.evaluate import evaluate_experiment
from reviewlens.experiments.runner import run_experiments
from tests.test_report import is_semi, make_pseudo, make_runs


def test_learning_curves_one_line_per_method_semi_dashed() -> None:
    fig = plots.learning_curves(make_runs(), is_semi)
    lines = {line.get_label(): line for line in fig.axes[0].get_lines()}

    assert set(lines) == {"logistic_regression", "naive_bayes", "pseudo_labeling"}
    assert lines["pseudo_labeling"].get_linestyle() == "--"
    assert lines["naive_bayes"].get_linestyle() == "-"
    assert fig.axes[0].get_xscale() == "log"


def test_learning_curves_zoom_panel_shows_top_of_range() -> None:
    fig = plots.learning_curves(make_runs(), is_semi)
    full, zoom = fig.axes

    assert len(zoom.get_lines()) == len(full.get_lines())
    low, high = zoom.get_ylim()
    assert low == pytest.approx(0.90 - 0.1)  # best mean in make_runs() is 0.90
    assert high == pytest.approx(0.91)


def test_learning_curves_use_display_names() -> None:
    fig = plots.learning_curves(make_runs(), is_semi, {"naive_bayes": "Naive Bayes"})
    labels = {line.get_label() for line in fig.axes[0].get_lines()}
    assert "Naive Bayes" in labels


def test_pseudo_label_curves_two_panels_per_method() -> None:
    fig = plots.pseudo_label_curves(report.pseudo_label_curve(make_pseudo()))
    assert len(fig.axes) == 2


def test_pseudo_label_curves_skip_single_iteration_methods() -> None:
    one_step = make_pseudo().query("iteration == 1").assign(method="label_spreading")
    curve = report.pseudo_label_curve(pd.concat([make_pseudo(), one_step]))

    fig = plots.pseudo_label_curves(curve)

    assert len(fig.axes) == 2  # only pseudo_labeling
    assert "pseudo_labeling" in fig.axes[0].get_title()


def test_confusion_grid_hides_unused_cells() -> None:
    matrix = np.array([[0.8, 0.2], [0.1, 0.9]])
    fig = plots.confusion_grid({f"m{i}": matrix for i in range(5)}, ["neg", "pos"], "t")

    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(fig.axes) == 8  # 2 rows x 4 columns
    assert len(visible) == 5


def test_save_figure_writes_png(tmp_path: Path) -> None:
    path = plots.save_figure(plots.learning_curves(make_runs(), is_semi), tmp_path / "a" / "f.png")
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_evaluate_experiment_end_to_end(toy_config: AppConfig) -> None:
    experiment = ExperimentConfig(
        name="eval",
        methods=["naive_bayes", "pseudo_labeling"],
        label_fractions=[0.1, 1.0],
        seeds=[0, 1],
    )
    cfg = toy_config.model_copy(update={"experiment": experiment})
    train, test = load_prepared(cfg)
    run_experiments(cfg, train, test)

    output = evaluate_experiment(cfg)

    names = sorted(path.name for path in output.figures)
    assert names == [
        "confusion_0.1.png",
        "confusion_1.png",
        "macro_f1_vs_fraction.png",
        "pseudo_labels.png",
    ]
    assert all(path.is_file() for path in output.figures)
    assert set(output.comparison["label_fraction"]) == {0.1, 1.0}
    results = cfg.paths.results_dir / "eval"
    assert "paired by seed" in output.report_path.read_text(encoding="utf-8")
    assert not pd.read_csv(results / "pseudo_summary.csv").empty
    assert (results / "comparison.csv").is_file()
