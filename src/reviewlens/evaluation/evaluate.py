"""Evaluate a finished experiment: saved CSVs in, report tables and figures out.

    runs.csv + pseudo_labels.csv
        -> results_dir/<experiment>/  report.md, comparison.csv, pseudo_summary.csv
        -> figures_dir/<experiment>/  macro_f1_vs_fraction.png, pseudo_labels.png,
                                      confusion_<fraction>.png (smallest and largest fraction)

No model is trained here, so this runs in seconds and can be repeated after any change
to the tables or plots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from reviewlens.config import AppConfig
from reviewlens.evaluation import plots, report
from reviewlens.experiments.runner import experiment_dir
from reviewlens.models.registry import get_spec

logger = logging.getLogger(__name__)

REPORT_FILE = "report.md"
COMPARISON_FILE = "comparison.csv"
PSEUDO_SUMMARY_FILE = "pseudo_summary.csv"
MAIN_FIGURE = "macro_f1_vs_fraction.png"
PSEUDO_FIGURE = "pseudo_labels.png"


@dataclass(frozen=True)
class EvaluationOutput:
    comparison: pd.DataFrame
    report_path: Path
    figures: list[Path]


def is_semi_supervised(method: str) -> bool:
    return get_spec(method).semi_supervised


def evaluate_experiment(cfg: AppConfig) -> EvaluationOutput:
    """Build all tables and figures for the experiment named in ``cfg``."""
    results_dir = experiment_dir(cfg)
    figures_dir = cfg.paths.figures_dir / cfg.experiment.name
    runs, pseudo = report.load_results(results_dir)
    methods = sorted(runs["method"].unique())
    names = {m: get_spec(m).description for m in methods}

    figures = [
        plots.save_figure(
            plots.learning_curves(runs, is_semi_supervised, names), figures_dir / MAIN_FIGURE
        )
    ]
    if not pseudo.empty:
        curve = report.pseudo_label_curve(pseudo)
        figures.append(
            plots.save_figure(plots.pseudo_label_curves(curve), figures_dir / PSEUDO_FIGURE)
        )
    fractions = sorted(runs["label_fraction"].unique())
    for fraction in sorted({fractions[0], fractions[-1]}):
        matrices = {
            m: report.mean_confusion(runs, m, fraction, len(cfg.task.labels)) for m in methods
        }
        title = (
            f"Confusion matrices at {fraction * 100:g}% labels (row-normalized, mean over seeds)"
        )
        fig = plots.confusion_grid(matrices, cfg.task.labels, title)
        figures.append(plots.save_figure(fig, figures_dir / f"confusion_{fraction:g}.png"))

    comparison = report.compare_to_supervised(runs, is_semi_supervised)
    comparison.to_csv(results_dir / COMPARISON_FILE, index=False)
    report.pseudo_label_summary(pseudo).to_csv(results_dir / PSEUDO_SUMMARY_FILE, index=False)
    # as_posix: "default/x.png" on every OS (Windows would write "default\x.png")
    figure_names = [path.relative_to(cfg.paths.figures_dir).as_posix() for path in figures]
    text = report.build_report(runs, pseudo, is_semi_supervised, figure_names)
    report_path = results_dir / REPORT_FILE
    report_path.write_text(text, encoding="utf-8")

    logger.info("Evaluation written report=%s figures=%d", report_path, len(figures))
    return EvaluationOutput(comparison=comparison, report_path=report_path, figures=figures)
