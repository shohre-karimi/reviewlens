"""Report tables, built from small hand-made result tables with known answers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reviewlens.evaluation import report
from reviewlens.experiments.runner import summarize

SEMI = {"pseudo_labeling"}


def is_semi(method: str) -> bool:
    return method in SEMI


def make_runs() -> pd.DataFrame:
    """2 supervised + 1 semi-supervised method, 2 fractions, 3 seeds, with chosen scores."""
    scores = {
        # fraction 0.1: logistic_regression is the best supervised (0.80 > 0.70);
        # pseudo_labeling beats it on seeds 0 and 1, loses on seed 2.
        ("naive_bayes", 0.1): [0.70, 0.70, 0.70],
        ("logistic_regression", 0.1): [0.80, 0.80, 0.80],
        ("pseudo_labeling", 0.1): [0.85, 0.82, 0.77],
        # fraction 1.0: naive_bayes is the best supervised; pseudo_labeling is worse everywhere.
        ("naive_bayes", 1.0): [0.90, 0.90, 0.90],
        ("logistic_regression", 1.0): [0.88, 0.88, 0.88],
        ("pseudo_labeling", 1.0): [0.89, 0.89, 0.89],
    }
    rows = []
    for (method, fraction), values in scores.items():
        for seed, f1 in enumerate(values):
            rows.append(
                {
                    "method": method,
                    "label_fraction": fraction,
                    "seed": seed,
                    "n_labeled": int(fraction * 100),
                    "fit_seconds": 0.1,
                    "macro_f1": f1,
                    "accuracy": f1,
                    # 10 negatives (8 right), 10 positives (all right)
                    "cm_0_0": 8,
                    "cm_0_1": 2,
                    "cm_1_0": 0,
                    "cm_1_1": 10,
                }
            )
    return pd.DataFrame(rows)


def make_pseudo() -> pd.DataFrame:
    """One method, one fraction, two seeds, two iterations each."""
    return pd.DataFrame(
        {
            "method": ["pseudo_labeling"] * 4,
            "label_fraction": [0.1] * 4,
            "seed": [0, 0, 1, 1],
            "iteration": [1, 2, 1, 2],
            "n_added": [10, 30, 20, 0],
            "n_correct": [9, 21, 18, 0],
            "accuracy": [0.9, 0.7, 0.9, np.nan],
        }
    )


def test_compare_picks_best_supervised_per_fraction() -> None:
    comparison = report.compare_to_supervised(make_runs(), is_semi)
    best = dict(zip(comparison["label_fraction"], comparison["best_supervised"], strict=True))

    assert best == {0.1: "logistic_regression", 1.0: "naive_bayes"}


def test_compare_is_paired_by_seed() -> None:
    row = report.compare_to_supervised(make_runs(), is_semi).set_index("label_fraction").loc[0.1]

    # per-seed differences: +0.05, +0.02, -0.03
    assert row["diff_mean"] == pytest.approx(np.mean([0.05, 0.02, -0.03]))
    assert row["diff_std"] == pytest.approx(np.std([0.05, 0.02, -0.03], ddof=1))
    assert row["wins"] == 2
    assert row["n_seeds"] == 3


def test_compare_needs_both_kinds_of_method() -> None:
    supervised_only = make_runs().query("method != 'pseudo_labeling'")
    with pytest.raises(ValueError, match="semi-supervised"):
        report.compare_to_supervised(supervised_only, is_semi)


def test_pseudo_label_summary_pools_accuracy() -> None:
    summary = report.pseudo_label_summary(make_pseudo()).iloc[0]

    # seed 0 adds 40 (30 correct), seed 1 adds 20 (18 correct): means 30 and 24
    assert summary["added"] == pytest.approx(30)
    assert summary["accuracy"] == pytest.approx(24 / 30)
    assert summary["iterations"] == pytest.approx(2)


def test_pseudo_label_summary_of_empty_table() -> None:
    assert report.pseudo_label_summary(pd.DataFrame()).empty


def test_pseudo_label_curve_per_iteration() -> None:
    curve = report.pseudo_label_curve(make_pseudo()).set_index("iteration")

    assert curve.loc[1, "added"] == pytest.approx(15)  # (10 + 20) / 2
    assert curve.loc[1, "accuracy"] == pytest.approx(27 / 30)
    assert curve.loc[2, "accuracy"] == pytest.approx(21 / 30)


def test_mean_confusion_is_row_normalized() -> None:
    matrix = report.mean_confusion(make_runs(), "naive_bayes", 0.1, n_labels=2)

    np.testing.assert_allclose(matrix, [[0.8, 0.2], [0.0, 1.0]])


def test_mean_confusion_without_columns_asks_for_rerun() -> None:
    runs = make_runs().drop(columns=["cm_0_0"])
    with pytest.raises(ValueError, match="--fresh"):
        report.mean_confusion(runs, "naive_bayes", 0.1, n_labels=2)


def test_macro_f1_table_bolds_the_best() -> None:
    methods = ["naive_bayes", "logistic_regression", "pseudo_labeling"]
    table = report.macro_f1_table(summarize(make_runs()), methods).set_index("method")

    # at 10 %, pseudo_labeling has the best mean: (0.85 + 0.82 + 0.77) / 3 = 0.813
    assert str(table.loc["pseudo_labeling", "10%"]).startswith("**0.813 ± ")
    assert table.loc["logistic_regression", "10%"] == "0.800 ± 0.000"
    assert table.loc["naive_bayes", "100%"] == "**0.900 ± 0.000**"


def test_build_report_has_all_sections() -> None:
    text = report.build_report(make_runs(), make_pseudo(), is_semi, ["main/figure.png"])

    for heading in ("Macro-F1", "paired by seed", "Pseudo-labels", "Figures"):
        assert heading in text
    assert "main/figure.png" in text
    assert "2/3" in text  # seeds won at 10 %


def test_load_results_without_runs_explains_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run-experiments"):
        report.load_results(tmp_path)
