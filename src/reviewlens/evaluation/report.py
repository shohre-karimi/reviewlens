"""Turn raw experiment results into the tables that answer the research question.

Inputs are the CSV files written by the experiment runner:
- ``runs.csv``          one row per (method, label fraction, seed)
- ``pseudo_labels.csv`` one row per semi-supervised iteration

Everything here is plain pandas on those tables: no model is trained or loaded.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from reviewlens.experiments.runner import PSEUDO_FILE, RUNS_FILE, summarize


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``runs.csv`` and ``pseudo_labels.csv`` from one experiment directory."""
    runs_path = results_dir / RUNS_FILE
    if not runs_path.is_file():
        raise FileNotFoundError(
            f"No results in {results_dir}. Run `reviewlens run-experiments` first."
        )
    runs = pd.read_csv(runs_path)
    pseudo_path = results_dir / PSEUDO_FILE
    pseudo = pd.read_csv(pseudo_path) if pseudo_path.is_file() else pd.DataFrame()
    return runs, pseudo


def compare_to_supervised(
    runs: pd.DataFrame, is_semi_supervised: Callable[[str], bool]
) -> pd.DataFrame:
    """For each label fraction: every semi-supervised method vs the best supervised method.

    The "best supervised" method is chosen per fraction by its mean macro-F1 over seeds.
    The comparison is **paired by seed**: seed ``s`` hides the same labels for every method,
    so ``semi(s) - supervised(s)`` compares the two models on exactly the same labeled rows.

    Columns: label_fraction, method, best_supervised, semi_f1, supervised_f1,
    diff_mean, diff_std, wins (seeds where the semi-supervised method scored higher), n_seeds.
    """
    methods = runs["method"].unique()
    semi = [m for m in methods if is_semi_supervised(m)]
    supervised = [m for m in methods if not is_semi_supervised(m)]
    if not semi or not supervised:
        raise ValueError("Need at least one supervised and one semi-supervised method.")

    rows = []
    for fraction, group in runs.groupby("label_fraction", sort=True):
        # method x seed table of macro-F1 for this fraction
        scores = group.pivot_table(index="seed", columns="method", values="macro_f1")
        best = str(scores[supervised].mean().idxmax())
        for method in semi:
            if method not in scores:
                continue
            diff = (scores[method] - scores[best]).dropna()
            rows.append(
                {
                    "label_fraction": float(fraction),  # type: ignore[arg-type]
                    "method": method,
                    "best_supervised": best,
                    "semi_f1": float(scores[method].mean()),
                    "supervised_f1": float(scores[best].mean()),
                    "diff_mean": float(diff.mean()),
                    "diff_std": float(diff.std()) if len(diff) > 1 else 0.0,
                    "wins": int((diff > 0).sum()),
                    "n_seeds": len(diff),
                }
            )
    return pd.DataFrame(rows)


def pseudo_label_summary(pseudo: pd.DataFrame) -> pd.DataFrame:
    """Per method x fraction: how many pseudo-labels were added and how many were correct.

    Totals are summed over iterations within a run and then averaged over seeds. The
    accuracy is pooled (all correct / all added), so large iterations weigh more.
    """
    columns = ["method", "label_fraction", "iterations", "added", "accuracy"]
    if pseudo.empty:
        return pd.DataFrame(columns=columns)
    per_run = (
        pseudo.groupby(["method", "label_fraction", "seed"])
        .agg(
            iterations=("iteration", "max"), added=("n_added", "sum"), correct=("n_correct", "sum")
        )
        .reset_index()
    )
    per_method = (
        per_run.groupby(["method", "label_fraction"])
        .agg(
            iterations=("iterations", "mean"),
            added=("added", "mean"),
            correct=("correct", "mean"),
        )
        .reset_index()
    )
    per_method["accuracy"] = per_method["correct"] / per_method["added"].replace(0, np.nan)
    return per_method[columns]


def pseudo_label_curve(pseudo: pd.DataFrame) -> pd.DataFrame:
    """Per method x fraction x iteration: mean added and pooled accuracy over seeds."""
    grouped = pseudo.groupby(["method", "label_fraction", "iteration"])
    curve = grouped.agg(
        added=("n_added", "mean"), correct=("n_correct", "sum"), total=("n_added", "sum")
    ).reset_index()
    curve["accuracy"] = curve["correct"] / curve["total"].replace(0, np.nan)
    return curve[["method", "label_fraction", "iteration", "added", "accuracy"]]


def mean_confusion(
    runs: pd.DataFrame, method: str, label_fraction: float, n_labels: int
) -> npt.NDArray[np.float64]:
    """Confusion matrix averaged over seeds, rows normalized to 1 (row = true class).

    A normalized row reads as "of all truly negative reviews, which share was predicted
    negative / positive", i.e. the diagonal is the per-class recall.
    """
    columns = [f"cm_{t}_{p}" for t in range(n_labels) for p in range(n_labels)]
    missing = [c for c in columns if c not in runs]
    if missing:
        raise ValueError(
            "runs.csv has no confusion-matrix columns; re-run the experiment with --fresh."
        )
    selected = runs[(runs["method"] == method) & np.isclose(runs["label_fraction"], label_fraction)]
    if selected.empty:
        raise ValueError(f"No runs for method={method} label_fraction={label_fraction}.")
    counts = selected[columns].to_numpy(dtype=float).mean(axis=0).reshape(n_labels, n_labels)
    row_sums = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)


# --------------------------------------------------------------------------- Markdown


def _markdown_table(frame: pd.DataFrame) -> str:
    """Minimal Markdown table (avoids the optional `tabulate` dependency of ``to_markdown``)."""
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *body])


def _percent(fraction: float) -> str:
    return f"{fraction * 100:g}%"


def macro_f1_table(summary: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    """Methods as rows, fractions as columns, "mean ± std"; the best mean per column in bold."""
    fractions = sorted(summary["label_fraction"].unique())
    best = summary.groupby("label_fraction")["macro_f1_mean"].max()
    rows = []
    for method in methods:
        row = {"method": method}
        for fraction in fractions:
            cell = summary[(summary["method"] == method) & (summary["label_fraction"] == fraction)]
            if cell.empty:
                row[_percent(fraction)] = "-"
                continue
            mean = float(cell["macro_f1_mean"].iloc[0])
            std = float(np.nan_to_num(cell["macro_f1_std"].iloc[0]))
            text = f"{mean:.3f} ± {std:.3f}"
            # Bold every method whose rounded mean ties the best one.
            row[_percent(fraction)] = (
                f"**{text}**" if round(mean, 3) == round(best[fraction], 3) else text
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_report(
    runs: pd.DataFrame,
    pseudo: pd.DataFrame,
    is_semi_supervised: Callable[[str], bool],
    figure_names: Sequence[str] = (),
) -> str:
    """The full Markdown report: macro-F1 table, paired comparison, pseudo-label summary."""
    summary = summarize(runs)
    order = summary.groupby("method")["macro_f1_mean"].mean().sort_values(ascending=False).index
    comparison = compare_to_supervised(runs, is_semi_supervised)
    pseudo_stats = pseudo_label_summary(pseudo)

    comparison_view = pd.DataFrame(
        {
            "fraction": comparison["label_fraction"].map(_percent),
            "semi-supervised": comparison["method"],
            "best supervised": comparison["best_supervised"],
            "semi F1": comparison["semi_f1"].map("{:.3f}".format),
            "supervised F1": comparison["supervised_f1"].map("{:.3f}".format),
            "difference": [
                f"{m:+.3f} ± {s:.3f}"
                for m, s in zip(comparison["diff_mean"], comparison["diff_std"], strict=True)
            ],
            "seeds won": [
                f"{w}/{n}" for w, n in zip(comparison["wins"], comparison["n_seeds"], strict=True)
            ],
        }
    )
    pseudo_view = pd.DataFrame(
        {
            "method": pseudo_stats["method"],
            "fraction": pseudo_stats["label_fraction"].map(_percent),
            "iterations": pseudo_stats["iterations"].map("{:.1f}".format),
            "pseudo-labels added": pseudo_stats["added"].map("{:,.0f}".format),
            "accuracy": pseudo_stats["accuracy"].map("{:.1%}".format),
        }
    )

    parts = [
        "# Experiment report",
        "",
        "## Macro-F1 on the test set (mean ± std over seeds)",
        "",
        _markdown_table(macro_f1_table(summary, list(order))),
        "",
        "## Semi-supervised vs the best supervised model (paired by seed)",
        "",
        "`difference` = semi-supervised minus best supervised macro-F1, computed per seed "
        "(same hidden labels), then mean ± std. `seeds won` = seeds where the "
        "semi-supervised method scored higher.",
        "",
        _markdown_table(comparison_view),
        "",
        "## Pseudo-labels (mean over seeds; accuracy against the hidden true labels)",
        "",
        _markdown_table(pseudo_view) if not pseudo_view.empty else "_No pseudo-label data._",
    ]
    if figure_names:
        parts += ["", "## Figures", "", *[f"- `{name}`" for name in figure_names]]
    return "\n".join(parts) + "\n"
