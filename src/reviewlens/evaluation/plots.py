"""Figures for the experiment results.

Each function returns a matplotlib ``Figure`` and does not save or show it, so the same
figure can be written to a file (``save_figure``) or displayed in the Streamlit app.
``Figure`` is used directly instead of ``pyplot``: no global state, no GUI window needed.

Figure text is in English: matplotlib does not shape Persian (right-to-left, joined letters)
without extra libraries, and the report figures do not need it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure

from reviewlens.experiments.runner import summarize


def learning_curves(
    runs: pd.DataFrame,
    is_semi_supervised: Callable[[str], bool],
    display_names: dict[str, str] | None = None,
) -> Figure:
    """The main figure: macro-F1 vs labeled fraction, one line per method, mean ± std band.

    Supervised methods are solid lines, semi-supervised methods dashed. The x axis is
    logarithmic because the fractions (1 %, 5 %, ... 100 %) span two orders of magnitude.

    Two panels with the same lines: the left shows the full range (including collapses
    such as a method at 0.34), the right zooms into the top ``zoom`` of macro-F1 where the
    best methods differ by less than 0.01 and would otherwise overlap.
    """
    summary = summarize(runs)
    names = display_names or {}
    fractions = sorted(summary["label_fraction"].unique())
    best = float(summary["macro_f1_mean"].max())
    zoom = 0.1

    fig = Figure(figsize=(13, 5), layout="constrained")
    ax_full, ax_zoom = fig.subplots(1, 2)
    for method, group in summary.groupby("method", sort=True):
        group = group.sort_values("label_fraction")
        x = group["label_fraction"].to_numpy()
        mean = group["macro_f1_mean"].to_numpy()
        std = np.nan_to_num(group["macro_f1_std"].to_numpy())
        style = "--" if is_semi_supervised(str(method)) else "-"
        label = names.get(str(method), str(method))
        for ax in (ax_full, ax_zoom):
            (line,) = ax.plot(x, mean, style, marker="o", label=label)
            ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.15)

    for ax in (ax_full, ax_zoom):
        ax.set_xscale("log")
        ax.set_xticks(fractions, [f"{f * 100:g}%" for f in fractions])
        ax.minorticks_off()
        ax.set_xlabel("Labeled fraction of the training set")
        ax.set_ylabel("Macro-F1 on the test set")
        ax.grid(alpha=0.3)
    ax_full.set_title("Full range")
    ax_zoom.set_ylim(best - zoom, best + 0.01)
    ax_zoom.set_title(f"Zoom: top {zoom:g} of macro-F1")
    ax_zoom.legend(
        title="solid = supervised, dashed = semi-supervised", fontsize=8, title_fontsize=8
    )
    fig.suptitle("Macro-F1 vs labeled fraction (mean ± std over seeds)")
    return fig


def pseudo_label_curves(curve: pd.DataFrame) -> Figure:
    """Pseudo-labels per iteration: one row per method, (added, accuracy) as columns.

    ``curve`` is the output of ``report.pseudo_label_curve``. Each line is one label fraction.
    The accuracy panel explains the results: a method that adds many *wrong* labels
    teaches its own mistakes to the next model (confirmation bias).

    Methods with a single iteration (Label Spreading labels everything at once) have no
    curve to show; they are left out here and reported in the pseudo-label table instead.
    """
    iterations = curve.groupby("method")["iteration"].max()
    methods = sorted(str(m) for m in iterations[iterations > 1].index)
    fig = Figure(figsize=(10, 3.2 * max(len(methods), 1)), layout="constrained")
    axes = fig.subplots(max(len(methods), 1), 2, squeeze=False)
    for row, method in enumerate(methods):
        ax_added, ax_acc = axes[row]
        for fraction, group in curve[curve["method"] == method].groupby("label_fraction"):
            label = f"{float(fraction) * 100:g}% labeled"  # type: ignore[arg-type]
            ax_added.plot(group["iteration"], group["added"], marker="o", label=label)
            ax_acc.plot(group["iteration"], group["accuracy"], marker="o", label=label)
        ax_added.set_title(f"{method}: pseudo-labels added")
        ax_added.set_ylabel("count (mean over seeds)")
        ax_acc.set_title(f"{method}: pseudo-label accuracy")
        ax_acc.set_ylabel("share correct")
        ax_acc.set_ylim(0, 1.02)
        for ax in (ax_added, ax_acc):
            ax.set_xlabel("iteration")
            ax.grid(alpha=0.3)
            ax.xaxis.get_major_locator().set_params(integer=True)
        ax_acc.legend(fontsize=8)
    return fig


def confusion_grid(
    matrices: dict[str, npt.NDArray[np.float64]], labels: Sequence[str], title: str
) -> Figure:
    """One row-normalized confusion matrix per method, side by side."""
    n = max(len(matrices), 1)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig = Figure(figsize=(3.2 * cols, 3.0 * rows), layout="constrained")
    axes = fig.subplots(rows, cols, squeeze=False).ravel()
    for ax, (method, matrix) in zip(axes, matrices.items(), strict=False):
        ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
            )
        ticks = range(len(labels))
        ax.set_xticks(ticks, labels)
        ax.set_yticks(ticks, labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(method, fontsize=9)
    for ax in axes[len(matrices) :]:
        ax.set_visible(False)  # empty cells of the grid
    fig.suptitle(title)
    return fig


def save_figure(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    return path
