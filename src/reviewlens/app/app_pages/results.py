"""Page 2: experiment results (table + main figure), read from the saved CSV files.

Nothing is trained here: the page reuses the phase-5 functions (``report`` and ``plots``)
on ``runs.csv`` and ``pseudo_labels.csv``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from reviewlens.app.common import load_app_config
from reviewlens.evaluation import plots, report
from reviewlens.evaluation.evaluate import is_semi_supervised
from reviewlens.experiments.runner import experiment_dir, format_summary_table, summarize
from reviewlens.models.registry import get_spec


@st.cache_data(show_spinner=False)
def _load(results_dir: str, modified: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached until runs.csv changes (``modified`` is only part of the cache key)."""
    del modified
    return report.load_results(Path(results_dir))


def _percent(fraction: float) -> str:
    return f"{fraction * 100:g}٪"


def research_answer(comparison: pd.DataFrame) -> str:
    """One Persian sentence answering the research question, computed from the data.

    A semi-supervised method counts as "better" at a fraction only if it beat the best
    supervised method on average AND on every seed (paired comparison).
    """
    better = comparison[
        (comparison["diff_mean"] > 0) & (comparison["wins"] == comparison["n_seeds"])
    ]
    if better.empty:
        return (
            "در هیچ درصدی از داده‌ی برچسب‌دار، روش نیمه‌نظارتی به‌طور پایدار (در همه‌ی seedها) "
            "از بهترین روش نظارتی بهتر نبود."
        )
    cases = "، ".join(
        f"{method} در {_percent(float(fraction))}"
        for method, fraction in zip(better["method"], better["label_fraction"], strict=True)
    )
    return f"روش نیمه‌نظارتی در این حالت‌ها در همه‌ی seedها بهتر بود: {cases}."


cfg = load_app_config()
st.title("نتایج آزمایش‌ها")
st.markdown(
    "**سؤال پژوهش:** وقتی فقط بخش کوچکی از داده برچسب دارد، آیا روش‌های نیمه‌نظارتی از "
    "مدل‌های نظارتی که فقط با همان بخش برچسب‌دار آموزش دیده‌اند بهتر عمل می‌کنند؟"
)

results_dir = experiment_dir(cfg)
runs_file = results_dir / "runs.csv"
if not runs_file.is_file():
    st.info(
        "هنوز نتیجه‌ای ذخیره نشده است. آزمایش‌ها را اجرا کنید:\n\n"
        "`reviewlens run-experiments`  و سپس  `reviewlens evaluate`"
    )
    st.stop()

runs, pseudo = _load(str(results_dir), runs_file.stat().st_mtime)
comparison = report.compare_to_supervised(runs, is_semi_supervised)
st.info(research_answer(comparison))

st.subheader("macro-F1 روی داده‌ی آزمون (میانگین ± انحراف معیار روی seedها)")
table = format_summary_table(summarize(runs))
table.columns = [_percent(float(c)) for c in table.columns]
table.index = [get_spec(str(m)).description for m in table.index]
st.dataframe(table, width="stretch")
st.caption("ستون‌ها: درصد داده‌ی آموزش که برچسب دارد. macro-F1 بالاتر بهتر است.")

st.subheader("نمودار اصلی")
names = {str(m): get_spec(str(m)).description for m in runs["method"].unique()}
st.pyplot(plots.learning_curves(runs, is_semi_supervised, names))
st.caption("خط پیوسته: نظارتی، خط‌چین: نیمه‌نظارتی. نوار رنگی: ± انحراف معیار روی seedها.")

with st.expander("مقایسه‌ی جفت‌شده با بهترین روش نظارتی"):
    view = pd.DataFrame(
        {
            "درصد برچسب": comparison["label_fraction"].map(_percent),
            "روش نیمه‌نظارتی": comparison["method"],
            "بهترین نظارتی": comparison["best_supervised"],
            "اختلاف F1": comparison["diff_mean"].map("{:+.3f}".format),
            "برد در seedها": comparison["wins"].astype(str)
            + "/"
            + comparison["n_seeds"].astype(str),
        }
    )
    st.dataframe(view, hide_index=True, width="stretch")
    st.caption("اختلاف = F1 نیمه‌نظارتی منهای F1 بهترین نظارتی، برای هر seed جدا و سپس میانگین.")

if not pseudo.empty:
    with st.expander("شبه‌برچسب‌ها: چرا روش‌های نیمه‌نظارتی کمک نکردند؟"):
        st.pyplot(plots.pseudo_label_curves(report.pseudo_label_curve(pseudo)))
        st.caption(
            "ستون راست: درصد شبه‌برچسب‌های درست در هر تکرار. وقتی این درصد پایین می‌آید، مدل "
            "خطاهای خودش را یاد می‌گیرد (confirmation bias)."
        )
