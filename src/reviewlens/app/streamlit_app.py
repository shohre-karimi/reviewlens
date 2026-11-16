"""ReviewLens Streamlit app (Persian, right-to-left). Start it with ``make app``.

Three pages, one file each in ``app_pages/``:
1. Analyze  - sentiment of one review with the trained model (no language model needed)
2. Results  - experiment results table and the main figure
3. Agent    - chat with the tool-calling agent, with the tool-call trace of each answer
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from reviewlens.app.common import apply_rtl

PAGES_DIR = Path(__file__).parent / "app_pages"

st.set_page_config(page_title="ReviewLens", page_icon="🔎", layout="wide")
apply_rtl()

page = st.navigation(
    [
        st.Page(PAGES_DIR / "analyze.py", title="تحلیل نظر", icon="📝", default=True),
        st.Page(PAGES_DIR / "results.py", title="نتایج آزمایش‌ها", icon="📊"),
        st.Page(PAGES_DIR / "agent_chat.py", title="گفتگو با دستیار", icon="🤖"),
    ]
)
st.sidebar.caption("ReviewLens: تحلیل احساس نظرهای فارسی و یادگیری نیمه‌نظارتی")
page.run()
