"""Page 1: analyze one review with the trained model. No language model is involved.

It calls the same ``analyze_review`` tool the agent uses, so the app, the agent and the CLI
all give exactly the same prediction and the same input validation.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from reviewlens.app.common import get_tools, label_fa

ERRORS_FA = {
    "invalid_input": "لطفاً متن یک نظر را بنویسید (حداکثر ۲۰۰۰ نویسه).",
    "unavailable": "مدل هنوز آموزش داده نشده است. این دستور را اجرا کنید:\n\n"
    "`reviewlens train -m logistic_regression -f 1.0 --save`",
}

st.title("تحلیل احساس یک نظر")
st.caption(
    "پیش‌بینی مستقیم با مدل آموزش‌دیده (Logistic Regression روی همه‌ی داده‌ی آموزش)؛ "
    "به مدل زبانی و کلید API نیازی نیست."
)

text = st.text_area(
    "متن نظر",
    height=120,
    placeholder="مثلاً: غذا خیلی خوشمزه بود ولی پیک دیر رسید",
)

if st.button("تحلیل کن", type="primary"):
    result = get_tools().run_tool("analyze_review", {"text": text})
    if result.is_error:
        kind = result.output["error"]["type"]
        st.error(ERRORS_FA.get(kind, result.output["error"]["message"]))
    else:
        output = result.output
        left, right = st.columns(2)
        left.metric("احساس", label_fa(output["sentiment"]))
        right.metric("اطمینان مدل", f"{output['confidence']:.0%}")
        st.progress(output["confidence"])

        st.subheader("کلمه‌های مهم")
        if output["top_words"]:
            words = pd.DataFrame(output["top_words"]).rename(
                columns={"word": "کلمه", "contribution": "سهم"}
            )
            st.bar_chart(words.set_index("کلمه")["سهم"], horizontal=True)
            st.caption(
                f"سهم مثبت یعنی این کلمه نظر را به سمت «{label_fa(output['sentiment'])}» برده است؛ "
                "سهم منفی یعنی در جهت مخالف. کلمه‌هایی که مدل در آموزش ندیده، سهمی ندارند."
            )
        else:
            st.info("هیچ‌کدام از کلمه‌های این متن در واژگان مدل نبود؛ پیش‌بینی قابل‌اعتماد نیست.")
