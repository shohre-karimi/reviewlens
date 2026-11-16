"""Page 3: chat with the tool-calling agent; every answer shows its tool-call trace.

If no language model is configured (no key, no OPENAI_MODEL, ...), the page explains the
three run modes instead of failing: the tools still work on the "Analyze" page.
"""

from __future__ import annotations

import json
import os
from typing import Any

import streamlit as st

from reviewlens.agent.loop import TraceEntry, run_agent
from reviewlens.agent.providers import ProviderError, create_client
from reviewlens.app.common import get_tools, load_app_config

SETUP_HELP = """
**سه حالت اجرا:**

1. **API (پیش‌فرض، Anthropic):** متغیر محیطی `ANTHROPIC_API_KEY` را تنظیم کنید
   (نام مدل اختیاری: `ANTHROPIC_MODEL`).
2. **مدل محلی (Ollama):** Ollama را نصب و اجرا کنید، یک مدل با قابلیت tool calling بگیرید
   (مثلاً `ollama pull qwen2.5:7b`)، سپس:
   ```
   OPENAI_MODEL=qwen2.5:7b
   REVIEWLENS_EXPERIMENT=configs/agent/ollama.yaml
   ```
   و برنامه را دوباره اجرا کنید (`make app`). بسته‌ی `openai` لازم است: `pip install -e .[openai]`
3. **بدون مدل:** صفحه‌ی «تحلیل نظر» و دستور `reviewlens agent local-test` بدون مدل زبانی کار می‌کنند.
"""

EXAMPLES = [
    "این نظر را تحلیل کن: «غذا سرد بود و پیک دیر رسید»",
    "با ۱٪ داده‌ی برچسب‌دار کدام روش بهتر بود؟",
    "چند نظر منفی درباره‌ی پیتزا نشانم بده",
]


def show_trace(trace: list[TraceEntry]) -> None:
    """The tool calls behind one answer: step, tool, input and output."""
    if not trace:
        st.caption("بدون فراخوانی ابزار")
        return
    with st.expander(f"ردپای ابزارها ({len(trace)} فراخوانی)"):
        for entry in trace:
            status = "⚠️ خطا" if entry.is_error else "✅"
            st.markdown(f"**گام {entry.step}:** `{entry.tool}` {status}")
            st.code(json.dumps(entry.arguments, ensure_ascii=False, indent=2), language="json")
            st.json(entry.output, expanded=False)


cfg = load_app_config()
st.title("گفتگو با دستیار")

try:
    client = create_client(cfg.agent)
except ProviderError as error:
    st.warning(error.persian_message)
    st.markdown(SETUP_HELP)
    st.stop()

# The Anthropic SDK only looks for credentials when a request is sent, so a missing key is
# detected here to explain the setup up front (a login profile may still work, so the
# chat stays available).
if cfg.agent.provider == "anthropic" and not (
    os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
):
    st.info("کلید API در متغیرهای محیطی پیدا نشد؛ احتمالاً گفتگو کار نمی‌کند.")
    with st.expander("راهنمای راه‌اندازی"):
        st.markdown(SETUP_HELP)

st.caption(
    f"مدل: `{client.model}` ({cfg.agent.provider}). دستیار خودش تصمیم می‌گیرد کدام ابزار را "
    f"صدا بزند؛ حداکثر {cfg.agent.max_steps} گام برای هر پاسخ."
)

state = st.session_state
state.setdefault("history", [])  # provider-neutral turns, sent back to the model
state.setdefault("messages", [])  # what the page shows: role, text, trace

if st.sidebar.button("گفتگوی تازه"):
    state.history, state.messages = [], []

if not state.messages:
    st.markdown("**نمونه‌ی سؤال:**\n" + "\n".join(f"- {example}" for example in EXAMPLES))

for message in state.messages:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["text"])
        else:
            st.markdown(message["text"])
        if message["role"] == "assistant":
            show_trace(message["trace"])

if prompt := st.chat_input("سؤال یا متن یک نظر را بنویسید…"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"), st.spinner("در حال فکر کردن…"):
        result = run_agent(
            prompt, client, get_tools(), max_steps=cfg.agent.max_steps, history=state.history
        )
    reply: dict[str, Any] = {
        "role": "assistant",
        "text": result.answer,
        "trace": result.trace,
        "error": result.status == "provider_error",
    }
    state.messages += [{"role": "user", "text": prompt, "trace": []}, reply]
    state.history = result.history
    st.rerun()  # redraw the whole conversation, including the new answer and its trace
