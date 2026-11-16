"""Shared pieces of the Streamlit app: configuration, cached tools, RTL style, Persian labels.

The app reads the same YAML configuration as the CLI. Which files are used comes from
environment variables, so the run mode is chosen without changing code:

    REVIEWLENS_CONFIG      main config (default: configs/default.yaml)
    REVIEWLENS_EXPERIMENT  optional override, e.g. configs/agent/ollama.yaml (local model mode)
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from reviewlens.agent.tools import ReviewTools
from reviewlens.config import DEFAULT_CONFIG_PATH, AppConfig, load_config

# Label names in the config are English identifiers; the UI shows Persian.
LABELS_FA = {"negative": "منفی", "positive": "مثبت"}

RTL_CSS = """
<style>
.stApp, [data-testid="stSidebar"], [data-testid="stChatMessage"] {
    direction: rtl;
    text-align: right;
    font-family: Vazirmatn, Tahoma, sans-serif;
}
/* Streamlit sets left alignment on its text containers: override it. */
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {
    text-align: right;
}
[data-testid="stMarkdownContainer"] ul, [data-testid="stMarkdownContainer"] ol {
    padding-right: 1.5em;
    padding-left: 0;
}
/* Code, JSON, tables and figures stay left-to-right. */
code, pre, [data-testid="stCode"], [data-testid="stJson"], [data-testid="stDataFrame"],
[data-testid="stImage"], [data-testid="stVegaLiteChart"] {
    direction: ltr;
    text-align: left;
}
</style>
"""


def config_paths() -> tuple[str, str]:
    """(main config, override or ""): plain strings, so they can be cache keys."""
    main = os.environ.get("REVIEWLENS_CONFIG", str(DEFAULT_CONFIG_PATH))
    return main, os.environ.get("REVIEWLENS_EXPERIMENT", "")


def load_app_config() -> AppConfig:
    main, override = config_paths()
    return load_config(Path(main), [Path(override)] if override else [])


@st.cache_resource(show_spinner=False)
def _cached_tools(main: str, override: str) -> ReviewTools:
    del main, override  # only used as the cache key
    return ReviewTools(load_app_config())


def get_tools() -> ReviewTools:
    """One ``ReviewTools`` per configuration: the model and data are loaded only once."""
    return _cached_tools(*config_paths())


def label_fa(label: str) -> str:
    return LABELS_FA.get(label, label)


def apply_rtl() -> None:
    st.markdown(RTL_CSS, unsafe_allow_html=True)
