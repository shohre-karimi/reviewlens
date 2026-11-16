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
/* A collapsed sidebar is moved out of view to the LEFT, but RTL puts the sidebar on the
   right, so its contents would stay visible as a thin strip over the page. Hide them; the
   expand button (>>) is outside the sidebar and keeps working. "*" and !important are
   needed because Streamlit sets visibility: visible on its own element containers. */
[data-testid="stSidebar"][data-collapsed="true"],
[data-testid="stSidebar"][data-collapsed="true"] * {
    visibility: hidden !important;
}
/* The Deploy button and the menu: their box packs them to its "end", which in RTL is the
   LEFT; pack them to the start (the right). */
[data-testid="stToolbar"] div:has(> [data-testid="stToolbarActions"]) {
    justify-content: flex-start;
}
/* On wide screens (>= 768px, where the sidebar sits beside the page) move them to the
   top-right corner of the WINDOW, above the right sidebar (whose header is empty there; its
   collapse button is on its left side). Below 768px the sidebar opens OVER the page, and a
   lifted header would cover the sidebar's own close button, so nothing changes there. */
@media (min-width: 768px) {
    [data-testid="stToolbar"] div:has(> [data-testid="stToolbarActions"]) {
        position: fixed; /* out of the header, which in RTL ends at the sidebar edge */
        top: 16px;
        right: 16px;
        margin: 0 !important;
        width: auto !important;
        height: auto !important; /* its default height: 100% would be the whole window */
    }
    /* The box is only as high as its header's layer, which is below the sidebar's
       (999991): lift the header so the buttons are not hidden behind the sidebar. */
    [data-testid="stHeader"] {
        z-index: 999992;
    }
    /* Sidebar collapsed: its expand button (>>) takes the corner, so move left of it. */
    .stApp:has([data-testid="stSidebar"][data-collapsed="true"])
        [data-testid="stToolbar"] div:has(> [data-testid="stToolbarActions"]) {
        right: 52px;
    }
}
/* Code, JSON, tables and figures stay left-to-right. */
code, pre, [data-testid="stCode"], [data-testid="stJson"], [data-testid="stDataFrame"],
[data-testid="stImage"], [data-testid="stVegaLiteChart"] {
    direction: ltr;
    text-align: left;
}
</style>
"""


# Streamlit's own top-right menu and Deploy button are English-only (no translation option).
# Each label is hidden (font-size: 0) and replaced by Persian text in ::after, at the original
# size. Items are found by their data-testid (stable). The footer ("Made with Streamlit") is
# hidden; it has no data-testid, so it is found as the menu's only span outside a button.
# Checked against Streamlit 1.67; after an upgrade, a label may show in English again if
# Streamlit renames a data-testid.
MENU_LABELS_FA = {
    "stMainMenuItem-theme-System": "سیستم",
    "stMainMenuItem-theme-Light": "روشن",
    "stMainMenuItem-theme-Dark": "تیره",
    "stMainMenuItem-rerun": "اجرای دوباره",
    "stMainMenuItem-autoRerun": "اجرای خودکار",
    "stMainMenuItem-clearCache": "پاک کردن حافظه‌ی موقت",
    "stMainMenuItem-print": "چاپ",
    "stMainMenuItem-recordScreencast": "ضبط صفحه",
}
# Items whose label is a bare text node inside the button (no span of its own).
_TEXT_NODE_ITEMS = {
    "stMainMenuItem-theme-System",
    "stMainMenuItem-theme-Light",
    "stMainMenuItem-theme-Dark",
    "stMainMenuItem-autoRerun",
}


def menu_fa_css() -> str:
    """CSS that shows Persian labels in Streamlit's built-in menu and Deploy button."""
    rules = [
        '[data-testid="stMainMenuPopover"] { direction: rtl; }',
        '[data-testid="stMainMenuPopover"] [data-testid="stIconMaterial"] { font-size: 20px; }',
    ]
    for testid, text in MENU_LABELS_FA.items():
        item = f'[data-testid="{testid}"]'
        # Text-node labels: shrink the whole button; otherwise only its label span.
        label = (
            item if testid in _TEXT_NODE_ITEMS else f'{item} [data-testid="stMainMenuItemLabel"]'
        )
        rules.append(f"{label} {{ font-size: 0; }}")
        rules.append(f'{label}::after {{ content: "{text}"; font-size: 14px; }}')
    rules += [
        '[data-testid="stAppDeployButton"] span { font-size: 0; }',
        '[data-testid="stAppDeployButton"] span::after { content: "انتشار"; font-size: 14px; }',
        # The "Made with Streamlit vX" footer is only a credit line with no function: hidden.
        '[data-testid="stMainMenuPopover"] span:not([data-testid]):not(button *) '
        "{ display: none; }",
    ]
    return "<style>\n" + "\n".join(rules) + "\n</style>"


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
    """Right-to-left layout plus Persian labels for Streamlit's built-in menu."""
    st.markdown(RTL_CSS + menu_fa_css(), unsafe_allow_html=True)
