"""Streamlit app: every page is run headless with Streamlit's AppTest (no browser).

The app reads its config from REVIEWLENS_CONFIG, so each test points it at a temporary
config built from the toy fixtures. The agent page uses a scripted fake model.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from reviewlens.agent import providers
from reviewlens.config import AppConfig
from reviewlens.experiments.runner import PSEUDO_FILE, RUNS_FILE, experiment_dir
from tests.test_agent_loop import FakeClient, answer, call
from tests.test_report import make_pseudo, make_runs

APP = Path(__file__).resolve().parents[1] / "src" / "reviewlens" / "app" / "streamlit_app.py"


def use_config(cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "app_config.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    monkeypatch.setenv("REVIEWLENS_CONFIG", str(path))
    monkeypatch.delenv("REVIEWLENS_EXPERIMENT", raising=False)


def start(page: str | None = None) -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    if page is not None:
        app.switch_page(f"app_pages/{page}.py").run()
    assert not app.exception, app.exception
    return app


@pytest.fixture
def app_config(
    agent_config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AppConfig:
    use_config(agent_config, tmp_path, monkeypatch)
    return agent_config


# --------------------------------------------------------------------------- menu


def test_menu_css_has_a_persian_label_for_every_item() -> None:
    from reviewlens.app.common import MENU_LABELS_FA, menu_fa_css

    css = menu_fa_css()
    for testid, text in MENU_LABELS_FA.items():
        assert f'[data-testid="{testid}"]' in css
        assert f'content: "{text}"' in css
    assert 'content: "انتشار"' in css  # the Deploy button
    assert "display: none" in css  # the "Made with Streamlit" footer is hidden
    assert css.startswith("<style>") and css.endswith("</style>")


# --------------------------------------------------------------------------- analyze


def test_analyze_page_shows_sentiment_and_words(app_config: AppConfig) -> None:
    app = start()
    assert app.title[0].value == "تحلیل احساس یک نظر"

    app.text_area[0].input("غذا عالی و خوشمزه بود")
    app.button[0].click().run()

    assert not app.exception
    assert app.metric[0].value == "مثبت"
    assert app.metric[1].value.endswith("%")


def test_analyze_page_empty_text_shows_persian_error(app_config: AppConfig) -> None:
    app = start()
    app.button[0].click().run()
    assert "لطفاً متن یک نظر" in app.error[0].value


def test_analyze_page_without_model_explains_training(
    toy_config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_config(toy_config, tmp_path, monkeypatch)
    app = start()
    app.text_area[0].input("غذا خوب بود")
    app.button[0].click().run()
    assert "reviewlens train" in app.error[0].value


# --------------------------------------------------------------------------- results


def test_results_page_without_results_explains_what_to_run(app_config: AppConfig) -> None:
    app = start("results")
    assert "reviewlens run-experiments" in app.info[0].value


def test_results_page_shows_answer_table_and_figures(app_config: AppConfig) -> None:
    out_dir = experiment_dir(app_config)
    make_runs().to_csv(out_dir / RUNS_FILE, index=False)
    make_pseudo().to_csv(out_dir / PSEUDO_FILE, index=False)

    app = start("results")

    # pseudo_labeling beats the best supervised model on only 2 of 3 seeds: not "stable"
    assert "به‌طور پایدار" in app.info[0].value
    assert len(app.dataframe) == 2  # main table + paired comparison
    assert "Logistic Regression" in app.dataframe[0].value.index


# --------------------------------------------------------------------------- agent


def test_agent_page_without_model_shows_setup_help(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_model(cfg: object, api_key: str | None = None) -> providers.LLMClient:
        raise providers.ProviderError("missing_key")

    monkeypatch.setattr(providers, "create_client", no_model)
    app = start("agent_chat")

    assert "کلید API تنظیم نشده" in app.warning[0].value
    assert "Ollama" in app.markdown[-1].value
    assert len(app.chat_input) == 0


def test_agent_page_chat_shows_answer_and_trace(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeClient(
        [call(("analyze_review", {"text": "غذا عالی بود"})), answer("این نظر مثبت است.")]
    )
    monkeypatch.setattr(providers, "create_client", lambda cfg, api_key=None: fake)
    app = start("agent_chat")

    app.chat_input[0].set_value("تحلیل کن: غذا عالی بود").run()

    assert not app.exception
    assert [m.name for m in app.chat_message] == ["user", "assistant"]
    assert "این نظر مثبت است." in app.chat_message[1].markdown[0].value
    assert "ردپای ابزارها (1 فراخوانی)" in [e.label for e in app.expander]


def test_agent_page_shows_provider_errors_in_red(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeClient([providers.ProviderError("auth")])
    monkeypatch.setattr(providers, "create_client", lambda cfg, api_key=None: fake)
    app = start("agent_chat")

    app.chat_input[0].set_value("سلام").run()

    assert "کلید API نامعتبر" in app.chat_message[1].error[0].value


def test_agent_page_warns_up_front_when_no_key_is_set(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(providers, "create_client", lambda cfg, api_key=None: FakeClient([]))

    app = start("agent_chat")

    assert "کلید API در متغیرهای محیطی پیدا نشد" in app.info[0].value
    assert len(app.chat_input) == 1  # chat stays available (a login profile may work)


def test_agent_page_new_chat_clears_history(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeClient([answer("سلام!")])
    monkeypatch.setattr(providers, "create_client", lambda cfg, api_key=None: fake)
    app = start("agent_chat")
    app.chat_input[0].set_value("سلام").run()
    assert len(app.chat_message) == 2

    app.sidebar.button[0].click().run()

    assert len(app.chat_message) == 0
