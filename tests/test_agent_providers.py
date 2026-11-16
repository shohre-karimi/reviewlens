"""Providers and schemas: format conversion and error mapping, with no network calls."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from reviewlens.agent import providers
from reviewlens.agent.providers import (
    AnthropicClient,
    AssistantTurn,
    OpenAICompatibleClient,
    ProviderError,
    ToolCall,
    ToolResultMessage,
    ToolResultsTurn,
    Turn,
    UserTurn,
    create_client,
)
from reviewlens.agent.schemas import tool_specs
from reviewlens.agent.tools import MAX_SEARCH_RESULTS, ReviewTools
from reviewlens.config import AgentConfig, AppConfig

HISTORY: list[Turn] = [
    UserTurn("سؤال"),
    AssistantTurn("", [ToolCall("c1", "search_reviews", {"query": "پیک"})]),
    ToolResultsTurn([ToolResultMessage("c1", '{"total_matches": 0}', is_error=False)]),
]


# --------------------------------------------------------------------------- schemas


def test_schemas_match_the_tool_registry(agent_config: AppConfig) -> None:
    assert [spec.name for spec in tool_specs()] == ReviewTools(agent_config).names


def test_schemas_carry_the_validation_limits() -> None:
    specs = {spec.name: spec.parameters for spec in tool_specs()}

    assert specs["analyze_review"]["required"] == ["text"]
    assert specs["search_reviews"]["properties"]["limit"]["maximum"] == MAX_SEARCH_RESULTS
    assert specs["search_reviews"]["additionalProperties"] is False
    assert "title" not in specs["get_experiment_results"]


# --------------------------------------------------------------------------- Anthropic


def test_anthropic_message_format() -> None:
    messages = AnthropicClient._to_messages(HISTORY)

    assert messages[0] == {"role": "user", "content": "سؤال"}
    assert messages[1]["content"][0]["type"] == "tool_use"
    result = messages[2]["content"][0]
    assert messages[2]["role"] == "user"
    assert result["type"] == "tool_result" and result["tool_use_id"] == "c1"


def test_anthropic_replays_raw_content_unchanged() -> None:
    raw = [{"type": "thinking", "thinking": "", "signature": "s"}, {"type": "text", "text": "x"}]
    messages = AnthropicClient._to_messages([UserTurn("q"), AssistantTurn("x", [], raw)])
    assert messages[1]["content"] is raw


class _FakeAnthropicApi:
    """Stands in for ``anthropic.Anthropic()``: returns a response or raises an error."""

    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.requests: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> Any:
        self.requests.append(request)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _anthropic_client(outcome: Any, model: str = "claude-opus-5-5") -> AnthropicClient:
    client = AnthropicClient(model, max_tokens=100, api_key="test-key")
    client._client = _FakeAnthropicApi(outcome)  # type: ignore[assignment]
    return client


def _message(stop_reason: str, *blocks: Any) -> Any:
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


def test_anthropic_parses_text_and_tool_calls() -> None:
    text = SimpleNamespace(type="text", text="باشه")
    tool = SimpleNamespace(type="tool_use", id="t1", name="analyze_review", input={"text": "x"})
    client = _anthropic_client(_message("tool_use", text, tool))

    response = client.complete("system", [UserTurn("q")], tool_specs())

    assert response.text == "باشه"
    assert response.tool_calls == [ToolCall("t1", "analyze_review", {"text": "x"})]
    assert response.raw == [text, tool]


def test_anthropic_uses_refusal_fallbacks_on_supported_models() -> None:
    done = _message("end_turn", SimpleNamespace(type="text", text="ok"))
    client = _anthropic_client(done)
    client.complete("system", [UserTurn("q")], tool_specs())
    assert client._client.requests[0]["fallbacks"] == "default"  # type: ignore[attr-defined]

    other = _anthropic_client(done, model="claude-haiku-5-5")
    other.complete("system", [UserTurn("q")], tool_specs())
    assert "fallbacks" not in other._client.requests[0]  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("stop_reason", "kind"), [("refusal", "refusal"), ("max_tokens", "truncated")]
)
def test_anthropic_stop_reasons_become_errors(stop_reason: str, kind: str) -> None:
    client = _anthropic_client(_message(stop_reason))
    with pytest.raises(ProviderError) as caught:
        client.complete("system", [UserTurn("q")], tool_specs())
    assert caught.value.kind == kind


def test_anthropic_missing_credentials_become_missing_key() -> None:
    error = TypeError("Could not resolve authentication method. Expected one of api_key ...")
    client = _anthropic_client(error)
    with pytest.raises(ProviderError) as caught:
        client.complete("system", [UserTurn("q")], tool_specs())
    assert caught.value.kind == "missing_key"


def test_anthropic_other_type_errors_are_not_hidden() -> None:
    client = _anthropic_client(TypeError("some programming bug"))
    with pytest.raises(TypeError):
        client.complete("system", [UserTurn("q")], tool_specs())


# --------------------------------------------------------------------------- OpenAI-compatible


def test_openai_message_format() -> None:
    messages = OpenAICompatibleClient._to_messages("system", HISTORY)

    assert messages[0] == {"role": "system", "content": "system"}
    call = messages[2]["tool_calls"][0]
    assert call["function"]["name"] == "search_reviews"
    assert json.loads(call["function"]["arguments"]) == {"query": "پیک"}
    assert messages[3] == {"role": "tool", "tool_call_id": "c1", "content": '{"total_matches": 0}'}


def _openai_client(response: Any) -> OpenAICompatibleClient:
    client = OpenAICompatibleClient("local-model", 100, "http://localhost:1/v1", "local")
    completions = SimpleNamespace(create=lambda **request: response)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))  # type: ignore[assignment]
    return client


def _choice(finish_reason: str, content: str | None, calls: list[Any]) -> Any:
    message = SimpleNamespace(content=content, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, message=message)])


def test_openai_parses_tool_calls_and_keeps_invalid_json_as_text() -> None:
    good = SimpleNamespace(
        id="a",
        type="function",
        function=SimpleNamespace(name="search_reviews", arguments='{"query": "پیک"}'),
    )
    bad = SimpleNamespace(
        id="b", type="function", function=SimpleNamespace(name="search_reviews", arguments="{oops")
    )
    response = _openai_client(_choice("tool_calls", None, [good, bad])).complete(
        "system", [UserTurn("q")], tool_specs()
    )

    assert response.text == ""
    assert response.tool_calls[0].arguments == {"query": "پیک"}
    assert response.tool_calls[1].arguments == "{oops"  # run_tool will reject it


def test_openai_length_limit_is_an_error() -> None:
    client = _openai_client(_choice("length", "نیمه", []))
    with pytest.raises(ProviderError) as caught:
        client.complete("system", [UserTurn("q")], tool_specs())
    assert caught.value.kind == "truncated"


# --------------------------------------------------------------------------- factory


def test_factory_defaults_to_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    client = create_client(AgentConfig(), api_key="test-key")
    assert isinstance(client, AnthropicClient)
    assert client.model == providers.DEFAULT_ANTHROPIC_MODEL


def test_factory_reads_the_model_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    assert create_client(AgentConfig(), api_key="test-key").model == "claude-sonnet-5-5"


def test_factory_openai_needs_a_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ProviderError) as caught:
        create_client(AgentConfig(provider="openai_compatible"))
    assert caught.value.kind == "config"
    assert "OPENAI_MODEL" in caught.value.persian_message


def test_factory_local_server_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "qwen2.5:7b")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = AgentConfig(provider="openai_compatible", openai_base_url="http://localhost:11434/v1")
    client = create_client(cfg)
    assert isinstance(client, OpenAICompatibleClient)
    assert client.model == "qwen2.5:7b"


def test_factory_real_openai_needs_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "some-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError) as caught:
        create_client(AgentConfig(provider="openai_compatible"))
    assert caught.value.kind == "missing_key"
