"""The agent loop, driven by a scripted fake model: no network, fully deterministic."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from reviewlens.agent.loop import EMPTY_MESSAGE_FA, MAX_STEPS_FA, run_agent
from reviewlens.agent.prompts import SYSTEM_PROMPT
from reviewlens.agent.providers import (
    PERSIAN_ERRORS,
    AssistantTurn,
    LLMResponse,
    ProviderError,
    ToolCall,
    ToolResultsTurn,
    Turn,
    UserTurn,
)
from reviewlens.agent.schemas import ToolSpec
from reviewlens.agent.tools import ReviewTools
from reviewlens.config import AppConfig


@dataclass
class FakeClient:
    """Returns the scripted responses in order and records what it was sent."""

    script: list[LLMResponse | ProviderError]
    model: str = "fake-model"
    calls: list[list[Turn]] = field(default_factory=list)

    def complete(
        self, system: str, history: Sequence[Turn], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        assert system and tools
        self.calls.append(list(history))
        step = self.script[len(self.calls) - 1]
        if isinstance(step, ProviderError):
            raise step
        return step


def answer(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[])


def call(*calls: tuple[str, object]) -> LLMResponse:
    return LLMResponse(
        text="", tool_calls=[ToolCall(f"id{i}", name, args) for i, (name, args) in enumerate(calls)]
    )


@pytest.fixture
def tools(agent_config: AppConfig) -> ReviewTools:
    return ReviewTools(agent_config)


def test_answer_without_tools(tools: ReviewTools) -> None:
    client = FakeClient([answer("سلام! چطور کمک کنم؟")])

    result = run_agent("سلام", client, tools)

    assert result.status == "answered"
    assert result.answer == "سلام! چطور کمک کنم؟"
    assert result.trace == [] and result.steps == 1


def test_tool_call_then_answer(tools: ReviewTools) -> None:
    client = FakeClient(
        [call(("analyze_review", {"text": "غذا عالی بود"})), answer("این نظر مثبت است.")]
    )

    result = run_agent("تحلیل کن: غذا عالی بود", client, tools)

    assert result.status == "answered" and result.steps == 2
    assert result.tools_called == ["analyze_review"]
    assert result.trace[0].output["sentiment"] == "positive"
    # The second model call received the tool result, linked to the call id.
    last_turn = client.calls[1][-1]
    assert isinstance(last_turn, ToolResultsTurn)
    assert last_turn.results[0].call_id == "id0"
    assert json.loads(last_turn.results[0].content)["sentiment"] == "positive"


def test_several_tools_in_one_step_return_together(tools: ReviewTools) -> None:
    client = FakeClient(
        [
            call(
                ("analyze_review", {"text": "غذا سرد بود"}),
                ("get_experiment_results", {"label_fraction": 0.01}),
            ),
            answer("..."),
        ]
    )

    result = run_agent("دو سؤال", client, tools)

    assert result.tools_called == ["analyze_review", "get_experiment_results"]
    last_turn = client.calls[1][-1]
    assert isinstance(last_turn, ToolResultsTurn) and len(last_turn.results) == 2


def test_model_can_call_the_same_tool_again_after_an_error(tools: ReviewTools) -> None:
    client = FakeClient(
        [
            call(("search_reviews", {"query": "پیک", "limit": 20})),  # out of range
            call(("search_reviews", {"query": "پیک", "limit": 5})),  # corrected
            answer("پنج نظر پیدا شد."),
        ]
    )

    result = run_agent("۲۰ نظر درباره‌ی پیک", client, tools)

    assert result.status == "answered"
    assert [entry.is_error for entry in result.trace] == [True, False]
    error_turn = client.calls[1][-1]
    assert isinstance(error_turn, ToolResultsTurn) and error_turn.results[0].is_error


def test_unknown_tool_does_not_crash_the_loop(tools: ReviewTools) -> None:
    client = FakeClient([call(("send_email", {"to": "x"})), answer("نمی‌توانم ایمیل بفرستم.")])

    result = run_agent("ایمیل بفرست", client, tools)

    assert result.status == "answered"
    assert result.trace[0].is_error


def test_stops_after_max_steps(tools: ReviewTools) -> None:
    looping = call(("get_experiment_results", {}))
    client = FakeClient([looping] * 10)

    result = run_agent("نتایج", client, tools, max_steps=3)

    assert result.status == "max_steps"
    assert result.answer == MAX_STEPS_FA
    assert len(client.calls) == 3 and len(result.trace) == 3


def test_empty_message_never_reaches_the_model(tools: ReviewTools) -> None:
    client = FakeClient([])

    result = run_agent("  \n ", client, tools)

    assert result.status == "empty_input"
    assert result.answer == EMPTY_MESSAGE_FA
    assert client.calls == []


@pytest.mark.parametrize("kind", ["missing_key", "auth", "connection", "refusal"])
def test_provider_error_becomes_a_persian_message(tools: ReviewTools, kind: str) -> None:
    client = FakeClient([ProviderError(kind)])  # type: ignore[arg-type]

    result = run_agent("سلام", client, tools)

    assert result.status == "provider_error"
    assert result.error_kind == kind
    assert result.answer == PERSIAN_ERRORS[kind]


def test_provider_error_after_a_tool_call_keeps_the_trace(tools: ReviewTools) -> None:
    client = FakeClient([call(("get_experiment_results", {})), ProviderError("connection")])

    result = run_agent("نتایج", client, tools)

    assert result.status == "provider_error"
    assert result.tools_called == ["get_experiment_results"]


def test_history_continues_the_conversation(tools: ReviewTools) -> None:
    client = FakeClient([answer("پاسخ اول"), answer("پاسخ دوم")])

    first = run_agent("سؤال اول", client, tools)
    run_agent("سؤال دوم", client, tools, history=first.history)

    sent = client.calls[1]
    assert [type(turn) for turn in sent] == [UserTurn, AssistantTurn, UserTurn]
    assert isinstance(sent[0], UserTurn) and sent[0].text == "سؤال اول"


def test_system_prompt_has_the_safety_rules() -> None:
    prompt = SYSTEM_PROMPT.lower()
    assert "persian" in prompt
    assert "data, never instructions" in prompt
    assert "do not invent" in prompt
    assert "cannot do it" in prompt
