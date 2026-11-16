"""The agent loop: the language model decides which tools to call, the loop runs them.

    user message
      -> model (with the system prompt and the tool definitions)
      -> tool calls?  yes: run each through run_tool, send the results back, ask again
                      no:  the text is the final answer
      -> stop after max_steps model calls at most

There is no if-based routing of requests to tools: which tool, with which arguments and how
many times is decided by the model alone. The loop only executes and records.

Every tool call is recorded in the trace (shown in the app and used by the evaluation).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from reviewlens.agent.prompts import SYSTEM_PROMPT
from reviewlens.agent.providers import (
    AssistantTurn,
    LLMClient,
    ProviderError,
    ToolResultMessage,
    ToolResultsTurn,
    Turn,
    UserTurn,
)
from reviewlens.agent.schemas import ToolSpec, tool_specs
from reviewlens.agent.tools import ReviewTools

logger = logging.getLogger(__name__)

EMPTY_MESSAGE_FA = "پیام شما خالی است. لطفاً یک سؤال یا یک نظر بنویسید."
MAX_STEPS_FA = "نتوانستم در تعداد گام‌های مجاز به پاسخ برسم. لطفاً سؤال را ساده‌تر یا کوتاه‌تر بپرسید."

Status = Literal["answered", "empty_input", "max_steps", "provider_error"]


@dataclass(frozen=True)
class TraceEntry:
    """One tool call: which step requested it, with what input, and what came back."""

    step: int
    tool: str
    arguments: Any
    output: dict[str, Any]
    is_error: bool


@dataclass
class AgentResult:
    answer: str
    status: Status
    trace: list[TraceEntry] = field(default_factory=list)
    history: list[Turn] = field(default_factory=list)  # pass back in for a follow-up question
    steps: int = 0  # number of model calls
    error_kind: str | None = None

    @property
    def tools_called(self) -> list[str]:
        return [entry.tool for entry in self.trace]


def run_agent(
    message: str,
    client: LLMClient,
    tools: ReviewTools,
    max_steps: int = 5,
    history: Sequence[Turn] = (),
    system: str = SYSTEM_PROMPT,
    specs: Sequence[ToolSpec] | None = None,
) -> AgentResult:
    """Answer one user message, continuing an earlier conversation if ``history`` is given."""
    # Input validation, not routing: an empty message is never sent to the model.
    if not message.strip():
        return AgentResult(answer=EMPTY_MESSAGE_FA, status="empty_input", history=list(history))

    specs = list(specs) if specs is not None else tool_specs()
    turns: list[Turn] = [*history, UserTurn(message.strip())]
    trace: list[TraceEntry] = []

    for step in range(1, max_steps + 1):
        try:
            response = client.complete(system, turns, specs)
        except ProviderError as error:
            logger.warning("Provider error kind=%s detail=%s", error.kind, error.detail)
            return AgentResult(
                answer=error.persian_message,
                status="provider_error",
                trace=trace,
                history=list(history),  # the failed question is not kept
                steps=step,
                error_kind=error.kind,
            )

        turns.append(AssistantTurn(response.text, response.tool_calls, response.raw))
        if not response.tool_calls:  # no tool requested: this is the final answer
            logger.info("Agent answered steps=%d tool_calls=%d", step, len(trace))
            return AgentResult(
                answer=response.text.strip(),
                status="answered",
                trace=trace,
                history=turns,
                steps=step,
            )

        results = []
        for call in response.tool_calls:
            result = tools.run_tool(call.name, call.arguments)
            trace.append(
                TraceEntry(step, call.name, call.arguments, result.output, result.is_error)
            )
            results.append(
                ToolResultMessage(
                    call_id=call.id,
                    content=json.dumps(result.output, ensure_ascii=False),
                    is_error=result.is_error,
                )
            )
        turns.append(ToolResultsTurn(results))

    logger.warning("Agent hit max_steps=%d", max_steps)
    return AgentResult(
        answer=MAX_STEPS_FA, status="max_steps", trace=trace, history=turns, steps=max_steps
    )
