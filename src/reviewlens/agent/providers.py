"""Language-model providers behind one small interface, ``LLMClient``.

The agent loop only knows the provider-neutral conversation below (user turns, assistant
turns with tool calls, tool results). Each client converts it to its API's format:

- ``AnthropicClient``         Claude through the official ``anthropic`` SDK (default)
- ``OpenAICompatibleClient``  any OpenAI-compatible server: OpenAI itself, or a local
                              model through Ollama (optional ``openai`` extra)

Tests use a scripted fake client with the same interface, so the loop is tested without
any network access.

Every API problem (no key, wrong key, no network, refusal, ...) becomes a ``ProviderError``
that carries a ready-made Persian message for the user instead of a stack trace.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from reviewlens.agent.schemas import ToolSpec
from reviewlens.config import AgentConfig

logger = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
REQUEST_TIMEOUT_SECONDS = 120.0  # local models on a laptop CPU can be slow

# --------------------------------------------------------------------------- conversation


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Any  # normally a dict; the raw string if the model sent invalid JSON


@dataclass(frozen=True)
class UserTurn:
    text: str


@dataclass(frozen=True)
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    # The provider's own content (e.g. Claude's blocks, including thinking blocks), sent
    # back unchanged on the next request. Some models require their thinking to be returned.
    raw: Any = None


@dataclass(frozen=True)
class ToolResultMessage:
    call_id: str
    content: str  # JSON text of the tool output
    is_error: bool


@dataclass(frozen=True)
class ToolResultsTurn:
    results: list[ToolResultMessage]


Turn = UserTurn | AssistantTurn | ToolResultsTurn


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    raw: Any = None


class LLMClient(Protocol):
    """Anything that can answer one step of the conversation."""

    model: str

    def complete(
        self, system: str, history: Sequence[Turn], tools: Sequence[ToolSpec]
    ) -> LLMResponse: ...


# --------------------------------------------------------------------------- errors

ErrorKind = Literal[
    "missing_key", "auth", "connection", "rate_limit", "refusal", "truncated", "config", "api"
]

PERSIAN_ERRORS: dict[str, str] = {
    "missing_key": "کلید API تنظیم نشده است. متغیر محیطی ANTHROPIC_API_KEY (یا OPENAI_API_KEY) "
    "را تنظیم کنید، یا از حالت مدل محلی (Ollama) استفاده کنید.",
    "auth": "کلید API نامعتبر است یا اجازه‌ی دسترسی ندارد. کلید را بررسی کنید.",
    "connection": "اتصال به سرویس مدل زبانی برقرار نشد. اینترنت را بررسی کنید؛ "
    "اگر از Ollama استفاده می‌کنید، مطمئن شوید که اجرا شده است.",
    "rate_limit": "تعداد درخواست‌ها از حد مجاز گذشته است. کمی بعد دوباره امتحان کنید.",
    "refusal": "مدل زبانی به این درخواست پاسخ نداد. لطفاً سؤال را به شکل دیگری بپرسید.",
    "truncated": "پاسخ مدل نیمه‌کاره ماند (محدودیت طول پاسخ). سؤال کوتاه‌تری بپرسید.",
    "config": "تنظیمات مدل زبانی کامل نیست.",
    "api": "سرویس مدل زبانی خطا داد. کمی بعد دوباره امتحان کنید.",
}


class ProviderError(Exception):
    """An API problem, with a Persian message that can be shown to the user as-is."""

    def __init__(self, kind: ErrorKind, detail: str = "") -> None:
        self.kind = kind
        self.detail = detail
        message = PERSIAN_ERRORS[kind]
        self.persian_message = f"{message} ({detail})" if detail and kind == "config" else message
        super().__init__(f"{kind}: {detail}" if detail else kind)


def _parse_arguments(text: str) -> Any:
    try:
        return json.loads(text) if text else {}
    except json.JSONDecodeError:
        return text  # run_tool reports "arguments must be a JSON object" to the model


# --------------------------------------------------------------------------- Anthropic

# Models that accept server-side refusal fallbacks in their "default" form: if a safety
# classifier declines, the API retries the same request on a suitable fallback model.
_FALLBACK_MODELS = {
    "claude-fable-5-2",
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicClient:
    """Claude through the official SDK (Messages API with client-side tools)."""

    def __init__(self, model: str, max_tokens: int, api_key: str | None = None) -> None:
        import anthropic

        self._anthropic = anthropic
        self.model = model
        self.max_tokens = max_tokens
        # api_key=None lets the SDK find credentials itself (ANTHROPIC_API_KEY, ...).
        self._client = anthropic.Anthropic(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)

    @staticmethod
    def _to_messages(history: Sequence[Turn]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for turn in history:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.text})
            elif isinstance(turn, AssistantTurn):
                content: Any = turn.raw
                if content is None:  # e.g. a turn created by tests: rebuild the blocks
                    content = ([{"type": "text", "text": turn.text}] if turn.text else []) + [
                        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                        for c in turn.tool_calls
                    ]
                messages.append({"role": "assistant", "content": content})
            else:  # all results of one step go back together, in a single user message
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": r.call_id,
                                "content": r.content,
                                "is_error": r.is_error,
                            }
                            for r in turn.results
                        ],
                    }
                )
        return messages

    def complete(
        self, system: str, history: Sequence[Turn], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        anthropic = self._anthropic
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": self._to_messages(history),
            "tools": [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ],
        }
        try:
            if self.model in _FALLBACK_MODELS:
                response = self._client.beta.messages.create(
                    betas=[_FALLBACK_BETA], fallbacks="default", **request
                )
            else:
                response = self._client.messages.create(**request)
        except anthropic.AuthenticationError as error:
            raise ProviderError("auth") from error
        except anthropic.PermissionDeniedError as error:
            raise ProviderError("auth") from error
        except anthropic.NotFoundError as error:
            raise ProviderError("config", f"model '{self.model}' not found") from error
        except anthropic.RateLimitError as error:
            raise ProviderError("rate_limit") from error
        except anthropic.APIStatusError as error:
            raise ProviderError("api", str(error.status_code)) from error
        except anthropic.APIConnectionError as error:  # also covers timeouts
            raise ProviderError("connection") from error
        except TypeError as error:  # the SDK's "no credentials found" error
            if "authentication" in str(error).lower():
                raise ProviderError("missing_key") from error
            raise

        if response.stop_reason == "refusal":
            raise ProviderError("refusal")
        if response.stop_reason == "max_tokens":
            raise ProviderError("truncated")
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, b.input) for b in response.content if b.type == "tool_use"]
        return LLMResponse(text=text, tool_calls=calls, raw=response.content)


# --------------------------------------------------------------------------- OpenAI-compatible


class OpenAICompatibleClient:
    """OpenAI Chat Completions format: works with OpenAI and with Ollama's /v1 endpoint."""

    def __init__(
        self, model: str, max_tokens: int, base_url: str | None, api_key: str | None
    ) -> None:
        try:
            import openai
        except ImportError as error:
            raise ProviderError("config", "install the extra: pip install -e .[openai]") from error
        self._openai = openai
        self.model = model
        self.max_tokens = max_tokens
        self._client = openai.OpenAI(
            api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT_SECONDS
        )

    @staticmethod
    def _to_messages(system: str, history: Sequence[Turn]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for turn in history:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.text})
            elif isinstance(turn, AssistantTurn):
                message: dict[str, Any] = {"role": "assistant", "content": turn.text or None}
                if turn.tool_calls:
                    message["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": c.arguments
                                if isinstance(c.arguments, str)
                                else json.dumps(c.arguments, ensure_ascii=False),
                            },
                        }
                        for c in turn.tool_calls
                    ]
                messages.append(message)
            else:  # one "tool" message per result
                messages.extend(
                    {"role": "tool", "tool_call_id": r.call_id, "content": r.content}
                    for r in turn.results
                )
        return messages

    def complete(
        self, system: str, history: Sequence[Turn], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        openai = self._openai
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=self._to_messages(system, history),  # type: ignore[arg-type]
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        },
                    }
                    for t in tools
                ],
            )
        except (openai.AuthenticationError, openai.PermissionDeniedError) as error:
            raise ProviderError("auth") from error
        except openai.NotFoundError as error:  # e.g. the model is not pulled in Ollama
            raise ProviderError("config", f"model '{self.model}' not found") from error
        except openai.RateLimitError as error:
            raise ProviderError("rate_limit") from error
        except openai.APIStatusError as error:
            raise ProviderError("api", str(error.status_code)) from error
        except openai.APIConnectionError as error:  # also covers timeouts
            raise ProviderError("connection") from error

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise ProviderError("truncated")
        calls = [
            ToolCall(c.id, c.function.name, _parse_arguments(c.function.arguments))
            for c in choice.message.tool_calls or []
            if c.type == "function"
        ]
        return LLMResponse(text=choice.message.content or "", tool_calls=calls)


# --------------------------------------------------------------------------- factory


def create_client(cfg: AgentConfig, api_key: str | None = None) -> LLMClient:
    """Build the client chosen in the config. ``api_key`` overrides the environment.

    Model names come from environment variables, never from code or committed files:
    ANTHROPIC_MODEL (default: claude-opus-5-5) or OPENAI_MODEL (required).
    """
    if cfg.provider == "anthropic":
        model = os.environ.get("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL
        return AnthropicClient(model, cfg.max_tokens, api_key=api_key)

    model = os.environ.get("OPENAI_MODEL", "")
    if not model:
        raise ProviderError("config", "set OPENAI_MODEL, e.g. qwen2.5:7b for Ollama")
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        if cfg.openai_base_url is None:  # the real OpenAI API needs a key
            raise ProviderError("missing_key")
        key = "local"  # Ollama ignores the key, but the SDK requires a value
    return OpenAICompatibleClient(model, cfg.max_tokens, cfg.openai_base_url, key)
