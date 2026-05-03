"""
Base class and types for LLM providers.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .types import AssistantResponse, Message, ToolDefinition, ToolResultRef

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration passed to each provider."""

    model: str = ""
    model_fallbacks: list[str] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 4096
    reasoning_effort: str = "medium"
    api_key: str = ""
    api_base: str = ""
    home_dir: str = "~/.predacore"
    # Provider-specific extras
    extras: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    name: str = "base"
    supports_native_tools: bool = True
    executes_tool_loop_internally: bool = False

    def __init__(self, config: ProviderConfig, log: logging.Logger | None = None):
        self.config = config
        self._log = log or logger

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream_fn: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request and return standardized result.

        Args:
            stream_fn: Optional callback invoked with each token as it arrives.
                       When provided and the response contains no tool_calls,
                       the provider should stream tokens via this callback.

        Returns:
            {
                "content": str,
                "tool_calls": list[dict],
                "usage": {"prompt_tokens": int, "completion_tokens": int},
                "finish_reason": str,
            }
        """
        ...

    # ------------------------------------------------------------------
    # Tool-turn serialization (Phase A refactor — 2026-04-21).
    #
    # These methods let each provider OWN how tool calls and tool results
    # are appended into the message history for the NEXT round-trip. This
    # replaces the structured-vs-flat fork that used to live in core.py
    # (which emitted literal "[Calling tool: X]" text stubs for non-Anthropic
    # providers, causing in-context poisoning).
    #
    # The defaults below are OpenAI-shaped — correct for OpenAI itself and
    # for all 11 OpenAI-compatible providers (Groq, xAI, Cerebras, DeepSeek,
    # OpenRouter, SambaNova, Mistral, Fireworks, NVIDIA, Zhipu, Together,
    # Ollama v0.3+). Anthropic and Gemini override with their own shapes.
    #
    # New providers: inherit these defaults if OpenAI-compatible; override
    # only when the wire format differs.
    # ------------------------------------------------------------------

    def append_assistant_turn(
        self,
        messages: list[Message],
        response: AssistantResponse,
    ) -> None:
        """Append the assistant's response as a new Message at the end.

        OpenAI-shaped default: ``role="assistant"`` with ``content`` plus any
        ``tool_calls``. Most providers that speak OpenAI's wire format inherit
        this unchanged.

        Override when the provider needs to round-trip additional per-turn
        state (e.g. Anthropic preserves raw ``content_blocks`` with thinking
        signatures via ``provider_extras``; Gemini remaps ``role`` to
        ``"model"`` at wire-serialization time but the abstract role stays
        ``"assistant"``).
        """
        messages.append(
            Message(
                role="assistant",
                content=response.content,
                tool_calls=list(response.tool_calls),
                provider_extras=dict(response.provider_extras),
            )
        )

    def append_tool_results_turn(
        self,
        messages: list[Message],
        results: list[ToolResultRef],
    ) -> None:
        """Append tool result(s) as new Message(s) at the end.

        OpenAI-shaped default: one ``role="tool"`` Message per result. Each
        carries a single ``ToolResultRef`` whose ``call_id`` matches the
        earlier ``ToolCallRef.id`` from the assistant turn — the provider's
        wire serializer uses it to set ``tool_call_id`` in the OpenAI-format
        message.

        Override when the provider bundles results into one turn (Anthropic
        emits a single ``role="user"`` message containing N ``tool_result``
        content blocks; Gemini bundles them as ``functionResponse`` parts in
        a single user turn).
        """
        for r in results:
            messages.append(
                Message(
                    role="tool",
                    content=r.result,
                    tool_results=[r],
                )
            )

    # ------------------------------------------------------------------
    # Cache hints (Phase B — 2026-04-21).
    #
    # Called once per turn by ``PredaCoreCore`` before ``chat()``. Lets the
    # provider annotate messages/tools with its own cache markers, manage
    # server-side cached content (Gemini's ``cachedContent`` — Phase C),
    # or otherwise shape the prompt for cheaper/faster inference.
    #
    # Default: no-op. Providers with no explicit cache API (OpenAI —
    # automatic server-side prefix caching) don't need to override.
    #
    # Anthropic relies on its wire serializer's automatic ``cache_control``
    # placement (top-level ephemeral + first system block), so it doesn't
    # override this either. Anthropic COULD override here if we later want
    # caller-level control (e.g. "don't cache this turn's system prompt").
    # ------------------------------------------------------------------

    def apply_cache_hints(
        self,
        messages: list[Any],
        tools: list[Any] | None = None,
    ) -> None:
        """Annotate ``messages`` / ``tools`` in place with provider-specific
        cache markers. Default: no-op.

        Signature is deliberately loose (``list[Any]``) because callers may
        pass either typed ``Message`` objects or legacy dict-shaped messages
        during the Phase A→B migration window. Providers that need typed
        access should convert internally via ``types.message_from_dict``.

        Exceptions raised by a provider's implementation are caught by core
        and logged — cache placement failures MUST NOT block inference.
        """
        del messages, tools  # default is no-op; silence "unused" linters
        return None

    # ------------------------------------------------------------------
    # Wire-format invariant repair (Phase 1.5 — 2026-05-02).
    #
    # Mirrors Anthropic's ``message_validator.repair_tool_flow``: each
    # provider scans the message list before serialization and auto-fixes
    # invariant violations (orphan tool_use → synthetic tool_result, etc.)
    # so a malformed conversation never reaches the wire. The default is
    # a no-op; Anthropic / OpenAI / Gemini override to call their
    # respective validator.
    #
    # Env toggle ``PREDACORE_REPAIR_TOOL_FLOW``:
    #   ``repair`` (default) — fix violations in-place, log a warning per fix
    #   ``strict``           — raise on any violation (CI / debug)
    #   ``off``              — skip validation entirely (legacy / debug)
    # ------------------------------------------------------------------

    def repair_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate and auto-repair tool-flow invariants. Default: pass-through.

        Subclasses override to call their provider-specific validator. The
        signature accepts dict-shaped messages (post-serialize) so each
        provider can validate against its own wire format.
        """
        return messages
