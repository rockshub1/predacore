"""
Typed message and response models for LLM providers.

These are provider-neutral types. Each provider adapter translates between
these types and its own wire format:

  Anthropic  ─ content_blocks with tool_use / tool_result blocks, thinking sigs
  OpenAI     ─ message.tool_calls + role="tool" responses with tool_call_id
  Gemini     ─ parts[] with functionCall / functionResponse + role="model"

Part of the Phase A refactor (2026-04-21): tool-turn serialization moves out
of ``core.py`` and into each provider adapter. The typed models below are the
handoff surface between core (agent-layer) and providers (wire-layer).

See also:
  - ``base.LLMProvider.append_assistant_turn`` / ``append_tool_results_turn``
    — the methods providers implement to serialize into their own format.
  - ``types.message_from_dict`` / ``message_to_dict`` — compatibility shims
    used during incremental migration while some call-sites still produce dicts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolDefinition:
    """Abstract tool schema shipped to the model.

    Each provider translates this into its wire format:
      - Anthropic: ``{"name", "description", "input_schema"}``
      - OpenAI: ``{"type": "function", "function": {"name", "description", "parameters"}}``
      - Gemini: ``{"functionDeclarations": [{"name", "description", "parameters"}]}``
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSONSchema for tool arguments


@dataclass
class ToolCallRef:
    """A single tool call the model wants us to execute.

    ``id`` is the provider-assigned identifier (round-trippable). For providers
    that don't supply one natively (Gemini), the provider adapter synthesizes a
    stable value so the downstream ``ToolResultRef.call_id`` can match.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResultRef:
    """A single tool result to hand back to the model.

    ``call_id`` must match a preceding ``ToolCallRef.id``. Providers that link
    call→result by id (Anthropic, OpenAI) require this; providers that link
    by position (Gemini) tolerate any value but still set it for consistency.
    """

    call_id: str
    name: str
    result: str
    is_error: bool = False


@dataclass
class Message:
    """One turn in the conversation, provider-neutral.

    Role semantics:
      ``system``    — system prompt
      ``user``      — user input, OR tool results for providers that colocate
                      results in user turns (Anthropic, Gemini)
      ``assistant`` — model response (may include ``tool_calls``)
      ``tool``      — tool result for providers with a dedicated role (OpenAI)

    ``provider_extras`` is an opaque per-provider bag — providers use it to
    round-trip state that doesn't fit the common fields (e.g. Anthropic stashes
    raw ``content_blocks`` here to preserve thinking-block signatures across
    turns; DeepSeek could stash reasoning-token metadata; etc.). Core code
    never reads from ``provider_extras`` — it's purely provider↔provider.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCallRef] = field(default_factory=list)
    tool_results: list[ToolResultRef] = field(default_factory=list)
    provider_extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssistantResponse:
    """Standardized LLM chat response.

    All providers return this shape from their ``chat()`` method. Tool calls
    are normalized from whatever provider-specific format (OpenAI
    ``tool_calls``, Anthropic ``tool_use`` blocks, Gemini ``functionCall``
    parts) into ``ToolCallRef``.

    ``provider_extras`` carries raw provider-specific state a provider may
    need to round-trip back on the next turn (e.g. Anthropic thinking blocks
    with signatures). Core never reads it.
    """

    content: str = ""
    tool_calls: list[ToolCallRef] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    provider_extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Backward-compat helpers used during incremental migration.
#
# Before the full Phase A migration lands, some callers still produce/consume
# dict-shaped messages. These converters let typed Messages flow through the
# dict layer without losing information and vice versa.
# ---------------------------------------------------------------------------


def message_from_dict(d: dict[str, Any]) -> Message:
    """Convert a legacy dict-shaped message into a typed Message.

    Unknown keys land in ``provider_extras`` so nothing is silently dropped.
    """
    known = {"role", "content", "tool_calls", "tool_results"}
    extras = {k: v for k, v in d.items() if k not in known}
    role_value = d.get("role", "user")
    return Message(
        role=role_value,  # type: ignore[arg-type]
        content=(d.get("content") or "") if isinstance(d.get("content"), str) else "",
        tool_calls=[_tool_call_from_dict(tc) for tc in (d.get("tool_calls") or [])],
        tool_results=[_tool_result_from_dict(tr) for tr in (d.get("tool_results") or [])],
        provider_extras=extras,
    )


def _tool_call_from_dict(d: dict[str, Any]) -> ToolCallRef:
    return ToolCallRef(
        id=str(d.get("id", "")),
        name=str(d.get("name", "")),
        arguments=dict(d.get("arguments") or d.get("input") or {}),
    )


def _tool_result_from_dict(d: dict[str, Any]) -> ToolResultRef:
    return ToolResultRef(
        call_id=str(d.get("call_id") or d.get("tool_use_id", "")),
        name=str(d.get("name", "")),
        result=str(d.get("result") or d.get("content", "")),
        is_error=bool(d.get("is_error", False)),
    )


def message_to_dict(m: Message) -> dict[str, Any]:
    """Convert a typed Message back into a plain dict for legacy callers.

    ``provider_extras`` keys are merged last, so providers that stash custom
    shape (Anthropic's ``content_blocks``) see it preserved.
    """
    d: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in m.tool_calls
        ]
    if m.tool_results:
        d["tool_results"] = [
            {
                "call_id": tr.call_id,
                "name": tr.name,
                "result": tr.result,
                "is_error": tr.is_error,
            }
            for tr in m.tool_results
        ]
    for k, v in m.provider_extras.items():
        d[k] = v
    return d


__all__ = [
    "Role",
    "ToolDefinition",
    "ToolCallRef",
    "ToolResultRef",
    "Message",
    "AssistantResponse",
    "message_from_dict",
    "message_to_dict",
]
