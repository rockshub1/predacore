"""
Tool-flow validator for OpenAI Chat Completions wire format.

Mirrors :mod:`message_validator` (Anthropic-shaped) but operates on
OpenAI's nested ``tool_calls`` / ``role:"tool"`` shape. Runs after
``OpenAIProvider._serialize_messages_for_wire`` and before the request
hits the wire.

Invariants enforced:

  1. Every assistant turn with ``tool_calls`` must be followed by N
     ``role:"tool"`` messages whose ``tool_call_id`` covers every id in
     ``tool_calls[].id``. Missing → inject synthetic placeholder.
  2. ``role:"tool"`` messages whose ``tool_call_id`` does not match any
     preceding assistant ``tool_calls[].id`` are dropped.
  3. Empty ``tool_calls: []`` arrays on assistant turns are stripped
     (OpenAI 400s on empty arrays).

Honors ``PREDACORE_REPAIR_TOOL_FLOW`` (same env toggle as
:mod:`message_validator`):
    ``repair`` (default) — auto-fix violations + log warning per fix
    ``strict``           — raise :class:`ToolFlowInvariantError` on any violation
    ``off``              — skip validation entirely
"""
from __future__ import annotations

import logging
from typing import Any

from .message_validator import ToolFlowInvariantError, _repair_mode

logger = logging.getLogger(__name__)


def _is_assistant_with_tool_calls(msg: dict[str, Any]) -> bool:
    if msg.get("role") != "assistant":
        return False
    tcs = msg.get("tool_calls")
    return isinstance(tcs, list) and len(tcs) > 0


def _is_tool_message(msg: dict[str, Any]) -> bool:
    return msg.get("role") == "tool"


def _tool_call_ids(msg: dict[str, Any]) -> list[str]:
    """Return ordered list of ``tool_calls[].id`` from an assistant turn."""
    return [str(tc.get("id", "")) for tc in (msg.get("tool_calls") or [])]


def validate_wire(messages: list[dict[str, Any]]) -> list[str]:
    """Return a list of human-readable invariant violations. Empty = valid."""
    issues: list[str] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]

        # Invariant 3: empty tool_calls array on assistant
        if (
            msg.get("role") == "assistant"
            and isinstance(msg.get("tool_calls"), list)
            and len(msg["tool_calls"]) == 0
        ):
            issues.append(f"assistant turn {i}: empty tool_calls array")

        # Invariant 1: assistant tool_calls must be followed by matching tool messages
        if _is_assistant_with_tool_calls(msg):
            call_ids = [tcid for tcid in _tool_call_ids(msg) if tcid]
            if not call_ids:
                # tool_calls present but none have ids — malformed
                issues.append(f"assistant turn {i}: tool_calls without ids")
            else:
                # Walk forward consuming matching tool messages
                seen_ids: set[str] = set()
                j = i + 1
                while j < n and _is_tool_message(messages[j]):
                    tcid = str(messages[j].get("tool_call_id", ""))
                    if tcid:
                        seen_ids.add(tcid)
                    j += 1
                missing = [cid for cid in call_ids if cid not in seen_ids]
                if missing:
                    issues.append(
                        f"assistant turn {i}: missing tool_result for ids {missing}"
                    )

        # Invariant 2: tool message must reference a preceding assistant tool_call
        if _is_tool_message(msg):
            tcid = str(msg.get("tool_call_id", ""))
            # Walk backward to find the most recent assistant turn with tool_calls
            preceding_ids: set[str] = set()
            for k in range(i - 1, -1, -1):
                prev = messages[k]
                if _is_assistant_with_tool_calls(prev):
                    preceding_ids = {x for x in _tool_call_ids(prev) if x}
                    break
                if _is_tool_message(prev):
                    continue
                # Hit a non-tool, non-assistant-with-calls turn — stop
                break
            if not tcid or tcid not in preceding_ids:
                issues.append(
                    f"tool turn {i}: orphaned tool_call_id={tcid!r}"
                )

        i += 1

    return issues


def repair_wire(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return a new message list with invariant violations auto-repaired.

    Repairs:
      * Empty ``tool_calls: []`` array on assistant → key removed.
      * Assistant ``tool_calls`` not followed by matching tool messages →
        synthesize placeholder ``role:"tool"`` for each missing id.
      * Orphaned ``role:"tool"`` (no matching preceding assistant call) →
        drop the message.

    Honors ``PREDACORE_REPAIR_TOOL_FLOW`` (same env var as
    :mod:`message_validator`).
    """
    mode = _repair_mode()
    if mode == "off":
        return messages
    if mode == "strict":
        issues = validate_wire(messages)
        if issues:
            raise ToolFlowInvariantError(issues)
        return messages

    # Pass 1: drop orphaned tool messages (no preceding assistant tool_calls
    # with matching id).
    filtered: list[dict[str, Any]] = []
    last_tool_call_ids: set[str] = set()
    for i, msg in enumerate(messages):
        if _is_assistant_with_tool_calls(msg):
            last_tool_call_ids = {tcid for tcid in _tool_call_ids(msg) if tcid}
            filtered.append(msg)
            continue
        if _is_tool_message(msg):
            tcid = str(msg.get("tool_call_id", ""))
            if not tcid or tcid not in last_tool_call_ids:
                logger.warning(
                    "openai_validator: dropping orphaned tool message at index %d "
                    "(tool_call_id=%r)",
                    i, tcid,
                )
                continue
            filtered.append(msg)
            continue
        # Any non-tool, non-assistant-with-calls message resets the window
        if not _is_tool_message(msg):
            last_tool_call_ids = set()
        filtered.append(msg)

    # Pass 2: strip empty tool_calls arrays + inject missing placeholders
    out: list[dict[str, Any]] = []
    i = 0
    n = len(filtered)
    while i < n:
        msg = dict(filtered[i])  # shallow copy so we don't mutate the input

        # Strip empty tool_calls array
        if (
            msg.get("role") == "assistant"
            and isinstance(msg.get("tool_calls"), list)
            and len(msg["tool_calls"]) == 0
        ):
            msg.pop("tool_calls", None)
            logger.warning(
                "openai_validator: stripped empty tool_calls array on assistant "
                "turn %d",
                i,
            )

        out.append(msg)

        if _is_assistant_with_tool_calls(msg):
            call_ids = [tcid for tcid in _tool_call_ids(msg) if tcid]
            # Walk forward through following tool messages; collect seen ids
            seen_ids: set[str] = set()
            j = i + 1
            while j < n and _is_tool_message(filtered[j]):
                tcid = str(filtered[j].get("tool_call_id", ""))
                if tcid:
                    seen_ids.add(tcid)
                out.append(filtered[j])
                j += 1
            # Inject placeholders for missing ids — preserve original order
            missing = [cid for cid in call_ids if cid not in seen_ids]
            for cid in missing:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": cid,
                        "content": "[Tool result lost — synthetic placeholder]",
                    }
                )
            if missing:
                logger.warning(
                    "openai_validator: injected %d synthetic tool message(s) "
                    "after assistant turn %d (missing ids: %s)",
                    len(missing), i, missing,
                )
            i = j
            continue

        i += 1

    return out


__all__ = ["validate_wire", "repair_wire"]
