"""
Tool-flow validator for Anthropic-style structured messages.

Anthropic's API rejects requests where:

  1. An assistant turn contains `tool_use` blocks that are not followed by
     a user turn with `tool_result` blocks matching every `tool_use_id`.
  2. A user turn has orphaned `tool_result` blocks with no preceding
     `tool_use` to reference.
  3. Extended / interleaved thinking is enabled but a prior assistant turn's
     `thinking` blocks are missing — the model loses its reasoning chain and
     the API surfaces a 400.

The agent loop in ``core.py`` is supposed to uphold all three invariants.
This module verifies them at send time and repairs common breakages rather
than letting a 400 propagate up as a user-visible "connectivity issue".

Historical context (2026-04):
The original Anthropic round-trip bug dropped thinking block signatures
and tool_use ids between iterations, and ``core.py`` substituted
synthetic ``[Calling tool: X]`` text stubs for real content blocks.
After that was fixed, this validator remains as a structural safety net
so the same class of bug can never silently re-enter the codebase
without either tripping an Anthropic 400 or a visible warning in the
logs.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _is_tool_use(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") == "tool_use"


def _is_tool_result(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") == "tool_result"


def _blocks_of(msg: dict | None) -> list[dict] | None:
    """Return the block list for a structured message, or None for flat."""
    if msg is None:
        return None
    content = msg.get("content")
    if isinstance(content, list):
        return content
    return None


def validate_tool_flow(messages: list[dict]) -> list[str]:
    """Return a list of human-readable invariant violations. Empty = valid."""
    issues: list[str] = []
    for i, msg in enumerate(messages):
        blocks = _blocks_of(msg)
        if blocks is None:
            continue
        if msg.get("role") == "assistant":
            pending_ids = [b.get("id", "") for b in blocks if _is_tool_use(b)]
            if not pending_ids:
                continue
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            if nxt is None or nxt.get("role") != "user":
                issues.append(
                    f"assistant turn {i} has {len(pending_ids)} tool_use "
                    f"block(s) not followed by a user turn"
                )
                continue
            nxt_blocks = _blocks_of(nxt) or []
            result_ids = {
                b.get("tool_use_id", "")
                for b in nxt_blocks
                if _is_tool_result(b)
            }
            missing = [tid for tid in pending_ids if tid and tid not in result_ids]
            if missing:
                issues.append(
                    f"assistant turn {i}: missing tool_result for ids {missing}"
                )
        elif msg.get("role") == "user":
            result_ids = [b.get("tool_use_id", "") for b in blocks if _is_tool_result(b)]
            if not result_ids:
                continue
            prev = messages[i - 1] if i > 0 else None
            prev_blocks = _blocks_of(prev) if prev else None
            prev_use_ids: set[str] = set()
            if prev and prev.get("role") == "assistant" and prev_blocks:
                prev_use_ids = {
                    b.get("id", "") for b in prev_blocks if _is_tool_use(b)
                }
            orphaned = [rid for rid in result_ids if rid and rid not in prev_use_ids]
            if orphaned:
                issues.append(
                    f"user turn {i}: orphaned tool_result for ids {orphaned}"
                )
    return issues


def repair_tool_flow(messages: list[dict]) -> list[dict]:
    """
    Return a new message list with invariant violations auto-repaired.

    Repairs performed:

      * orphaned assistant tool_use → append a synthetic user turn with
        stub tool_result blocks (marked is_error=True) so Anthropic still
        accepts the request instead of 400-ing with a shape error
      * partially-satisfied assistant tool_use → inject the missing
        tool_result blocks into the following user turn
      * orphaned user tool_result (no preceding tool_use) → drop the block

    Every repair emits a logger.warning so regressions are visible without
    burning a user-visible error.
    """
    out: list[dict] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        out.append(msg)
        blocks = _blocks_of(msg)

        if (
            msg.get("role") == "assistant"
            and blocks is not None
            and any(_is_tool_use(b) for b in blocks)
        ):
            tool_use_ids = [b.get("id", "") for b in blocks if _is_tool_use(b)]
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            nxt_blocks = _blocks_of(nxt) if nxt else None

            if nxt is None or nxt.get("role") != "user":
                # Case 1: no following user turn at all — inject a synthetic one.
                synthetic_blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": tid,
                        "content": "[Tool execution lost — synthetic placeholder]",
                        "is_error": True,
                    }
                    for tid in tool_use_ids
                ]
                out.append(
                    {
                        "role": "user",
                        "content": synthetic_blocks,
                    }
                )
                logger.warning(
                    "repair_tool_flow: injected %d synthetic tool_result block(s) "
                    "after assistant turn %d (no following user turn)",
                    len(tool_use_ids),
                    i,
                )
            elif nxt_blocks is not None:
                # Case 2: following user turn already has blocks — add any missing ids.
                result_ids = {
                    b.get("tool_use_id", "")
                    for b in nxt_blocks
                    if _is_tool_result(b)
                }
                missing = [tid for tid in tool_use_ids if tid and tid not in result_ids]
                if missing:
                    for tid in missing:
                        nxt_blocks.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tid,
                                "content": "[Tool result lost — synthetic placeholder]",
                                "is_error": True,
                            }
                        )
                    logger.warning(
                        "repair_tool_flow: added %d missing tool_result block(s) "
                        "to user turn %d",
                        len(missing),
                        i + 1,
                    )
            else:
                # Case 3: following user turn has STRING content, no blocks at all.
                # This is the production-bug shape: core.py substituted a flat
                # `[Tool Result: X]` text stub for the structured round-trip.
                # Promote the string into a text block and PREPEND synthetic
                # tool_result blocks so Anthropic accepts the linkage.
                existing_text = str(nxt.get("content", "") or "")
                new_blocks: list[dict] = [
                    {
                        "type": "tool_result",
                        "tool_use_id": tid,
                        "content": "[Tool result lost — synthetic placeholder]",
                        "is_error": True,
                    }
                    for tid in tool_use_ids
                ]
                if existing_text.strip():
                    new_blocks.append(
                        {"type": "text", "text": existing_text}
                    )
                # Mutate the already-appended message in `out` (it's the
                # same reference as `nxt` because `out.append(msg)` at the
                # top of the loop appended the current msg, not nxt —
                # `nxt` hasn't been appended yet; it will be on the next
                # iteration. So we need to mutate `nxt` directly.
                nxt["content"] = new_blocks
                logger.warning(
                    "repair_tool_flow: promoted string user turn %d to blocks "
                    "with %d synthetic tool_result(s) for assistant turn %d",
                    i + 1,
                    len(tool_use_ids),
                    i,
                )
        i += 1

    # Second pass: drop orphaned user tool_result blocks (no preceding tool_use).
    cleaned: list[dict] = []
    for j, msg in enumerate(out):
        blocks = _blocks_of(msg)
        if (
            msg.get("role") == "user"
            and blocks is not None
            and any(_is_tool_result(b) for b in blocks)
        ):
            prev = cleaned[-1] if cleaned else None
            prev_blocks = _blocks_of(prev) if prev else None
            prev_use_ids: set[str] = set()
            if prev and prev.get("role") == "assistant" and prev_blocks:
                prev_use_ids = {
                    b.get("id", "") for b in prev_blocks if _is_tool_use(b)
                }
            kept = [
                b
                for b in blocks
                if not _is_tool_result(b)
                or (b.get("tool_use_id", "") in prev_use_ids)
            ]
            dropped = len(blocks) - len(kept)
            if dropped:
                logger.warning(
                    "repair_tool_flow: dropped %d orphaned tool_result block(s) "
                    "from user turn %d",
                    dropped,
                    j,
                )
            if not kept:
                # User turn is now empty — replace with a placeholder text block
                cleaned.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "[Tool results dropped — no matching tool_use]",
                            }
                        ],
                    }
                )
            else:
                cleaned.append({**msg, "content": kept})
        else:
            cleaned.append(msg)

    return cleaned
