"""
Tool-flow validator for Gemini's ``contents[]`` wire format.

Gemini's API shape (post-``_build_payload``):
  contents[]: [
    {"role": "user"|"model", "parts": [
      {"text": ...} |
      {"functionCall": {"name": ..., "args": {...}}, "thoughtSignature": "..."} |
      {"functionResponse": {"name": ..., "response": {...}}}
    ]}
  ]

Invariants enforced:

  1. Every ``role:"model"`` turn that contains ``functionCall`` parts must
     be followed by a ``role:"user"`` turn whose parts include a
     ``functionResponse`` for *every* functionCall, matched by **name +
     position** (Gemini lacks IDs natively; predacore's
     ``ToolCallRef.id`` is internal, not on the wire).

  2. ``functionResponse`` parts that aren't matched by a preceding
     ``functionCall`` (within the immediately-preceding ``model`` turn)
     are dropped.

  3. ``thoughtSignature`` preservation — the validator never reconstructs
     ``functionCall`` parts from scratch during repair, so any signature
     already present is preserved verbatim. Per Google docs, on parallel
     calls only the FIRST functionCall in the turn carries the signature
     — that placement is also preserved.

Honors ``PREDACORE_REPAIR_TOOL_FLOW`` (same env toggle as the other
validators).
"""
from __future__ import annotations

import logging
from typing import Any

from .message_validator import ToolFlowInvariantError, _repair_mode

logger = logging.getLogger(__name__)


def _is_function_call_part(part: Any) -> bool:
    return isinstance(part, dict) and "functionCall" in part


def _is_function_response_part(part: Any) -> bool:
    return isinstance(part, dict) and "functionResponse" in part


def _function_call_names(turn: dict[str, Any]) -> list[str]:
    """Return ordered list of functionCall names from a model turn."""
    parts = turn.get("parts") or []
    return [
        str((p.get("functionCall") or {}).get("name", ""))
        for p in parts
        if _is_function_call_part(p)
    ]


def _function_response_names(turn: dict[str, Any]) -> list[str]:
    """Return ordered list of functionResponse names from a user turn."""
    parts = turn.get("parts") or []
    return [
        str((p.get("functionResponse") or {}).get("name", ""))
        for p in parts
        if _is_function_response_part(p)
    ]


def _has_thought_signature(turn: dict[str, Any]) -> bool:
    """Return True if any functionCall part has a thoughtSignature."""
    parts = turn.get("parts") or []
    return any(
        _is_function_call_part(p) and "thoughtSignature" in p
        for p in parts
    )


def validate_wire(contents: list[dict[str, Any]]) -> list[str]:
    """Return a list of human-readable invariant violations. Empty = valid."""
    issues: list[str] = []
    for i, turn in enumerate(contents):
        role = turn.get("role")
        if role == "model":
            call_names = _function_call_names(turn)
            if not call_names:
                continue
            nxt = contents[i + 1] if i + 1 < len(contents) else None
            if nxt is None or nxt.get("role") != "user":
                issues.append(
                    f"model turn {i}: {len(call_names)} functionCall part(s) "
                    "not followed by user turn with functionResponse"
                )
                continue
            response_names = _function_response_names(nxt)
            # Match by position (name at index k must match) — Gemini's
            # default linkage. If the same tool is called twice, position
            # disambiguates.
            for k, name in enumerate(call_names):
                if k >= len(response_names) or response_names[k] != name:
                    issues.append(
                        f"model turn {i}: functionCall #{k} (name={name!r}) "
                        f"not matched by functionResponse in user turn {i+1}"
                    )
        elif role == "user":
            response_names = _function_response_names(turn)
            if not response_names:
                continue
            prev = contents[i - 1] if i > 0 else None
            if prev is None or prev.get("role") != "model":
                issues.append(
                    f"user turn {i}: orphan functionResponse part(s) "
                    "(no preceding model turn)"
                )
                continue
            call_names = _function_call_names(prev)
            for k, name in enumerate(response_names):
                if k >= len(call_names) or call_names[k] != name:
                    # Only flag responses past the matched-call zone OR with
                    # mismatched name. (Position k is matched against position
                    # k of the call list.)
                    issues.append(
                        f"user turn {i}: functionResponse #{k} (name={name!r}) "
                        f"has no matching functionCall in model turn {i-1}"
                    )
    return issues


def repair_wire(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Auto-repair Gemini wire-format invariant violations.

    Repairs:
      * Model turn with functionCall parts and no following functionResponse
        user turn → inject a synthetic user turn with placeholder
        functionResponse parts (one per functionCall, in order, by name).
      * Partially-matched functionCall list → append missing
        functionResponse parts to the existing user turn (in name+position
        order).
      * Orphan functionResponse parts (no preceding model turn with the
        matching functionCall) → drop the part. If the turn becomes empty,
        replace its parts with a single ``{"text": ""}`` placeholder so
        Gemini doesn't 400 on an empty parts list.

    Never touches functionCall parts (preserves thoughtSignature).
    Honors ``PREDACORE_REPAIR_TOOL_FLOW``.
    """
    mode = _repair_mode()
    if mode == "off":
        return contents
    if mode == "strict":
        issues = validate_wire(contents)
        if issues:
            raise ToolFlowInvariantError(issues)
        return contents

    out: list[dict[str, Any]] = []
    n = len(contents)
    i = 0
    while i < n:
        turn = dict(contents[i])  # shallow copy
        role = turn.get("role")

        if role == "model" and _function_call_names(turn):
            call_names = _function_call_names(turn)
            out.append(turn)

            nxt = contents[i + 1] if i + 1 < n else None
            if nxt is None or nxt.get("role") != "user":
                # Inject a synthetic user turn with placeholder responses.
                synthetic_parts = [
                    {
                        "functionResponse": {
                            "name": name,
                            "response": {
                                "result": "[Tool execution lost — synthetic placeholder]",
                                "is_error": True,
                            },
                        }
                    }
                    for name in call_names
                ]
                out.append({"role": "user", "parts": synthetic_parts})
                logger.warning(
                    "gemini_validator: injected synthetic user turn with "
                    "%d functionResponse part(s) after model turn %d",
                    len(call_names), i,
                )
                i += 1
                continue

            # Following turn IS user — patch it to cover all functionCall names.
            nxt_copy = dict(nxt)
            existing_parts = list(nxt_copy.get("parts") or [])
            # Index existing functionResponse parts by position
            existing_resp_names = [
                str((p.get("functionResponse") or {}).get("name", ""))
                for p in existing_parts
                if _is_function_response_part(p)
            ]
            non_resp_parts = [
                p for p in existing_parts if not _is_function_response_part(p)
            ]
            # Build the new ordered functionResponse list — match by position
            # against call_names, fill missing with placeholders.
            new_resp_parts: list[dict[str, Any]] = []
            for k, name in enumerate(call_names):
                if k < len(existing_resp_names) and existing_resp_names[k] == name:
                    # Find the original functionResponse part by index
                    orig = [
                        p for p in existing_parts if _is_function_response_part(p)
                    ][k]
                    new_resp_parts.append(orig)
                else:
                    new_resp_parts.append(
                        {
                            "functionResponse": {
                                "name": name,
                                "response": {
                                    "result": "[Tool result lost — synthetic placeholder]",
                                    "is_error": True,
                                },
                            }
                        }
                    )
                    logger.warning(
                        "gemini_validator: injected synthetic functionResponse "
                        "for name=%r at user turn %d (position %d)",
                        name, i + 1, k,
                    )
            nxt_copy["parts"] = non_resp_parts + new_resp_parts
            out.append(nxt_copy)
            i += 2
            continue

        if role == "user" and _function_response_names(turn):
            # User turn with functionResponse parts — verify each matches a
            # preceding model turn's functionCall (by position+name).
            prev = out[-1] if out else None
            prev_call_names: list[str] = []
            if prev and prev.get("role") == "model":
                prev_call_names = _function_call_names(prev)

            kept_parts: list[dict[str, Any]] = []
            resp_seen = 0
            for p in turn.get("parts") or []:
                if _is_function_response_part(p):
                    name = str((p.get("functionResponse") or {}).get("name", ""))
                    if (
                        resp_seen < len(prev_call_names)
                        and prev_call_names[resp_seen] == name
                    ):
                        kept_parts.append(p)
                    else:
                        logger.warning(
                            "gemini_validator: dropping orphan functionResponse "
                            "(name=%r) at user turn %d position %d",
                            name, i, resp_seen,
                        )
                    resp_seen += 1
                else:
                    kept_parts.append(p)

            if not kept_parts:
                # Empty turn — Gemini 400s on parts:[]. Replace with placeholder.
                kept_parts = [{"text": ""}]
            turn["parts"] = kept_parts
            out.append(turn)
            i += 1
            continue

        out.append(turn)
        i += 1

    return out


__all__ = ["validate_wire", "repair_wire"]
