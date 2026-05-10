"""Trust-level → runner routing.

Flame skills (peer-published code from `_vendor/common/skill_collective`)
carry a TrustLevel from skill_genome:
  - UNTRUSTED  — just arrived; never run
  - SANDBOXED  — sandbox trial; isolate via DAFRunner + Docker sandbox
  - LIMITED    — passed sandbox; still untrusted enough for DAF isolation
  - TRUSTED    — fully trusted; in-process is acceptable

This module provides the `runner_for_trust_level` helper so any caller
(marketplace handler, MCP tool wrapper, OpenClaw delegation) can pick
the right runner without duplicating the rule.

Design choice: anything below TRUSTED gets DAFRunner. The 1-2s cold start
is cheap insurance vs. running peer-published code in your daemon's
event loop. Once Phase 14 lands a pre-warmed pool, even UNTRUSTED-but-
needed-fast cases get sub-100ms start.
"""
from __future__ import annotations

import logging
from typing import Any

from .runners import DAFRunner, InProcessRunner, Runner

logger = logging.getLogger(__name__)


def runner_for_trust_level(level: Any) -> Runner:
    """Return the right Runner for a given TrustLevel.

    Accepts either a `TrustLevel` enum value or a string. Anything below
    TRUSTED gets DAFRunner. Unknown levels default to DAFRunner (fail-safe).
    """
    name = _normalize(level)
    if name == "trusted":
        return InProcessRunner()
    # untrusted, sandboxed, limited → DAF for process isolation.
    return DAFRunner()


def is_high_trust(level: Any) -> bool:
    """True only for TRUSTED — any other state should be isolated."""
    return _normalize(level) == "trusted"


def needs_isolation(level: Any) -> bool:
    """True for any non-TRUSTED level. Inverse of is_high_trust."""
    return not is_high_trust(level)


def _normalize(level: Any) -> str:
    if level is None:
        return "untrusted"
    # TrustLevel enum has .value; strings just lower-case
    if hasattr(level, "value"):
        return str(level.value).lower()
    return str(level).strip().lower()


__all__ = [
    "is_high_trust",
    "needs_isolation",
    "runner_for_trust_level",
]
