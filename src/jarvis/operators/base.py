"""
JARVIS Operator Base — Abstract interface for all device operators.

Every operator (macOS desktop, Android, future Linux/Windows) implements
this interface so the tool layer can dispatch uniformly without caring
which platform is underneath.

Provides:
  - Uniform execute(action, params) → dict contract
  - Shared macro runner with abort support
  - Health check interface
  - Operator capability discovery
"""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────────


class OperatorPlatform(str, Enum):
    """Supported operator platforms."""
    MACOS = "macos"
    ANDROID = "android"
    LINUX = "linux"
    WINDOWS = "windows"


class ActionCategory(str, Enum):
    """Categories of operator actions."""
    APP_CONTROL = "app_control"
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    SCREENSHOT = "screenshot"
    CLIPBOARD = "clipboard"
    WINDOW = "window"
    ACCESSIBILITY = "accessibility"
    MACRO = "macro"
    SMART_INPUT = "smart_input"
    DEVICE = "device"
    FILE_TRANSFER = "file_transfer"


# ── Error Types ──────────────────────────────────────────────────────


class OperatorError(RuntimeError):
    """Base error for all operator failures."""

    def __init__(
        self,
        message: str,
        *,
        action: str = "",
        recoverable: bool = True,
        suggestion: str = "",
    ):
        self.action = action
        self.recoverable = recoverable
        self.suggestion = suggestion
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Serialize error to dictionary."""
        return {
            "error": str(self),
            "action": self.action,
            "recoverable": self.recoverable,
            "suggestion": self.suggestion,
        }


# ── Macro Abort Token ────────────────────────────────────────────────


class MacroAbortToken:
    """Thread-safe abort token for cancelling running macros.

    Usage:
        token = MacroAbortToken()
        # Pass to macro runner
        operator.execute_macro(steps, abort_token=token)
        # From another thread:
        token.abort("User requested cancellation")
    """

    def __init__(self) -> None:
        self._aborted = False
        self._reason: str = ""
        self._lock = threading.Lock()

    def abort(self, reason: str = "Aborted") -> None:
        """Signal abort with an optional reason message."""
        with self._lock:
            self._aborted = True
            self._reason = reason

    @property
    def is_aborted(self) -> bool:
        """Return True if abort has been signalled."""
        with self._lock:
            return self._aborted

    @property
    def reason(self) -> str:
        """Return the abort reason, or empty string if not aborted."""
        with self._lock:
            return self._reason


# ── Base Operator ────────────────────────────────────────────────────


class BaseOperator(ABC):
    """Abstract base for all device operators.

    Subclasses must implement:
      - execute(action, params) → dict
      - available → bool
      - health_check() → dict
      - platform → OperatorPlatform
      - supported_actions → set[str]
    """

    def __init__(self, log: logging.Logger | None = None, operators_config: Any = None):
        self._log = log or logger
        self._ops_cfg = operators_config
        # Macro state
        self._macro_depth: int = 0
        self._macro_lock = threading.Lock()
        # Per-action telemetry
        self._action_stats: dict[str, dict[str, Any]] = {}

    # ── Abstract Interface ───────────────────────────────────────

    @property
    @abstractmethod
    def platform(self) -> OperatorPlatform:
        """Return the platform this operator controls."""
        ...

    @property
    @abstractmethod
    def available(self) -> bool:
        """Check if this operator is available on the current system."""
        ...

    @property
    @abstractmethod
    def supported_actions(self) -> set[str]:
        """Return the set of action names this operator supports."""
        ...

    @abstractmethod
    def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute one action and return structured result.

        Returns:
            dict with at minimum {"ok": True/False, "action": action_name}
        """
        ...

    @abstractmethod
    def health_check(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run diagnostics and return health status."""
        ...

    # ── Shared Macro Runner ──────────────────────────────────────

    @property
    def _max_macro_steps(self) -> int:
        return getattr(self._ops_cfg, "macro_max_steps", 50) if self._ops_cfg else 50

    @property
    def _max_macro_depth(self) -> int:
        return getattr(self._ops_cfg, "macro_max_depth", 3) if self._ops_cfg else 3

    def execute_macro(
        self,
        steps: list[dict[str, Any]],
        *,
        stop_on_error: bool = True,
        delay_ms: int = 0,
        abort_token: MacroAbortToken | None = None,
    ) -> dict[str, Any]:
        """Execute a multi-step macro with abort support.

        This is the shared implementation — no need to duplicate in each
        operator subclass.

        Args:
            steps: List of {"action": ..., ...params} dicts.
            stop_on_error: Halt on first failure if True.
            delay_ms: Delay between steps in milliseconds.
            abort_token: Optional token to cancel mid-execution.

        Returns:
            {"completed_steps": N, "results": [...], ...}
        """
        if not isinstance(steps, list) or not steps:
            raise OperatorError(
                "run_macro requires a non-empty steps array",
                action="run_macro",
            )
        if len(steps) > self._max_macro_steps:
            raise OperatorError(
                f"run_macro limited to {self._max_macro_steps} steps (got {len(steps)})",
                action="run_macro",
            )

        # Prevent recursive macro nesting
        did_increment = False
        try:
            with self._macro_lock:
                if self._macro_depth >= self._max_macro_depth:
                    raise OperatorError(
                        f"macro nesting too deep (max {self._max_macro_depth} levels)",
                        action="run_macro",
                    )
                self._macro_depth += 1
                did_increment = True

            delay_ms = max(0, min(delay_ms, 10000))
            results: list[dict[str, Any]] = []

            for i, step in enumerate(steps, start=1):
                # ── Check abort token ──
                if abort_token and abort_token.is_aborted:
                    return {
                        "completed_steps": i - 1,
                        "aborted": True,
                        "abort_reason": abort_token.reason,
                        "results": results,
                    }

                if not isinstance(step, dict):
                    raise OperatorError(
                        f"invalid macro step at index {i - 1}",
                        action="run_macro",
                    )
                step_action = str(step.get("action") or "").strip().lower()
                if not step_action:
                    raise OperatorError(
                        f"missing step.action at index {i - 1}",
                        action="run_macro",
                    )

                step_params = dict(step)
                step_params.pop("action", None)

                try:
                    step_result = self.execute(step_action, step_params)
                    results.append({"index": i, "result": step_result})
                except Exception as exc:
                    err = {
                        "index": i,
                        "error": str(exc),
                        "action": step_action,
                    }
                    results.append(err)
                    if stop_on_error:
                        return {
                            "completed_steps": i - 1,
                            "failed_step": i,
                            "results": results,
                        }

                if delay_ms > 0:
                    time.sleep(delay_ms / 1000.0)

            return {"completed_steps": len(steps), "results": results}
        finally:
            if did_increment:
                with self._macro_lock:
                    self._macro_depth -= 1

    # ── Telemetry ─────────────────────────────────────────────────

    def _record_telemetry(self, action: str, elapsed_ms: float, ok: bool) -> None:
        """Record per-action timing and success rate."""
        if len(self._action_stats) >= 500 and action not in self._action_stats:
            if not getattr(self, "_telemetry_cap_warned", False):
                logger.warning(
                    "operator telemetry hit 500-action cap; further new actions will not be recorded (first dropped: %s)",
                    action,
                )
                self._telemetry_cap_warned = True
            return
        stats = self._action_stats.setdefault(action, {
            "calls": 0, "successes": 0, "failures": 0,
            "total_ms": 0.0, "min_ms": float("inf"), "max_ms": 0.0,
        })
        stats["calls"] += 1
        if ok:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
        stats["total_ms"] += elapsed_ms
        stats["min_ms"] = min(stats["min_ms"], elapsed_ms)
        stats["max_ms"] = max(stats["max_ms"], elapsed_ms)

    def telemetry(self) -> dict[str, Any]:
        """Return per-action telemetry stats."""
        result = {}
        for action, stats in self._action_stats.items():
            calls = stats["calls"]
            result[action] = {
                "calls": calls,
                "success_rate": f"{stats['successes'] / calls * 100:.1f}%" if calls else "N/A",
                "avg_ms": round(stats["total_ms"] / calls, 1) if calls else 0,
                "min_ms": round(stats["min_ms"], 1) if stats["min_ms"] != float("inf") else 0,
                "max_ms": round(stats["max_ms"], 1),
            }
        return result

    # ── Capability Discovery ─────────────────────────────────────

    def supports(self, action: str) -> bool:
        """Check if this operator supports a given action."""
        return action.strip().lower() in self.supported_actions

    def capabilities(self) -> dict[str, Any]:
        """Return full capability manifest for this operator."""
        return {
            "platform": self.platform.value,
            "available": self.available,
            "actions": sorted(self.supported_actions),
            "action_count": len(self.supported_actions),
            "max_macro_steps": self._max_macro_steps,
            "max_macro_depth": self._max_macro_depth,
        }
