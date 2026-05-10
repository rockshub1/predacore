"""OrchestrationBudget — hard ceilings for multi-agent runs.

Anthropic measured 15× chat tokens for orchestrator-worker on average,
30-50× at p99. Without a hard ceiling, a misclassified "simple" query
that goes orchestrator-worker burns $5 instead of $0.30. Hard abort
on budget exhaustion is non-negotiable.

The lead agent reads `remaining()` to decide whether to scope down,
and passes a per-spec slice to each subagent so subagents see their
own budget and can adapt.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .exceptions import BudgetExceededError


# Provider pricing $/1k tokens (input). Rough; over-estimate side.
# Used only for budget tracking — actual billing comes from provider.
_PRICE_PER_1K_TOKENS_INPUT = {
    "claude-opus-4-7": 15.0 / 1000,
    "claude-sonnet-4-6": 3.0 / 1000,
    "claude-haiku-4-5": 0.8 / 1000,
    "gpt-5": 5.0 / 1000,
    "gpt-5-mini": 0.5 / 1000,
    "o5": 15.0 / 1000,
    "gemini-2.5-pro": 2.5 / 1000,
    "gemini-2.5-flash": 0.5 / 1000,
}
_PRICE_PER_1K_TOKENS_OUTPUT = {
    "claude-opus-4-7": 75.0 / 1000,
    "claude-sonnet-4-6": 15.0 / 1000,
    "claude-haiku-4-5": 4.0 / 1000,
    "gpt-5": 20.0 / 1000,
    "gpt-5-mini": 2.0 / 1000,
    "o5": 60.0 / 1000,
    "gemini-2.5-pro": 10.0 / 1000,
    "gemini-2.5-flash": 1.5 / 1000,
}


def _estimate_dollars(model: str, in_tokens: int, out_tokens: int) -> float:
    """Cost estimate in USD. Unknown models default to mid-tier rates."""
    in_rate = _PRICE_PER_1K_TOKENS_INPUT.get(model, 5.0 / 1000)
    out_rate = _PRICE_PER_1K_TOKENS_OUTPUT.get(model, 15.0 / 1000)
    return (in_tokens / 1000) * in_rate + (out_tokens / 1000) * out_rate


@dataclass
class OrchestrationBudget:
    """Hard ceilings on resource consumption for one orchestration run.

    Treat all four dimensions as ANDed — first to hit zero stops the run.
    Defaults are sized for a single user question that may go up to a
    10-subagent orchestrator-worker fan-out.
    """

    # Hard ceilings
    max_total_tokens: int = 500_000          # ~500k = $50 at Opus rates
    max_total_dollars: float = 5.00          # hard $-cap
    max_wall_seconds: int = 300              # 5 minutes wall-time
    max_subagents: int = 15                  # caps Anthropic's "10+ for complex"

    # Live counters (mutated as work happens)
    used_input_tokens: int = 0
    used_output_tokens: int = 0
    used_dollars: float = 0.0
    spawned_subagents: int = 0
    started_at: float = field(default_factory=time.monotonic)

    # Per-call attribution (for observability)
    cost_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.used_input_tokens + self.used_output_tokens

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def record_llm_call(
        self, *, model: str, input_tokens: int, output_tokens: int, label: str = ""
    ) -> None:
        """Account for an LLM call's tokens + dollars.

        Raises BudgetExceededError if any ceiling is now breached.
        Caller is expected to call this AFTER the call completes,
        so a single overshoot won't break — but the next call will be
        refused.
        """
        cost = _estimate_dollars(model, input_tokens, output_tokens)
        self.used_input_tokens += input_tokens
        self.used_output_tokens += output_tokens
        self.used_dollars += cost
        self.cost_log.append(
            {
                "model": model,
                "in": input_tokens,
                "out": output_tokens,
                "dollars": cost,
                "label": label,
                "t": time.monotonic() - self.started_at,
            }
        )
        self._check_ceilings()

    def record_subagent_spawn(self) -> None:
        """Account for a subagent spawn. Refuses if over max_subagents."""
        if self.spawned_subagents >= self.max_subagents:
            raise BudgetExceededError(
                f"max_subagents={self.max_subagents} reached; refusing spawn"
            )
        self.spawned_subagents += 1

    def _check_ceilings(self) -> None:
        if self.total_tokens > self.max_total_tokens:
            raise BudgetExceededError(
                f"max_total_tokens={self.max_total_tokens} exceeded "
                f"(used={self.total_tokens})"
            )
        if self.used_dollars > self.max_total_dollars:
            raise BudgetExceededError(
                f"max_total_dollars=${self.max_total_dollars:.2f} exceeded "
                f"(used=${self.used_dollars:.2f})"
            )
        if self.elapsed_seconds > self.max_wall_seconds:
            raise BudgetExceededError(
                f"max_wall_seconds={self.max_wall_seconds} exceeded "
                f"(elapsed={self.elapsed_seconds:.1f}s)"
            )

    def remaining(self) -> dict[str, Any]:
        """Snapshot of remaining budget — passed to subagents so they
        can scope down when running low.
        """
        return {
            "tokens": max(0, self.max_total_tokens - self.total_tokens),
            "dollars": max(0.0, self.max_total_dollars - self.used_dollars),
            "wall_seconds": max(0, self.max_wall_seconds - int(self.elapsed_seconds)),
            "subagents": max(0, self.max_subagents - self.spawned_subagents),
        }

    def slice_for_subagent(self, *, max_steps: int) -> "OrchestrationBudget":
        """Carve a per-subagent budget slice from remaining.

        Approximates fair-share: a subagent can use up to remaining/spawned
        of the orchestration's remaining budget. The orchestrator's ledger
        is the source of truth — this slice is purely for the subagent's
        own DELIBERATE step (so it can decide whether to keep going).
        """
        remaining_subagents = max(1, self.max_subagents - self.spawned_subagents)
        remaining_tokens = max(0, self.max_total_tokens - self.total_tokens)
        remaining_dollars = max(0.0, self.max_total_dollars - self.used_dollars)
        return OrchestrationBudget(
            max_total_tokens=remaining_tokens // remaining_subagents,
            max_total_dollars=remaining_dollars / remaining_subagents,
            max_wall_seconds=max(
                15, int(self.max_wall_seconds - self.elapsed_seconds) // 2
            ),
            max_subagents=max(1, max_steps),  # subagents typically don't sub-spawn
        )

    def snapshot(self) -> dict[str, Any]:
        """Full state for logging / OutcomeStore."""
        return {
            "max_total_tokens": self.max_total_tokens,
            "max_total_dollars": self.max_total_dollars,
            "max_wall_seconds": self.max_wall_seconds,
            "max_subagents": self.max_subagents,
            "used_input_tokens": self.used_input_tokens,
            "used_output_tokens": self.used_output_tokens,
            "used_dollars": round(self.used_dollars, 4),
            "spawned_subagents": self.spawned_subagents,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


@dataclass
class CancellationToken:
    """Cooperative cancellation propagated through the PRELUDE loop.

    Workers/tools should check `is_cancelled()` at safe checkpoints and
    raise CancellationError if True. This avoids the orphan-write
    problem where a cancelled subagent's tool call still completes and
    writes stale data to memory.
    """

    _cancelled: bool = False
    _reason: str = ""

    def cancel(self, reason: str = "user_requested") -> None:
        self._cancelled = True
        self._reason = reason

    def is_cancelled(self) -> bool:
        return self._cancelled

    @property
    def reason(self) -> str:
        return self._reason

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            from .exceptions import CancellationError

            raise CancellationError(f"Cancelled: {self._reason}")
