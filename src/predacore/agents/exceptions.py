"""Exception hierarchy for the orchestrator + PRELUDE loop.

Distinct types so callers can branch on cause without string matching.
Every orchestration error is an OrchestrationError (or subclass).
"""
from __future__ import annotations


class OrchestrationError(RuntimeError):
    """Base for all orchestrator/loop-level errors."""


class BudgetExceededError(OrchestrationError):
    """Hard ceiling on tokens / dollars / wall-time / subagents was hit.

    The orchestrator should NOT retry on this — it means the user-set
    budget is exhausted. Surface to user with current usage stats.
    """


class SpecValidationError(OrchestrationError):
    """DynamicAgentSpec failed the rigour validator.

    Refuse-to-spawn is cheaper than running a vague subagent that
    produces duplicate work (Anthropic's documented failure mode).
    """


class CancellationError(OrchestrationError):
    """Cancellation requested via CancellationToken.

    Propagates through PRELUDE loop and tool calls. Workers should
    raise this on next checkpoint after token.is_cancelled() flips.
    """


class CriticVetoError(OrchestrationError):
    """High-stakes critic gate vetoed an action.

    Carries the critic's suggested revision (if any) so the agent
    loop can DELIBERATE again with the critic's note.
    """

    def __init__(self, message: str, *, revision_hint: str | None = None) -> None:
        super().__init__(message)
        self.revision_hint = revision_hint


class DuplicateOrchestrationError(OrchestrationError):
    """An orchestration with the same idempotency key is already in-flight.

    Carries the in-flight orchestration's id so the caller can await
    its result instead of starting a second run.
    """

    def __init__(self, message: str, *, existing_run_id: str) -> None:
        super().__init__(message)
        self.existing_run_id = existing_run_id


class PatternRejectError(OrchestrationError):
    """Classifier could not select a pattern (ambiguous task)."""
