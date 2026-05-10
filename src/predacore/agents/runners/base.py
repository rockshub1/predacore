"""Runner ABC — pluggable backend for executing one AgentSpec.

Two implementations:
  - InProcessRunner: asyncio.gather over local AgentEngine.run_task calls.
                     Default. Best for I/O-bound (LLM HTTP) work — which
                     is 95% of agent workloads.
  - DAFRunner:       gRPC dispatch to predacore DAF worker pool.
                     Escalation only: untrusted skills, ≥60 s background,
                     crash isolation.

Both runners share the same RunContext so patterns are runner-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..budget import CancellationToken, OrchestrationBudget
from ..spec import AgentSpec


@dataclass
class AgentResult:
    """Result returned by a runner after a subagent completes.

    `output` is the user-facing string. `meta` carries provenance (which
    runner, latency, tokens, tool calls) for the synthesizer.
    """

    spec_id: str
    spec: AgentSpec
    output: str
    success: bool = True
    error: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)  # memories stored
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    runner: str = "inprocess"


@dataclass
class RunContext:
    """Per-orchestration context passed through pattern → runner → loop.

    Carries everything a subagent needs to coordinate: budget, cancel
    signal, memory client, team_id for working-memory scope, trace_id.
    """

    run_id: str
    team_id: str  # = run_id by default; can override for shared-team work
    trace_id: str
    user_id: str
    session_id: str = ""

    # Resources (duck-typed to avoid import cycles)
    llm: Any = None              # LLMInterface
    memory: Any = None           # UnifiedMemoryStore | RemoteMemoryStore
    tool_executor: Any = None    # ToolExecutor
    handler_map: Any = None      # HANDLER_MAP

    # Coordination
    budget: OrchestrationBudget = field(default_factory=OrchestrationBudget)
    cancellation: CancellationToken = field(default_factory=CancellationToken)

    # Routing hints
    pattern_name: str = ""       # filled in by Orchestrator
    runner_name: str = "inprocess"

    # Original task (the user's question or upstream prompt)
    task: str = ""


class Runner(ABC):
    """Pluggable backend for executing one AgentSpec."""

    name: str = "abstract"

    @abstractmethod
    async def run_spec(self, spec: AgentSpec, ctx: RunContext) -> AgentResult:
        """Execute one subagent. Must respect ctx.budget + ctx.cancellation.

        Implementations must:
          - check ctx.cancellation.is_cancelled() before/after each tool
          - record token usage on ctx.budget.record_llm_call(...)
          - record_subagent_spawn() is called by the orchestrator before
            this is invoked, not by the runner itself
          - return an AgentResult even on error (success=False, error=...)
            rather than raising — orchestrator decides how to react
        """
