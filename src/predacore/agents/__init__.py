"""
predacore.agents — multi-agent orchestration layer.

Two execution surfaces share this package:

- **In-process agents** — `engine.AgentEngine` runs typed agents (`AGENT_TYPES`,
  `DynamicAgentSpec`) in the same Python process under a shared `ToolContext`.
  Capability routing, tool allowlists, and meta-cognition loop detection live
  here. Used directly by `tools/handlers/agent.py`.

- **Process-isolated agents (DAF)** — `daf/` runs each agent as a separate
  OS process driven by gRPC. `daf_bridge.DAFBridge` dispatches tasks; workers
  in `daf/agent_process.py` pull from `task_store` and execute either the
  static-type backend or, when supplied an `agent_spec_json`, bootstrap the
  full PredaCore stack inside the worker.

The same `engine.AgentEngine` runs in both surfaces — DAF wraps it for
isolation, in-process callers use it directly.

## New orchestration layer (May 2026)

A consolidated `Orchestrator` sits above both execution surfaces and
implements Anthropic's six patterns + Self-MoA, with the PRELUDE
memory-grounded loop driving each subagent run.

    from predacore.agents import Orchestrator, OrchestrationBudget
    orch = Orchestrator(llm=..., memory=..., handler_map=..., tool_executor=...)
    result = await orch.run("compare anthropic vs openai on safety")

The orchestrator picks the pattern (autonomous / orchestrator-workers /
self-moa / evaluator-optimizer / parallelize / routing / prompt-chain)
based on task class + prior-run hints from memory. Runner choice
(in-process vs DAF) is workload-class-based, not agent-count-based.

Activation: `PREDACORE_USE_ORCHESTRATOR=1` to flip the new path on.
Default: legacy `core.process()` path remains until callers migrate.

## Other modules

- `autonomy.OpenClawBridgeRuntime` — async OpenClaw delegation runtime with
  idempotency, action ledger, kill switch, retry/backoff.
- `collaboration` — DEPRECATED: fan-out / pipeline / consensus / supervise
  patterns. Will be removed once orchestrator migration completes; use
  `agents.patterns.*` instead.
- `meta_cognition` — response evaluation + tool-loop detection heuristics.
- `self_improvement.SelfImprovementEngine` — failure-pattern proposal engine.

## Lazy export policy

The orchestrator + patterns + runners + spec + budget are exported at
the top level (lightweight imports — no gRPC pulled). DAF modules
remain lazy-only (importing `daf_bridge` would pull gRPC).
"""
from .budget import CancellationToken, OrchestrationBudget
from .critic import CriticGate, CriticReview, DEFAULT_HIGH_STAKES_TOOLS
from .exceptions import (
    BudgetExceededError,
    CancellationError,
    CriticVetoError,
    DuplicateOrchestrationError,
    OrchestrationError,
    PatternRejectError,
    SpecValidationError,
)
from .inflight import OrchestrationInFlight, orchestration_key
from .orchestrator import (
    OrchestrationResult,
    Orchestrator,
    OrchestratorConfig,
    PatternName,
)
from .patterns import (
    AutonomousPattern,
    ChainStep,
    EvaluatorOptimizerPattern,
    OrchestratorWorkersPattern,
    ParallelizePattern,
    Pattern,
    PatternResult,
    PromptChainPattern,
    Route,
    RoutingPattern,
    SelfMoAPattern,
)
from .runner_loop import (
    PRELUDE_CONSTANTS,
    PreludeResult,
    StepRecord,
    reflect_progress,
    run_prelude,
)
from .runners import AgentResult, DAFRunner, InProcessRunner, RunContext, Runner
from .spec import ANTHROPIC_SCALING_RULES, AgentSpec, validate_spec
from .trust_routing import is_high_trust, needs_isolation, runner_for_trust_level

__all__ = [
    # Foundation
    "AgentSpec",
    "ANTHROPIC_SCALING_RULES",
    "validate_spec",
    "OrchestrationBudget",
    "CancellationToken",
    # Exceptions
    "OrchestrationError",
    "BudgetExceededError",
    "CancellationError",
    "CriticVetoError",
    "DuplicateOrchestrationError",
    "PatternRejectError",
    "SpecValidationError",
    # Orchestrator
    "Orchestrator",
    "OrchestratorConfig",
    "OrchestrationResult",
    "PatternName",
    # Patterns
    "Pattern",
    "PatternResult",
    "AutonomousPattern",
    "ChainStep",
    "EvaluatorOptimizerPattern",
    "OrchestratorWorkersPattern",
    "ParallelizePattern",
    "PromptChainPattern",
    "Route",
    "RoutingPattern",
    "SelfMoAPattern",
    # Runners
    "AgentResult",
    "DAFRunner",
    "InProcessRunner",
    "RunContext",
    "Runner",
    # Loop
    "PRELUDE_CONSTANTS",
    "PreludeResult",
    "StepRecord",
    "reflect_progress",
    "run_prelude",
    # Critic
    "CriticGate",
    "CriticReview",
    "DEFAULT_HIGH_STAKES_TOOLS",
    # In-flight dedup
    "OrchestrationInFlight",
    "orchestration_key",
    # Trust-level routing (Flame / untrusted skills → DAF)
    "is_high_trust",
    "needs_isolation",
    "runner_for_trust_level",
]
