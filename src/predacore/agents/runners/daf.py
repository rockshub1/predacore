"""DAFRunner — escalation runner that dispatches subagents to the
process-isolated DAF worker pool over gRPC.

Use only when:
  - Untrusted skill execution (Tier 1 SANDBOXED Flame skills)
  - Long-running background work (≥60 s) — don't block the main loop
  - Crash isolation required (a misbehaving subagent must not kill the daemon)
  - CPU-bound work that needs multi-core (rare for LLM-driven agents)

For typical LLM-bound subagents, use InProcessRunner instead. asyncio
gives equivalent parallelism for I/O work without the 1-2 s spawn cost.

This runner is a thin transport adapter over predacore.agents.daf_bridge.
The orchestration logic (which Pattern, when to use DAF) lives in the
Orchestrator. Once we delete the legacy `dispatch_multi_agent` from
daf_bridge.py, this is the only DAF entrypoint.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from ..exceptions import CancellationError
from ..spec import AgentSpec
from .base import AgentResult, RunContext, Runner

logger = logging.getLogger(__name__)


class DAFRunner(Runner):
    """gRPC dispatch to a DAF worker pool.

    Each `run_spec` spawns one DAF process worker, dispatches the spec
    via DAFBridge, awaits the result, and retires the worker.

    Workers are configured to use `RemoteMemoryStore` over UDS — they
    don't load BGE locally or rebuild HNSW. With Phase 1 in place,
    cold-start drops from ~5 s to ~500 ms.
    """

    name = "daf"

    def __init__(self, daf_bridge: Any | None = None) -> None:
        self._bridge = daf_bridge

    async def run_spec(self, spec: AgentSpec, ctx: RunContext) -> AgentResult:
        ctx.cancellation.raise_if_cancelled()
        started = time.monotonic()

        bridge = self._bridge or self._build_bridge()
        if bridge is None or not getattr(bridge, "available", False):
            return AgentResult(
                spec_id=spec.id,
                spec=spec,
                output="",
                success=False,
                error="DAFBridge unavailable",
                runner=self.name,
            )

        try:
            results = await bridge.dispatch_multi_agent(
                prompt=spec.objective,
                agent_roles=[spec.base_type],
                pattern="single",
                team_id=ctx.team_id,
                user_id=ctx.user_id,
                session_id=ctx.session_id,
                agent_specs=[spec.to_dict()],
                home_dir=None,
            )
        except CancellationError:
            return AgentResult(
                spec_id=spec.id,
                spec=spec,
                output="",
                success=False,
                error="cancelled",
                latency_ms=(time.monotonic() - started) * 1000,
                runner=self.name,
            )
        except Exception as exc:  # noqa: BLE001 — runner boundary
            logger.warning("DAFRunner.run_spec(%s) failed: %s", spec.id, exc, exc_info=True)
            return AgentResult(
                spec_id=spec.id,
                spec=spec,
                output="",
                success=False,
                error=str(exc),
                latency_ms=(time.monotonic() - started) * 1000,
                runner=self.name,
            )

        if not results:
            return AgentResult(
                spec_id=spec.id, spec=spec, output="", success=False,
                error="no DAF result", runner=self.name,
                latency_ms=(time.monotonic() - started) * 1000,
            )
        first = results[0]
        return AgentResult(
            spec_id=spec.id,
            spec=spec,
            output=getattr(first, "output", "") or "",
            success=getattr(first, "status", "") == "completed",
            error=getattr(first, "error", "") or "",
            latency_ms=getattr(first, "latency_ms", 0.0)
            or (time.monotonic() - started) * 1000,
            runner=self.name,
        )

    def _build_bridge(self) -> Any | None:
        try:
            from ..daf_bridge import DAFBridge, DAFBridgeConfig

            return DAFBridge(DAFBridgeConfig())
        except (ImportError, RuntimeError) as exc:
            logger.debug("DAFRunner could not construct DAFBridge: %s", exc)
            return None
