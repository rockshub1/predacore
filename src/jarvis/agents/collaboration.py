"""
Agent collaboration patterns for JARVIS.

Provides multi-agent orchestration strategies:
- FAN_OUT:    dispatch same task to N agents, collect all results
- PIPELINE:   chain agents sequentially (output → next input)
- CONSENSUS:  vote-based decision with configurable quorum
- SUPERVISOR: one agent reviews and corrects another's output
"""
from __future__ import annotations

import asyncio
import enum
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class CollaborationPattern(enum.Enum):
    FAN_OUT = "fan_out"
    PIPELINE = "pipeline"
    CONSENSUS = "consensus"
    SUPERVISOR = "supervisor"


@dataclass
class AgentSpec:
    """Describes a participating agent."""

    agent_id: str = ""
    agent_type: str = ""
    capabilities: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.agent_id:
            self.agent_id = f"agent-{uuid4().hex[:8]}"


@dataclass
class TaskPayload:
    """A unit of work dispatched to an agent."""

    task_id: str = ""
    prompt: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0

    def __post_init__(self) -> None:
        if not self.task_id:
            self.task_id = f"task-{uuid4().hex[:8]}"


@dataclass
class AgentResult:
    """Result returned by a single agent."""

    agent_id: str = ""
    output: Any = None
    error: str = ""
    latency_ms: float = 0.0
    success: bool = True


@dataclass
class CollaborationResult:
    """Aggregate result of a collaboration round."""

    pattern: CollaborationPattern = CollaborationPattern.FAN_OUT
    results: list[AgentResult] = field(default_factory=list)
    final_output: Any = None
    total_latency_ms: float = 0.0
    consensus_reached: bool = True

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if not r.success)


# Type alias for agent callables
AgentFn = Callable[[TaskPayload], Coroutine[Any, Any, Any]]


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------


class ResultAggregator:
    """Merge multiple agent outputs into a single answer."""

    @staticmethod
    def union(results: list[AgentResult]) -> list[Any]:
        """Combine all successful outputs into a list."""
        return [r.output for r in results if r.success]

    @staticmethod
    def majority_vote(results: list[AgentResult]) -> Any:
        """Return the most common successful output."""
        votes: dict[str, int] = {}
        output_map: dict[str, Any] = {}
        for r in results:
            if not r.success:
                continue
            key = str(r.output)
            votes[key] = votes.get(key, 0) + 1
            output_map[key] = r.output
        if not votes:
            return None
        winner = max(votes, key=lambda k: votes[k])
        return output_map[winner]

    @staticmethod
    def weighted_vote(
        results: list[AgentResult],
        weights: dict[str, float] | None = None,
    ) -> Any:
        """Return the output with the highest weighted score."""
        weights = weights or {}
        scores: dict[str, float] = {}
        output_map: dict[str, Any] = {}
        for r in results:
            if not r.success:
                continue
            key = str(r.output)
            w = weights.get(r.agent_id, 1.0)
            scores[key] = scores.get(key, 0.0) + w
            output_map[key] = r.output
        if not scores:
            return None
        winner = max(scores, key=lambda k: scores[k])
        return output_map[winner]


# ---------------------------------------------------------------------------
# Task distributor
# ---------------------------------------------------------------------------


class TaskDistributor:
    """Split a complex task into subtasks for multiple agents."""

    @staticmethod
    def split_by_sections(
        prompt: str,
        agents: list[AgentSpec],
    ) -> list[tuple[AgentSpec, TaskPayload]]:
        """
        Naively split prompt into N equal parts, one per agent.
        Real implementations would use LLM-based decomposition.
        """
        lines = prompt.strip().split("\n")
        chunk_size = max(1, len(lines) // len(agents))
        assignments: list[tuple[AgentSpec, TaskPayload]] = []
        for i, agent in enumerate(agents):
            start = i * chunk_size
            end = start + chunk_size if i < len(agents) - 1 else len(lines)
            chunk = "\n".join(lines[start:end])
            assignments.append(
                (
                    agent,
                    TaskPayload(prompt=chunk, context={"section": i}),
                )
            )
        return assignments

    @staticmethod
    def replicate(
        task: TaskPayload,
        agents: list[AgentSpec],
    ) -> list[tuple[AgentSpec, TaskPayload]]:
        """Send the same task to every agent."""
        return [(agent, task) for agent in agents]


# ---------------------------------------------------------------------------
# Agent team orchestrator
# ---------------------------------------------------------------------------


class AgentTeam:
    """
    Orchestrates multiple agents using a specified collaboration pattern.

    Usage:
        team = AgentTeam()
        team.add_agent(spec, agent_fn)
        result = await team.fan_out(task)
    """

    def __init__(self, log: logging.Logger | None = None) -> None:
        self._agents: dict[str, tuple[AgentSpec, AgentFn]] = {}
        self._log = log or logger

    def add_agent(self, spec: AgentSpec, fn: AgentFn) -> None:
        """Register an agent with its execution function."""
        self._agents[spec.agent_id] = (spec, fn)
        self._log.debug("Registered agent %s (%s)", spec.agent_id, spec.agent_type)

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    # -- Patterns -----------------------------------------------------------

    async def fan_out(self, task: TaskPayload) -> CollaborationResult:
        """Dispatch same task to all agents in parallel, collect all results."""
        t0 = time.monotonic()
        coros = []
        agent_ids = []
        for agent_id, (_spec, fn) in self._agents.items():
            agent_ids.append(agent_id)
            coros.append(self._execute_agent(agent_id, fn, task))

        agent_results = await asyncio.gather(*coros, return_exceptions=True)
        results: list[AgentResult] = []
        for aid, res in zip(agent_ids, agent_results):
            if isinstance(res, Exception):
                results.append(AgentResult(agent_id=aid, error=str(res), success=False))
            else:
                results.append(res)

        elapsed = (time.monotonic() - t0) * 1000
        return CollaborationResult(
            pattern=CollaborationPattern.FAN_OUT,
            results=results,
            final_output=ResultAggregator.union(results),
            total_latency_ms=elapsed,
        )

    async def pipeline(
        self,
        task: TaskPayload,
        order: list[str] | None = None,
    ) -> CollaborationResult:
        """Chain agents sequentially — each agent's output becomes the next's input."""
        t0 = time.monotonic()
        results: list[AgentResult] = []
        current_input = task

        agent_order = order or list(self._agents.keys())
        for agent_id in agent_order:
            if agent_id not in self._agents:
                results.append(
                    AgentResult(
                        agent_id=agent_id,
                        error=f"Agent {agent_id} not found",
                        success=False,
                    )
                )
                break
            _, fn = self._agents[agent_id]
            result = await self._execute_agent(agent_id, fn, current_input)
            results.append(result)
            if not result.success:
                break
            # Feed output as next prompt
            current_input = TaskPayload(
                prompt=str(result.output),
                context={**task.context, "pipeline_step": len(results)},
                timeout=task.timeout,
            )

        elapsed = (time.monotonic() - t0) * 1000
        final = results[-1].output if results and results[-1].success else None
        return CollaborationResult(
            pattern=CollaborationPattern.PIPELINE,
            results=results,
            final_output=final,
            total_latency_ms=elapsed,
        )

    async def consensus(
        self,
        task: TaskPayload,
        quorum: float = 0.5,
    ) -> CollaborationResult:
        """
        All agents vote on the task. Consensus is reached if
        the majority answer has >= quorum fraction of votes.
        """
        fan_result = await self.fan_out(task)
        winner = ResultAggregator.majority_vote(fan_result.results)

        # Check quorum
        if winner is not None and fan_result.success_count > 0:
            vote_count = sum(
                1
                for r in fan_result.results
                if r.success and str(r.output) == str(winner)
            )
            fraction = vote_count / max(fan_result.success_count, 1)
            reached = fraction >= quorum
        else:
            reached = False

        return CollaborationResult(
            pattern=CollaborationPattern.CONSENSUS,
            results=fan_result.results,
            final_output=winner,
            total_latency_ms=fan_result.total_latency_ms,
            consensus_reached=reached,
        )

    async def supervise(
        self,
        task: TaskPayload,
        worker_id: str,
        supervisor_id: str,
    ) -> CollaborationResult:
        """Worker produces output; supervisor reviews and optionally corrects."""
        t0 = time.monotonic()
        results: list[AgentResult] = []

        # Worker phase
        if worker_id not in self._agents:
            return CollaborationResult(
                pattern=CollaborationPattern.SUPERVISOR,
                results=[
                    AgentResult(agent_id=worker_id, error="Not found", success=False)
                ],
            )
        _, worker_fn = self._agents[worker_id]
        worker_result = await self._execute_agent(worker_id, worker_fn, task)
        results.append(worker_result)

        # Supervisor phase
        if supervisor_id not in self._agents:
            return CollaborationResult(
                pattern=CollaborationPattern.SUPERVISOR,
                results=results,
                final_output=worker_result.output if worker_result.success else None,
            )
        _, sup_fn = self._agents[supervisor_id]
        review_task = TaskPayload(
            prompt=f"Review this output and correct if needed:\n{worker_result.output}",
            context={
                **task.context,
                "original_prompt": task.prompt,
                "role": "supervisor",
            },
            timeout=task.timeout,
        )
        sup_result = await self._execute_agent(supervisor_id, sup_fn, review_task)
        results.append(sup_result)

        elapsed = (time.monotonic() - t0) * 1000
        final = sup_result.output if sup_result.success else worker_result.output
        return CollaborationResult(
            pattern=CollaborationPattern.SUPERVISOR,
            results=results,
            final_output=final,
            total_latency_ms=elapsed,
        )

    # -- Execution helper ---------------------------------------------------

    async def _execute_agent(
        self,
        agent_id: str,
        fn: AgentFn,
        task: TaskPayload,
    ) -> AgentResult:
        t0 = time.monotonic()
        try:
            output = await asyncio.wait_for(fn(task), timeout=task.timeout)
            elapsed = (time.monotonic() - t0) * 1000
            return AgentResult(
                agent_id=agent_id,
                output=output,
                latency_ms=elapsed,
                success=True,
            )
        except asyncio.TimeoutError:
            elapsed = (time.monotonic() - t0) * 1000
            return AgentResult(
                agent_id=agent_id,
                error="Timeout",
                latency_ms=elapsed,
                success=False,
            )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return AgentResult(
                agent_id=agent_id,
                error=str(e),
                latency_ms=elapsed,
                success=False,
            )
