"""
PredaCore ↔ DAF Bridge — Connects PredaCore orchestration to DAF process management.

Allows PredaCore multi-agent tasks to optionally dispatch to DAF for:
- True process-level parallelism (isolated agent processes)
- Agent lifecycle management (spawn, monitor, retire)
- Capability-based routing (match tasks to agent types)
- Persistent task tracking across restarts

When DAF is unavailable, falls back to PredaCore's in-process async agents.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


def _value_to_text(value: Any) -> str:
    """Render protobuf Value payloads as plain text."""
    if value is None:
        return ""
    which = getattr(value, "WhichOneof", lambda *_: None)("kind")
    if which == "string_value":
        return value.string_value
    if which == "number_value":
        return str(value.number_value)
    if which == "bool_value":
        return "true" if value.bool_value else "false"
    if which == "null_value":
        return "null"
    if which == "struct_value":
        try:
            return json.dumps(
                {
                    key: _value_to_text(inner)
                    for key, inner in value.struct_value.fields.items()
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        except (TypeError, ValueError, AttributeError):
            return str(value)
    if which == "list_value":
        try:
            return json.dumps(
                [_value_to_text(inner) for inner in value.list_value.values],
                ensure_ascii=False,
            )
        except (TypeError, ValueError, AttributeError):
            return str(value)
    return str(value or "")


@dataclass
class DAFTaskResult:
    """Result from a DAF-dispatched task."""

    task_id: str
    agent_type: str
    status: str  # "completed", "failed", "timeout"
    output: str = ""
    error: str = ""
    latency_ms: float = 0.0
    agent_instance_id: str = ""


@dataclass
class DAFBridgeConfig:
    """Configuration for the DAF bridge."""

    enabled: bool = True
    grpc_target: str = "localhost:50051"
    timeout_seconds: float = 120.0
    prefer_daf: bool = False  # If True, prefer DAF over in-process; if False, only use DAF for heavy tasks
    min_agents_for_daf: int = 3  # Only use DAF when >= this many agents needed


class DAFBridge:
    """Bridge between PredaCore tool system and DAF process fabric.

    Usage from handle_multi_agent:
        bridge = DAFBridge(config)
        if bridge.should_use_daf(num_agents, pattern):
            results = await bridge.dispatch_multi_agent(prompt, agents, pattern)
        else:
            # fall back to in-process AgentTeam
    """

    _AVAILABILITY_TTL = 60  # seconds before re-checking DAF reachability

    def __init__(self, config: DAFBridgeConfig | None = None):
        self.config = config or DAFBridgeConfig()
        self._stub = None
        self._channel = None
        self._available: bool | None = None  # Cached availability
        self._available_checked_at: float = 0.0
        self._channel_lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        """Check if DAF gRPC service is reachable (cached with TTL)."""
        if self._available is not None and (time.time() - self._available_checked_at) < self._AVAILABILITY_TTL:
            return self._available

        if not self.config.enabled:
            self._available = False
            return False

        try:
            import grpc
            use_tls = os.getenv("GRPC_USE_TLS", "").lower() in ("1", "true")
            if not use_tls and not self.config.grpc_target.startswith("localhost"):
                logger.warning(
                    "gRPC using insecure_channel to non-local target %s — "
                    "set GRPC_USE_TLS=1 for production",
                    self.config.grpc_target,
                )
            if use_tls:
                channel = grpc.secure_channel(self.config.grpc_target, grpc.ssl_channel_credentials())
            else:
                channel = grpc.insecure_channel(self.config.grpc_target)
            # Quick connectivity check — use a short timeout to avoid blocking the event loop
            try:
                grpc.channel_ready_future(channel).result(timeout=0.5)
                self._available = True
                self._available_checked_at = time.time()
            except grpc.FutureTimeoutError:
                self._available = False
                self._available_checked_at = time.time()
            finally:
                channel.close()
        except ImportError:
            logger.debug("grpc not installed — DAF bridge unavailable")
            self._available = False
            self._available_checked_at = time.time()
        except (OSError, ConnectionError) as e:
            logger.debug("DAF bridge connectivity check failed: %s", e)
            self._available = False
            self._available_checked_at = time.time()

        return self._available if self._available is not None else False

    def should_use_daf(self, num_agents: int, pattern: str = "fan_out") -> bool:
        """Decide whether to dispatch to DAF or use in-process agents."""
        if not self.config.enabled:
            return False
        if self.config.prefer_daf:
            return self.available
        # Only use DAF for heavier workloads
        if num_agents >= self.config.min_agents_for_daf:
            return self.available
        return False

    async def dispatch_multi_agent(
        self,
        prompt: str,
        agent_roles: list[str],
        pattern: str = "fan_out",
        timeout: float | None = None,
        *,
        team_id: str | None = None,
        user_id: str = "default",
        session_id: str | None = None,
        agent_specs: list[dict[str, Any]] | None = None,
        home_dir: str | None = None,
    ) -> list[DAFTaskResult]:
        """Dispatch a multi-agent task to DAF.

        Each agent role is mapped to a DAF agent type and spawned as
        an isolated process. Results are collected and returned.
        """
        timeout = timeout or self.config.timeout_seconds

        try:
            import grpc
            from predacore._vendor.common.protos import daf_pb2, daf_pb2_grpc, wil_pb2
            from google.protobuf.struct_pb2 import Struct
        except ImportError as e:
            logger.warning("DAF bridge: missing deps — %s", e)
            return [DAFTaskResult(
                task_id="",
                agent_type="",
                status="failed",
                error=f"DAF dependencies not available: {e}",
            )]

        async with self._channel_lock:
            if self._channel is None:
                if os.getenv("GRPC_USE_TLS", "").lower() in ("1", "true"):
                    self._channel = grpc.aio.secure_channel(
                        self.config.grpc_target, grpc.ssl_channel_credentials()
                    )
                else:
                    self._channel = grpc.aio.insecure_channel(self.config.grpc_target)
                self._stub = daf_pb2_grpc.DynamicAgentFabricControllerServiceStub(self._channel)

        results: list[DAFTaskResult] = []

        # Map PredaCore roles to DAF agent types (extensible via config)
        role_to_type = {
            "analyst": "web_searcher",
            "researcher": "web_searcher",
            "creative": "web_searcher",
            "critic": "web_searcher",
            "synthesizer": "web_searcher",
            "planner": "web_searcher",
            "coder": "web_searcher",
            "generalist": "web_searcher",
        }
        if hasattr(self.config, "role_type_overrides") and self.config.role_type_overrides:
            role_to_type.update(self.config.role_type_overrides)

        specs = list(agent_specs or [])

        async def _spawn_and_dispatch(role: str, spec_payload: dict[str, Any] | None = None) -> DAFTaskResult:
            started_at = time.time()
            spec_payload = dict(spec_payload or {})
            base_type = str(spec_payload.get("base_type") or role).strip().lower()
            backend_type = role_to_type.get(base_type, role_to_type.get(role, "web_searcher"))
            compiled_prompt = str(spec_payload.get("compiled_prompt") or prompt).strip() or prompt
            mission = str(spec_payload.get("mission") or prompt).strip() or prompt
            instance_id = ""
            task_id = f"daf-{uuid4().hex[:12]}"
            task_config = Struct()
            task_config.update(
                {
                    "role": role,
                    "base_type": base_type,
                    "specialization": str(spec_payload.get("specialization") or role),
                    "pattern": pattern,
                    "team_id": team_id or "",
                    "home_dir": home_dir or "",
                }
            )
            try:
                spawn_req = daf_pb2.SpawnAgentRequest(
                    agent_type_id=backend_type,
                    initial_configuration=task_config,
                    task_id_for_agent=task_id,
                )
                spawn_resp = await asyncio.wait_for(
                    self._stub.SpawnAgent(spawn_req),
                    timeout=timeout,
                )
                if not spawn_resp.success or not spawn_resp.HasField("agent_instance"):
                    raise RuntimeError(spawn_resp.error_message or "SpawnAgent failed")
                instance_id = spawn_resp.agent_instance.agent_instance_id
                logger.info(
                    "DAF spawned agent %s (backend=%s, role=%s)",
                    instance_id,
                    backend_type,
                    role,
                )

                parameters = Struct()
                parameters.update(
                    {
                        "prompt": compiled_prompt,
                        "original_task": prompt,
                        "query": mission,
                        "role": role,
                        "base_type": base_type,
                        "pattern": pattern,
                        "team_id": team_id or "",
                        "user_id": user_id,
                        "agent_spec_json": json.dumps(spec_payload, ensure_ascii=False),
                        "home_dir": home_dir or "",
                    }
                )
                context = Struct()
                context.update(
                    {
                        "use_pull_loop": True,
                        "team_id": team_id or "",
                        "user_id": user_id,
                        "session_id": session_id or "",
                        "role": role,
                        "base_type": base_type,
                        "pattern": pattern,
                        "backend_agent_type": backend_type,
                        "home_dir": home_dir or "",
                    }
                )
                task_req = daf_pb2.TaskAssignmentMessage(
                    task_id=task_id,
                    plan_step_id=f"plan-{uuid4().hex[:8]}",
                    agent_instance_id=instance_id,
                    parameters=parameters,
                    context=context,
                )
                task_resp = await asyncio.wait_for(
                    self._stub.DispatchTask(task_req),
                    timeout=timeout,
                )
                output_text = _value_to_text(task_resp.output)
                error_text = str(task_resp.error_message or "")
                success_status = int(
                    wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS
                )
                if int(task_resp.status) == success_status:
                    if not output_text:
                        output_text = (
                            f"DAF role {role} completed without returning textual output."
                        )
                    return DAFTaskResult(
                        task_id=task_resp.task_id or task_id,
                        agent_type=role,
                        status="completed",
                        output=output_text,
                        latency_ms=(time.time() - started_at) * 1000,
                        agent_instance_id=instance_id,
                    )
                return DAFTaskResult(
                    task_id=task_resp.task_id or task_id,
                    agent_type=role,
                    status="failed",
                    error=error_text or output_text or "DAF task failed",
                    latency_ms=(time.time() - started_at) * 1000,
                    agent_instance_id=instance_id,
                )
            except asyncio.TimeoutError:
                return DAFTaskResult(
                    task_id=task_id,
                    agent_type=role,
                    status="timeout",
                    error=f"DAF agent timed out after {timeout}s",
                    latency_ms=(time.time() - started_at) * 1000,
                    agent_instance_id=instance_id,
                )
            except (OSError, ConnectionError, RuntimeError, AttributeError, ValueError) as e:
                return DAFTaskResult(
                    task_id=task_id,
                    agent_type=role,
                    status="failed",
                    error=str(e),
                    latency_ms=(time.time() - started_at) * 1000,
                    agent_instance_id=instance_id,
                )
            finally:
                if instance_id:
                    try:
                        await asyncio.wait_for(
                            self._stub.RetireAgent(
                                daf_pb2.RetireAgentRequest(
                                    agent_instance_id=instance_id
                                )
                            ),
                            timeout=min(float(timeout), 5.0),
                        )
                    except (
                        asyncio.TimeoutError,
                        OSError,
                        ConnectionError,
                        RuntimeError,
                        AttributeError,
                        ValueError,
                    ):
                        logger.debug(
                            "DAF retire failed for %s",
                            instance_id,
                            exc_info=True,
                        )

        # Run all spawns concurrently
        daf_results = await asyncio.gather(
            *[
                _spawn_and_dispatch(
                    role,
                    specs[idx] if idx < len(specs) else None,
                )
                for idx, role in enumerate(agent_roles)
            ],
            return_exceptions=True,
        )

        for r in daf_results:
            if isinstance(r, Exception):
                results.append(DAFTaskResult(
                    task_id="", agent_type="unknown", status="failed",
                    error=str(r),
                ))
            else:
                results.append(r)

        return results

    async def close(self) -> None:
        """Close the gRPC channel."""
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None

    def reset_cache(self) -> None:
        """Reset cached availability (for testing or reconnection)."""
        self._available = None
