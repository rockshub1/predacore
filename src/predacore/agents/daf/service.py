"""
gRPC Service implementation for the Dynamic Agent Fabric (DAF) Controller.
v1.0 implements Dynamic Agent Spawning and Management.
"""
import asyncio  # For locks
import collections
import logging
import multiprocessing  # For spawning agent processes
import os
from concurrent import futures
from typing import Optional
from uuid import uuid4

import grpc
from google.protobuf.struct_pb2 import Struct, Value
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from predacore._vendor.common.logging_utils import extract_trace_id, log_json

# Import generated protobuf code and stubs
from predacore._vendor.common.protos import (
    daf_pb2,
    daf_pb2_grpc,
    egm_pb2,  # For EGM request/response types
    egm_pb2_grpc,  # For EGM stub dependency (optional checks)
    wil_pb2,  # For status enum
    wil_pb2_grpc,  # For WIL stub dependency
)

from . import scheduler as daf_scheduler

# Import new registry classes and abstract bases
from .agent_registry import (
    AbstractAgentInstanceRegistry,
    AbstractAgentTypeRegistry,
    ActiveAgentInstanceRegistry,
    StaticAgentTypeRegistry,
)
from .self_optimization import SelfOptimizer
from .task_store import AbstractTaskStore, get_task_store


class _ServiceAlertSystem:
    """Minimal alert sink for self-optimization loops."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    async def send_alert(self, alert_type: str, message: str) -> None:
        self._logger.warning("[self-opt:%s] %s", alert_type, message)


def _agent_process_target(instance_id, type_id, config):
    """Module-level entry point for agent processes (must be picklable for macOS spawn)."""
    import logging as _log

    _log.basicConfig(
        level=_log.INFO, format=f"[Agent {instance_id}] %(levelname)s: %(message)s"
    )
    subprocess_logger = _log.getLogger(f"agent.{instance_id}")
    from .agent_process import agent_process_main

    try:
        subprocess_logger.info(f"Starting agent process (type={type_id})")
        agent_process_main(instance_id, type_id, config)
    except Exception as e:  # catch-all: agent process entry point
        subprocess_logger.error(f"Process failed: {str(e)}", exc_info=True)
        raise


class DynamicAgentFabricControllerService(
    daf_pb2_grpc.DynamicAgentFabricControllerServiceServicer
):
    """gRPC service implementation with process management"""

    """
    Implements the gRPC methods for the DAF Controller service.
    Handles dynamic agent spawning, retirement, and task dispatching.
    """

    def __init__(
        self,
        agent_type_registry: AbstractAgentTypeRegistry,
        agent_instance_registry: AbstractAgentInstanceRegistry,  # Use new instance registry
        wil_stub: wil_pb2_grpc.WorldInteractionLayerServiceStub,
        egm_stub: egm_pb2_grpc.EthicalGovernanceModuleServiceStub | None = None,
        logger: logging.Logger | None = None,
        task_store: Optional["AbstractTaskStore"] = None,
    ):
        self.agent_type_registry = (
            agent_type_registry  # Stores agent *type* definitions
        )
        self.agent_instance_registry = (
            agent_instance_registry  # Stores active agent *instances*
        )
        self.wil_stub = wil_stub
        self.egm_stub = egm_stub
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(
            "DynamicAgentFabricControllerService (v1.0 Dynamic) initialized."
        )
        # Store references to spawned processes
        self._agent_processes: dict[str, multiprocessing.Process] = {}
        # Simple in-memory task queues keyed by agent type (bounded)
        self._MAX_MAP_SIZE = 10_000

        self._queues: collections.OrderedDict[str, asyncio.Queue] = collections.OrderedDict()
        self._task_results: collections.OrderedDict[str, daf_pb2.TaskResultMessage] = collections.OrderedDict()
        self._waiters: collections.OrderedDict[str, asyncio.Future] = collections.OrderedDict()
        # Task store
        self._task_store: AbstractTaskStore = task_store or get_task_store()
        # Metrics (handle duplicate registration in tests)
        from .metrics_util import get_or_create_metric as _get_or_create_metric

        self._m_tasks = _get_or_create_metric(
            Counter, "daf_tasks_total", "DAF tasks processed", ["status"]
        )
        self._m_lat = _get_or_create_metric(
            Histogram, "daf_task_latency_seconds", "DAF task latency"
        )
        self._g_qdepth = _get_or_create_metric(
            Gauge, "daf_queue_depth", "DAF queue depth", ["agent_type"]
        )
        self._h_qwait = _get_or_create_metric(
            Histogram,
            "daf_queue_wait_seconds",
            "DAF queue wait time seconds",
            ["agent_type"],
        )
        self._throttled_noncritical = False
        self._retry_multiplier = 1.0
        self._self_opt_task: asyncio.Task | None = None
        self._self_optimizer: SelfOptimizer | None = None
        try:
            enabled = str(os.getenv("DAF_SELF_OPT_ENABLED", "1")).strip().lower()
            if enabled not in {"0", "false", "no", "off"}:
                interval = int(os.getenv("DAF_SELF_OPT_INTERVAL_SECONDS", "30"))
                self._self_optimizer = SelfOptimizer(
                    optimization_interval=max(5, interval)
                )
                self._self_optimizer.orchestrator = self
                self._self_optimizer.alert_system = _ServiceAlertSystem(self.logger)
                self.logger.info(
                    "DAF self-optimization enabled (interval=%ss)",
                    max(5, interval),
                )
        except (ValueError, TypeError, AttributeError, OSError) as exc:
            self.logger.warning("Failed to initialize self-optimizer: %s", exc)

    def _evict_oldest(self, od: collections.OrderedDict, key: str) -> None:
        """Insert *key* into an OrderedDict, evicting the oldest entry if at capacity."""
        if key in od:
            od.move_to_end(key)
            return
        if len(od) >= self._MAX_MAP_SIZE:
            od.popitem(last=False)

    @property
    def self_optimizer(self) -> SelfOptimizer | None:
        return self._self_optimizer

    async def start_self_optimization(self) -> None:
        if self._self_optimizer is None:
            return
        if self._self_opt_task and not self._self_opt_task.done():
            return
        self._self_opt_task = asyncio.create_task(
            self._self_optimizer.start_monitoring()
        )
        self.logger.info("Started DAF self-optimization background loop")

    async def stop_self_optimization(self) -> None:
        if self._self_optimizer is not None:
            await self._self_optimizer.stop_monitoring()
        if self._self_opt_task is not None:
            self._self_opt_task.cancel()
            try:
                await self._self_opt_task
            except asyncio.CancelledError:
                pass
            self._self_opt_task = None

    async def count_active_agents(self) -> int:
        try:
            instances = await self.agent_instance_registry.list_instances()
            return len(instances)
        except (RuntimeError, AttributeError, KeyError, OSError):
            return 0

    async def scale_agents(self) -> None:
        """Scale up by spawning one additional agent instance when possible."""
        try:
            agent_types = self.agent_type_registry.list_agent_types()
            if not agent_types:
                return
            target_type = str(agent_types[0].get("agent_type_id") or "").strip()
            if not target_type:
                return
            await self.SpawnAgent(
                daf_pb2.SpawnAgentRequest(agent_type_id=target_type),
                None,  # context is not used by SpawnAgent
            )
            self.logger.info(
                "Self-opt action scale_agents spawned type=%s", target_type
            )
        except (RuntimeError, ValueError, KeyError, AttributeError, grpc.RpcError) as exc:
            self.logger.warning("Self-opt scale_agents failed: %s", exc)

    async def scale_down_agents(self) -> None:
        """Scale down by retiring one idle agent instance."""
        try:
            instances = await self.agent_instance_registry.list_instances()
            idle = [
                i
                for i in instances
                if i.get("status")
                == daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_IDLE
            ]
            if not idle:
                return
            target = str(idle[0].get("instance_id") or "").strip()
            if not target:
                return
            await self.RetireAgent(
                daf_pb2.RetireAgentRequest(agent_instance_id=target),
                None,  # context is not used by RetireAgent on happy path
            )
            self.logger.info("Self-opt action scale_down_agents retired id=%s", target)
        except (RuntimeError, ValueError, KeyError, AttributeError, grpc.RpcError) as exc:
            self.logger.warning("Self-opt scale_down_agents failed: %s", exc)

    async def optimize_network(self) -> None:
        self.logger.info("Self-opt action optimize_network applied")

    async def throttle_noncritical(self) -> None:
        self._throttled_noncritical = True
        self.logger.warning("Self-opt action throttle_noncritical applied")

    async def increase_retries(self) -> None:
        self._retry_multiplier = min(self._retry_multiplier + 0.25, 3.0)
        self.logger.info(
            "Self-opt action increase_retries multiplier=%.2f", self._retry_multiplier
        )

    async def warm_agent_pool(self) -> None:
        self.logger.info("Self-opt action warm_agent_pool requested")

    async def ListAgentTypes(
        self, request: daf_pb2.ListAgentTypesRequest, context: grpc.aio.ServicerContext
    ) -> daf_pb2.ListAgentTypesResponse:
        self.logger.debug("Received ListAgentTypes request.")
        try:
            # Use the type registry
            agent_types_data = self.agent_type_registry.list_agent_types()
            agent_type_messages = []
            for agent_dict in agent_types_data:
                config_struct = Struct()
                config_data = agent_dict.get("configuration", {})
                if isinstance(config_data, dict):
                    try:
                        config_struct.update(config_data)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            f"Could not convert configuration dict to Struct for agent {agent_dict.get('agent_type_id')}: {e}"
                        )
                desc = daf_pb2.AgentDescriptionMessage(
                    agent_type_id=agent_dict.get("agent_type_id", ""),
                    description=agent_dict.get("description", ""),
                    supported_actions=agent_dict.get("supported_actions", []),
                    required_tools=agent_dict.get("required_tools", []),
                    configuration=config_struct,
                )
                agent_type_messages.append(desc)
            return daf_pb2.ListAgentTypesResponse(agent_types=agent_type_messages)
        except Exception as e:  # catch-all: request handler boundary
            self.logger.error(f"Error listing agent types: {e}", exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, "Failed to list agent types")
            return daf_pb2.ListAgentTypesResponse()

    async def SpawnAgent(
        self, request: daf_pb2.SpawnAgentRequest, context: grpc.aio.ServicerContext
    ) -> daf_pb2.SpawnAgentResponse:
        agent_type_id = request.agent_type_id
        self.logger.info(f"Received SpawnAgent request for type: {agent_type_id}")

        # 1. Check if agent type exists using the type registry
        agent_type_info = self.agent_type_registry.get_agent_type(agent_type_id)
        if not agent_type_info:
            msg = f"Agent type '{agent_type_id}' not found in registry."
            self.logger.warning(msg)
            return daf_pb2.SpawnAgentResponse(success=False, error_message=msg)

        # 2. Generate unique ID for the instance
        agent_instance_id = str(uuid4())

        # 3. Spawn agent process using multiprocessing
        self.logger.info(
            f"Spawning agent instance {agent_instance_id} (type: {agent_type_id}) using multiprocessing"
        )
        spawn_successful = False
        try:
            process = multiprocessing.Process(
                target=_agent_process_target,
                args=(
                    agent_instance_id,
                    agent_type_id,
                    dict(request.initial_configuration),
                ),
                daemon=False,  # Non-daemon: agents get graceful shutdown, not SIGKILL
            )
            process.start()
            self._agent_processes[agent_instance_id] = process
            spawn_successful = process.is_alive()
            if not spawn_successful:
                self.logger.error(
                    f"Agent process {agent_instance_id} failed to start or exited immediately."
                )
        except (OSError, multiprocessing.ProcessError) as e:
            self.logger.error(
                f"Error spawning agent process {agent_instance_id}: {e}", exc_info=True
            )

        if not spawn_successful:
            msg = f"Failed to spawn agent instance of type '{agent_type_id}'."
            self.logger.error(msg)
            return daf_pb2.SpawnAgentResponse(success=False, error_message=msg)

        # 4. Register instance using the instance registry
        try:
            instance_config = (
                dict(request.initial_configuration)
                if request.initial_configuration
                else {}
            )
            instance_info = await self.agent_instance_registry.register_instance(
                instance_id=agent_instance_id,
                type_id=agent_type_id,
                config=instance_config,
            )
        except (RuntimeError, KeyError, AttributeError, OSError) as e:
            self.logger.error(
                f"Failed to register spawned agent instance {agent_instance_id}: {e}",
                exc_info=True,
            )
            # Terminate the spawned process if registration fails
            if agent_instance_id in self._agent_processes:
                try:
                    self._agent_processes[agent_instance_id].terminate()
                    self._agent_processes[agent_instance_id].join(
                        timeout=1
                    )  # Wait briefly
                    del self._agent_processes[agent_instance_id]
                    self.logger.info(
                        f"Terminated process for failed registration: {agent_instance_id}"
                    )
                except (OSError, multiprocessing.ProcessError) as term_err:
                    self.logger.error(
                        f"Error terminating process {agent_instance_id} after failed registration: {term_err}"
                    )
            return daf_pb2.SpawnAgentResponse(
                success=False, error_message=f"Failed to register instance: {e}"
            )

        # 5. Create response message
        instance_msg = daf_pb2.AgentInstanceMessage(
            agent_instance_id=instance_info["instance_id"],
            agent_type_id=instance_info["type_id"],
            status=instance_info["status"],
            last_heartbeat=instance_info["last_heartbeat"],
            # config is not part of the message, only stored internally
        )
        self.logger.info(
            f"Successfully spawned and registered agent instance {agent_instance_id}"
        )
        return daf_pb2.SpawnAgentResponse(success=True, agent_instance=instance_msg)

    async def RetireAgent(
        self, request: daf_pb2.RetireAgentRequest, context: grpc.aio.ServicerContext
    ) -> daf_pb2.RetireAgentResponse:
        agent_instance_id = request.agent_instance_id
        self.logger.info(
            f"Received RetireAgent request for instance: {agent_instance_id}"
        )

        # 1. Check if instance exists in instance registry
        instance_info = await self.agent_instance_registry.get_instance(
            agent_instance_id
        )
        if not instance_info:
            msg = f"Agent instance '{agent_instance_id}' not found."
            self.logger.warning(msg)
            return daf_pb2.RetireAgentResponse(success=False, error_message=msg)

        # 2. Terminate agent process
        termination_successful = False
        process = self._agent_processes.get(agent_instance_id)
        if process:
            self.logger.info(
                f"Terminating agent process for instance {agent_instance_id} (PID: {process.pid})"
            )
            try:
                # Send SIGTERM first for graceful shutdown (if agent handles it)
                process.terminate()
                process.join(timeout=5)  # Wait for graceful exit
                if process.is_alive():
                    self.logger.warning(
                        f"Agent process {agent_instance_id} did not terminate gracefully, sending SIGKILL."
                    )
                    process.kill()  # Force kill if still alive
                    process.join(timeout=1)
                if not process.is_alive():
                    termination_successful = True
                    del self._agent_processes[
                        agent_instance_id
                    ]  # Remove from tracking dict
                else:
                    self.logger.error(
                        f"Failed to terminate agent process {agent_instance_id} even with SIGKILL."
                    )
            except (OSError, multiprocessing.ProcessError) as e:
                self.logger.error(
                    f"Error terminating agent process {agent_instance_id}: {e}",
                    exc_info=True,
                )
        else:
            self.logger.warning(
                f"No active process found for agent instance {agent_instance_id} to terminate."
            )
            # If no process exists, consider retirement successful if it's unregistered
            termination_successful = True

        if not termination_successful:
            msg = f"Failed to terminate agent instance {agent_instance_id}."
            self.logger.error(msg)
            # Don't unregister if termination failed? Or mark as ERROR?
            return daf_pb2.RetireAgentResponse(success=False, error_message=msg)

        # 3. Unregister instance using the instance registry
        unregistered = await self.agent_instance_registry.unregister_instance(
            agent_instance_id
        )
        if not unregistered:
            # This might happen if it was already unregistered between check and now
            self.logger.warning(
                f"Instance {agent_instance_id} was not found during unregistration, possibly already retired."
            )
            # Return success=True as the end state (retired) is achieved.
            return daf_pb2.RetireAgentResponse(success=True)

        self.logger.info(
            f"Successfully retired and unregistered agent instance {agent_instance_id}"
        )
        return daf_pb2.RetireAgentResponse(success=True)

    async def DispatchTask(
        self, request: daf_pb2.TaskAssignmentMessage, context: grpc.aio.ServicerContext
    ) -> daf_pb2.TaskResultMessage:
        task_id = request.task_id or str(uuid4())
        trace_id = extract_trace_id(request.context, fallback=task_id)
        self.logger.info(
            f"[trace={trace_id}] Received DispatchTask request {task_id} targeting: {request.WhichOneof('assignment_target')}"
        )
        try:
            await self._task_store.set_task(
                task_id, {"plan_step_id": request.plan_step_id, "trace_id": trace_id}
            )
        except (RuntimeError, OSError, KeyError, TypeError):
            pass

        agent_type_id = None
        agent_instance_id = None
        agent_type_info = None  # Agent type info from type registry
        agent_instance_info = None  # Active agent instance info from instance registry

        # 1. Determine Target Agent (Type or Instance)
        try:
            if request.HasField("agent_instance_id"):
                agent_instance_id = request.agent_instance_id
                # Use instance registry to get info
                agent_instance_info = await self.agent_instance_registry.get_instance(
                    agent_instance_id
                )
                if not agent_instance_info:
                    raise ValueError(
                        f"Target agent instance '{agent_instance_id}' not found or not active."
                    )
                agent_type_id = agent_instance_info["type_id"]
                # Use type registry to get static info
                agent_type_info = self.agent_type_registry.get_agent_type(agent_type_id)
                if not agent_type_info:
                    raise ValueError(
                        f"Agent type '{agent_type_id}' for instance '{agent_instance_id}' not found in type registry."
                    )
                if (
                    agent_instance_info["status"]
                    != daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_IDLE
                ):
                    self.logger.warning(
                        f"Target agent instance {agent_instance_id} is not IDLE "
                        f"(status: {agent_instance_info['status']}). "
                        f"Rejecting task {task_id}."
                    )
                    raise ValueError(
                        f"Agent instance {agent_instance_id} is busy "
                        f"(status: {agent_instance_info['status']}). "
                        f"Retry later or target a different instance."
                    )
                self.logger.info(
                    f"[trace={trace_id}] Targeting specific agent instance {agent_instance_id} (type: {agent_type_id})"
                )

            elif request.HasField("agent_type_id"):
                agent_type_id = request.agent_type_id
                agent_type_info = self.agent_type_registry.get_agent_type(agent_type_id)
                if not agent_type_info:
                    raise ValueError(
                        f"Agent type '{agent_type_id}' not found in type registry."
                    )
                # Find an idle instance or spawn a new one
                agent_instance_id = (
                    await self.agent_instance_registry.find_idle_instance(agent_type_id)
                )
                if not agent_instance_id:
                    self.logger.info(
                        f"[trace={trace_id}] No idle instance found for type {agent_type_id}. Spawning new one for task {task_id}."
                    )
                    # Call SpawnAgent internally (or replicate its core logic)
                    spawn_request = daf_pb2.SpawnAgentRequest(
                        agent_type_id=agent_type_id
                    )
                    spawn_response = await self.SpawnAgent(
                        spawn_request, context
                    )  # Reuse SpawnAgent logic
                    if not spawn_response.success or not spawn_response.HasField(
                        "agent_instance"
                    ):
                        raise ValueError(
                            f"Failed to spawn required agent instance of type {agent_type_id} for task {task_id}: {spawn_response.error_message}"
                        )
                    agent_instance_id = spawn_response.agent_instance.agent_instance_id
                    self.logger.info(
                        f"[trace={trace_id}] Spawned new instance {agent_instance_id} for task {task_id}"
                    )
                self.logger.info(
                    f"[trace={trace_id}] Selected agent instance {agent_instance_id} for type {agent_type_id}"
                )

            elif request.HasField("required_capability"):
                capability = request.required_capability
                agent_type_id = self.agent_type_registry.find_agent_for_capability(
                    capability
                )
                if not agent_type_id:
                    raise ValueError(
                        f"No agent type found for capability: {capability}"
                    )
                agent_type_info = self.agent_type_registry.get_agent_type(agent_type_id)
                if not agent_type_info:
                    raise ValueError(
                        f"Agent type '{agent_type_id}' found for capability but not in type registry."
                    )
                # Find an idle instance or spawn a new one
                agent_instance_id = (
                    await self.agent_instance_registry.find_idle_instance(agent_type_id)
                )
                if not agent_instance_id:
                    self.logger.info(
                        f"[trace={trace_id}] No idle instance found for type {agent_type_id} (capability: {capability}). Spawning new one for task {task_id}."
                    )
                    # Call SpawnAgent internally
                    spawn_request = daf_pb2.SpawnAgentRequest(
                        agent_type_id=agent_type_id
                    )
                    spawn_response = await self.SpawnAgent(spawn_request, context)
                    if not spawn_response.success or not spawn_response.HasField(
                        "agent_instance"
                    ):
                        raise ValueError(
                            f"Failed to spawn required agent instance of type {agent_type_id} for task {task_id} (capability: {capability}): {spawn_response.error_message}"
                        )
                    agent_instance_id = spawn_response.agent_instance.agent_instance_id
                    self.logger.info(
                        f"[trace={trace_id}] Spawned new instance {agent_instance_id} for task {task_id}"
                    )
                self.logger.info(
                    f"[trace={trace_id}] Selected agent instance {agent_instance_id} for capability '{capability}' (type: {agent_type_id})"
                )
            else:
                raise ValueError(
                    "Task assignment must specify required_capability, agent_type_id, or agent_instance_id."
                )

            # Update instance status to BUSY
            await self.agent_instance_registry.update_instance_status(
                agent_instance_id,
                daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_BUSY,
                {"task_id": task_id, "plan_step_id": request.plan_step_id},
            )

        except ValueError as ve:
            self.logger.error(
                f"[trace={trace_id}] Agent selection/assignment failed for task {task_id}: {ve}"
            )
            return daf_pb2.TaskResultMessage(
                task_id=task_id,
                plan_step_id=request.plan_step_id,
                status=wil_pb2.InteractionStatus.INTERACTION_STATUS_ERROR,
                error_message=str(ve),
            )
        except (RuntimeError, KeyError, AttributeError, TypeError, grpc.RpcError) as e:
            self.logger.error(
                f"Unexpected error during agent selection/assignment for task {task_id}: {e}",
                exc_info=True,
            )
            return daf_pb2.TaskResultMessage(
                task_id=task_id,
                plan_step_id=request.plan_step_id,
                status=wil_pb2.InteractionStatus.INTERACTION_STATUS_ERROR,
                error_message="Internal error during agent selection.",
            )

        # 2. Map Task to WIL Tool (based on agent type)
        required_tools = agent_type_info.get("required_tools", [])
        tool_id = required_tools[0] if required_tools else None
        if not tool_id:
            self.logger.error(
                f"[trace={trace_id}] No WIL tool mapping defined for agent type '{agent_type_id}'"
            )
            # Reset agent status to IDLE if tool mapping fails
            await self.agent_instance_registry.update_instance_status(
                agent_instance_id,
                daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_IDLE,
            )
            return daf_pb2.TaskResultMessage(
                task_id=task_id,
                plan_step_id=request.plan_step_id,
                status=wil_pb2.InteractionStatus.INTERACTION_STATUS_ERROR,
                error_message=f"Internal configuration error: No tool mapping for agent '{agent_type_id}'.",
            )

        # 3. Prepare WIL Parameters
        wil_parameters = request.parameters

        # 4. EGM Pre-check (enterprise mode only)
        egm_mode = os.getenv("EGM_MODE", "off").strip().lower()
        if egm_mode != "off" and self.egm_stub is not None:
            try:
                desc = Struct()
                desc.update(
                    {
                        "action_type": "DISPATCH_TASK",
                        "agent_type": agent_type_id,
                        "tool_id": tool_id,
                        "task_id": task_id,
                    }
                )
                egm_req = egm_pb2.CheckActionComplianceRequest(
                    action_description=desc,
                )
                egm_res = await self.egm_stub.CheckActionCompliance(egm_req)
                if not egm_res.is_compliant:
                    if egm_mode == "strict":
                        self.logger.warning(
                            "[trace=%s] Task dispatch blocked by EGM compliance",
                            trace_id,
                        )
                        await self.agent_instance_registry.update_instance_status(
                            agent_instance_id,
                            daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_IDLE,
                        )
                        return daf_pb2.TaskResultMessage(
                            task_id=task_id,
                            plan_step_id=request.plan_step_id,
                            status=wil_pb2.InteractionStatus.INTERACTION_STATUS_ERROR,
                            error_message="Task dispatch blocked by EGM compliance check",
                        )
                    else:
                        self.logger.info(
                            "[trace=%s] EGM flagged dispatch (mode=%s, continuing)",
                            trace_id,
                            egm_mode,
                        )
            except grpc.RpcError as e:
                if egm_mode == "strict":
                    self.logger.error(
                        "[trace=%s] EGM pre-check failed: %s", trace_id, e
                    )
                    await self.agent_instance_registry.update_instance_status(
                        agent_instance_id,
                        daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_IDLE,
                    )
                    return daf_pb2.TaskResultMessage(
                        task_id=task_id,
                        plan_step_id=request.plan_step_id,
                        status=wil_pb2.InteractionStatus.INTERACTION_STATUS_ERROR,
                        error_message=f"EGM compliance check failed: {e}",
                    )
                else:
                    self.logger.warning(
                        "[trace=%s] EGM pre-check error (mode=%s, continuing): %s",
                        trace_id,
                        egm_mode,
                        e,
                    )

        # Decide execution mode: synchronous vs. pull loop
        use_pull = False
        try:
            if request.context and "use_pull_loop" in request.context.fields:
                use_pull = request.context.fields["use_pull_loop"].bool_value
        except (KeyError, AttributeError):
            use_pull = False

        # 5a. Pull loop path: enqueue and wait for result
        if use_pull:
            import asyncio as _asyncio

            queue_key = (
                f"instance:{agent_instance_id}"
                if request.HasField("agent_instance_id")
                else agent_type_id
            )
            if queue_key not in self._queues:
                self._evict_oldest(self._queues, queue_key)
                self._queues[queue_key] = _asyncio.Queue()
            # Annotate enqueue timestamp in context
            try:
                ctx = request.context or Struct()
                from time import time as _now

                ctx.update({"enqueued_at": float(_now())})
                request.context.CopyFrom(ctx)
            except (ValueError, TypeError):
                pass
            await self._queues[queue_key].put(request)
            try:
                self._g_qdepth.labels(agent_type=agent_type_id).set(
                    self._queues[queue_key].qsize()
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
            # Create a future and wait for ReportTaskResult
            loop = _asyncio.get_running_loop()
            fut = loop.create_future()
            self._evict_oldest(self._waiters, task_id)
            self._waiters[task_id] = fut
            start_t = loop.time()
            try:
                res: daf_pb2.TaskResultMessage = await _asyncio.wait_for(
                    fut, timeout=30
                )
                try:
                    self._m_tasks.labels(status="completed").inc()
                    self._m_lat.observe(loop.time() - start_t)
                except (AttributeError, KeyError, TypeError, ValueError):
                    pass
                return res
            except _asyncio.TimeoutError:
                self.logger.warning(
                    f"Pull loop timeout for task {task_id}; falling back to synchronous execution"
                )
            finally:
                self._waiters.pop(task_id, None)

        # 5b. Construct and Invoke WIL Request (synchronous path)
        self.logger.info(
            f"[trace={trace_id}] Dispatching task {task_id} to WIL tool '{tool_id}' via agent instance '{agent_instance_id}' (type: {agent_type_id})"
        )
        try:
            log_json(
                self.logger,
                logging.INFO,
                "daf.dispatch.start",
                trace_id=trace_id,
                task_id=task_id,
                tool_id=tool_id,
                agent_instance_id=agent_instance_id,
            )
        except (TypeError, ValueError, AttributeError):
            pass
        task_result = None
        try:
            # (WIL invocation logic remains largely the same)
            if tool_id == "python_sandbox":  # Or other code execution tools
                code_to_execute = wil_parameters.fields.get("code", None)
                if not code_to_execute or not code_to_execute.string_value:
                    raise ValueError(
                        f"Missing 'code' parameter for {tool_id} execution"
                    )
                # Ensure context carries a trace_id
                _ctx = request.context or Struct()
                try:
                    if not (
                        hasattr(_ctx, "fields")
                        and "trace_id" in _ctx.fields
                        and _ctx.fields["trace_id"].string_value
                    ):
                        _ctx.update({"trace_id": trace_id})
                except (ValueError, TypeError):
                    pass
                wil_request = wil_pb2.CodeExecutionRequestMessage(
                    request_id=task_id,
                    code=code_to_execute.string_value,
                    input_args=wil_parameters.fields.get("input_args"),
                    timeout_seconds=int(
                        wil_parameters.fields.get(
                            "timeout_seconds", Value(number_value=60)
                        ).number_value
                    ),
                    context=_ctx,
                )
                wil_result_proto = await self.wil_stub.ExecuteCode(wil_request)
                task_result = daf_pb2.TaskResultMessage(
                    task_id=task_id,
                    plan_step_id=request.plan_step_id,
                    status=wil_result_proto.status,
                    output=wil_result_proto.result,
                    error_message=wil_result_proto.error_message,
                    agent_type_id_used=agent_type_id,
                    agent_instance_id_used=agent_instance_id,
                )
                try:
                    await self._task_store.set_result(
                        task_id,
                        {
                            "status": int(task_result.status),
                            "agent": agent_instance_id,
                        },
                    )
                except (RuntimeError, OSError, KeyError, TypeError):
                    pass
            else:  # General tool execution
                _ctx = request.context or Struct()
                try:
                    if not (
                        hasattr(_ctx, "fields")
                        and "trace_id" in _ctx.fields
                        and _ctx.fields["trace_id"].string_value
                    ):
                        _ctx.update({"trace_id": trace_id})
                except (ValueError, TypeError):
                    pass
                wil_request = wil_pb2.InteractionRequestMessage(
                    request_id=task_id,
                    tool_id=tool_id,
                    parameters=wil_parameters,
                    context=_ctx,
                )
                wil_result_proto = await self.wil_stub.ExecuteTool(wil_request)
                task_result = daf_pb2.TaskResultMessage(
                    task_id=task_id,
                    plan_step_id=request.plan_step_id,
                    status=wil_result_proto.status,
                    output=wil_result_proto.output,
                    error_message=wil_result_proto.error_message,
                    agent_type_id_used=agent_type_id,
                    agent_instance_id_used=agent_instance_id,
                )
            self.logger.info(
                f"Task {task_id} completed with status {task_result.status}"
            )
            try:
                log_json(
                    self.logger,
                    logging.INFO,
                    "daf.dispatch.end",
                    trace_id=trace_id,
                    task_id=task_id,
                    status=int(task_result.status),
                )
            except (TypeError, ValueError, AttributeError):
                pass

        except grpc.aio.AioRpcError as e:
            self.logger.error(
                f"[trace={trace_id}] gRPC error calling WIL for task {task_id}: {e.details()}",
                exc_info=True,
            )
            task_result = daf_pb2.TaskResultMessage(
                task_id=task_id,
                plan_step_id=request.plan_step_id,
                status=wil_pb2.InteractionStatus.INTERACTION_STATUS_ERROR,
                error_message=f"WIL communication error: {e.details()}",
                agent_type_id_used=agent_type_id,
                agent_instance_id_used=agent_instance_id,
            )
        except ValueError as ve:
            self.logger.error(
                f"Value error dispatching task {task_id}: {ve}", exc_info=True
            )
            task_result = daf_pb2.TaskResultMessage(
                task_id=task_id,
                plan_step_id=request.plan_step_id,
                status=wil_pb2.InteractionStatus.INTERACTION_STATUS_ERROR,
                error_message=f"Parameter error: {ve}",
                agent_type_id_used=agent_type_id,
                agent_instance_id_used=agent_instance_id,
            )
        except Exception as e:  # catch-all: request handler boundary
            self.logger.error(
                f"[trace={trace_id}] Unexpected error dispatching task {task_id} to WIL: {e}",
                exc_info=True,
            )
            task_result = daf_pb2.TaskResultMessage(
                task_id=task_id,
                plan_step_id=request.plan_step_id,
                status=wil_pb2.InteractionStatus.INTERACTION_STATUS_ERROR,
                error_message=f"Unexpected error during task dispatch: {e}",
                agent_type_id_used=agent_type_id,
                agent_instance_id_used=agent_instance_id,
            )
        finally:
            # Reset agent status to IDLE after task completion/failure
            if agent_instance_id:
                await self.agent_instance_registry.update_instance_status(
                    agent_instance_id,
                    daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_IDLE,
                )

        try:
            if (
                task_result.status
                == wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS
            ):
                self._m_tasks.labels(status="completed").inc()
            else:
                self._m_tasks.labels(status="error").inc()
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        return task_result

    # --- Pull Loop RPCs ---
    async def RegisterAgentInstance(
        self,
        request: daf_pb2.RegisterAgentInstanceRequest,
        context: grpc.aio.ServicerContext,
    ) -> daf_pb2.RegisterAgentInstanceResponse:
        try:
            inst = request.agent_instance
            # Ensure type queue exists
            if inst.agent_type_id not in self._queues:
                import asyncio as _asyncio

                self._evict_oldest(self._queues, inst.agent_type_id)
                self._queues[inst.agent_type_id] = _asyncio.Queue()
            # Register instance in registry
            await self.agent_instance_registry.register_instance(
                inst.agent_instance_id, inst.agent_type_id, {}
            )
            return daf_pb2.RegisterAgentInstanceResponse(success=True)
        except Exception as e:  # catch-all: request handler boundary
            self.logger.error(f"RegisterAgentInstance failed: {e}", exc_info=True)
            return daf_pb2.RegisterAgentInstanceResponse(
                success=False, error_message=str(e)
            )

    async def AgentHeartbeat(
        self, request: daf_pb2.AgentHeartbeatMessage, context: grpc.aio.ServicerContext
    ) -> daf_pb2.AgentHeartbeatResponse:
        try:
            # Mark as IDLE on heartbeat for simplicity
            await self.agent_instance_registry.update_instance_status(
                request.agent_instance_id,
                daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_IDLE,
            )
            return daf_pb2.AgentHeartbeatResponse(success=True)
        except Exception as e:  # catch-all: request handler boundary
            self.logger.warning(f"Heartbeat update failed: {e}")
            return daf_pb2.AgentHeartbeatResponse(success=False)

    async def GetNextTask(
        self, request: daf_pb2.GetNextTaskRequest, context: grpc.aio.ServicerContext
    ) -> daf_pb2.GetNextTaskResponse:
        # Determine instance type
        inst = await self.agent_instance_registry.get_instance(
            request.agent_instance_id
        )
        if not inst:
            return daf_pb2.GetNextTaskResponse(has_task=False)
        q = self._queues.get(f"instance:{request.agent_instance_id}")
        if not q or q.empty():
            q = self._queues.get(inst["type_id"])
        if not q or q.empty():
            return daf_pb2.GetNextTaskResponse(has_task=False)
        task: daf_pb2.TaskAssignmentMessage = await q.get()
        # Observe wait time
        try:
            enq = None
            if task.context and "enqueued_at" in task.context.fields:
                v = task.context.fields["enqueued_at"]
                enq = float(v.number_value) if v.HasField("number_value") else None
            if enq is not None:
                import time as _t

                self._h_qwait.labels(agent_type=inst["type_id"]).observe(
                    max(0.0, _t.time() - enq)
                )
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        try:
            self._g_qdepth.labels(agent_type=inst["type_id"]).set(q.qsize())
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        return daf_pb2.GetNextTaskResponse(has_task=True, task=task)

    async def ReportTaskResult(
        self,
        request: daf_pb2.ReportTaskResultRequest,
        context: grpc.aio.ServicerContext,
    ) -> daf_pb2.ReportTaskResultResponse:
        try:
            result = request.result
            self._evict_oldest(self._task_results, result.task_id)
            self._task_results[result.task_id] = result
            # Reset instance state
            if result.agent_instance_id_used:
                await self.agent_instance_registry.update_instance_status(
                    result.agent_instance_id_used,
                    daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_IDLE,
                )
            # Fulfill waiter if any
            fut = self._waiters.get(result.task_id)
            if fut and not fut.done():
                fut.set_result(result)
            return daf_pb2.ReportTaskResultResponse(success=True)
        except Exception as e:  # catch-all: request handler boundary
            self.logger.error(f"ReportTaskResult failed: {e}")
            return daf_pb2.ReportTaskResultResponse(success=False)


# Example function to start the server
async def serve():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Instantiate both registries
    agent_type_registry = StaticAgentTypeRegistry(logger=logger)
    agent_instance_registry = ActiveAgentInstanceRegistry(logger=logger)

    # Setup stubs
    _use_tls = os.getenv("GRPC_USE_TLS", "").lower() in ("1", "true")
    _creds = grpc.ssl_channel_credentials() if _use_tls else None

    def _channel(addr: str):
        return grpc.aio.secure_channel(addr, _creds) if _use_tls else grpc.aio.insecure_channel(addr)

    wil_channel = _channel(os.getenv("WIL_ADDRESS", "localhost:50054"))
    wil_stub = wil_pb2_grpc.WorldInteractionLayerServiceStub(wil_channel)

    # EGM channel only created when enabled (enterprise mode)
    egm_mode = os.getenv("EGM_MODE", "off").strip().lower()
    egm_channel = None
    egm_stub = None
    if egm_mode != "off":
        egm_channel = _channel(
            os.getenv("EGM_ADDRESS", "localhost:50053")
        )
        egm_stub = egm_pb2_grpc.EthicalGovernanceModuleServiceStub(egm_channel)
        logger.info("DAF EGM compliance enabled (mode=%s)", egm_mode)
    else:
        logger.info("DAF EGM compliance disabled (EGM_MODE=off)")

    # Task store (memory/redis)
    task_store = get_task_store()

    # Pass both registries to the service
    service_impl = DynamicAgentFabricControllerService(
        agent_type_registry=agent_type_registry,
        agent_instance_registry=agent_instance_registry,
        wil_stub=wil_stub,
        egm_stub=egm_stub,
        logger=logger,
        task_store=task_store,
    )
    daf_pb2_grpc.add_DynamicAgentFabricControllerServiceServicer_to_server(
        service_impl, server
    )
    listen_addr = os.getenv("DAF_LISTEN_ADDR", "[::]:50055")
    if os.getenv("GRPC_USE_TLS", "").lower() in ("1", "true"):
        _tls_cert = os.getenv("GRPC_TLS_CERT_PATH", "")
        _tls_key = os.getenv("GRPC_TLS_KEY_PATH", "")
        if _tls_cert and _tls_key:
            with open(_tls_cert, "rb") as _cf, open(_tls_key, "rb") as _kf:
                _creds = grpc.ssl_server_credentials([(_kf.read(), _cf.read())])
            server.add_secure_port(listen_addr, _creds)
            logger.info("gRPC server using TLS on %s", listen_addr)
        else:
            logger.warning("GRPC_USE_TLS set but GRPC_TLS_CERT_PATH/GRPC_TLS_KEY_PATH missing; falling back to insecure port")
            server.add_insecure_port(listen_addr)
    else:
        server.add_insecure_port(listen_addr)
    logger.info(
        f"Starting DynamicAgentFabricControllerService (v1.0 Dynamic) on {listen_addr}"
    )
    await server.start()
    # Metrics server
    try:
        port = int(os.getenv("DAF_METRICS_PORT", "8012"))
        start_http_server(port)
        logger.info(f"DAF Prometheus metrics exposed on :{port}")
    except (ValueError, TypeError, OSError) as e:
        logger.warning(f"Failed to start DAF metrics server: {e}")

    # Optional background LLM scheduler (disabled unless LLM_SCHEDULER_ENABLED=1)
    async def _scheduler_loop():
        if os.getenv("LLM_SCHEDULER_ENABLED") != "1":
            return
        while True:
            try:
                await daf_scheduler.scheduler_tick(service_impl, egm_stub, logger)
            except Exception as e:  # catch-all: main loop boundary
                logger.debug(f"Scheduler tick error: {e}")
            await asyncio.sleep(15)

    if os.getenv("LLM_SCHEDULER_ENABLED") == "1":
        asyncio.create_task(_scheduler_loop())
    if service_impl.self_optimizer is not None:
        await service_impl.start_self_optimization()

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Stopping server...")
        await server.stop(0)
    finally:
        await service_impl.stop_self_optimization()
        await wil_channel.close()
        if egm_channel is not None:
            await egm_channel.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(serve())
