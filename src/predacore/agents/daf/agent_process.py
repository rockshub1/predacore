"""
Core implementation for agent processes spawned by DAF.
Each agent runs as an independent process with gRPC communication back to the controller.
"""
import asyncio
import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any

import grpc
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct, Value
from predacore._vendor.common.protos import daf_pb2, daf_pb2_grpc, wil_pb2, wil_pb2_grpc


def _value_to_text(value: Any) -> str:
    """Render protobuf Value payloads into plain text."""
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


def _struct_to_dict(struct_message: Any) -> dict[str, Any]:
    if struct_message is None:
        return {}
    try:
        parsed = MessageToDict(struct_message, preserving_proto_field_name=True)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


class AgentProcess:
    """
    Base class for all agent processes with core lifecycle management.
    """

    def __init__(self, instance_id: str, type_id: str, config: dict[str, Any]):
        self.instance_id = instance_id
        self.type_id = type_id
        self.config = config
        self.logger = self._setup_logging()
        self.running = False
        self._unified_memory = self._setup_unified_memory()
        self._rich_runtime_ready = False
        self._rich_runtime_failed = False
        self._rich_config = None
        self._tool_ctx = None
        self._agent_engine = None
        self._setup_grpc()

    def _setup_logging(self) -> logging.Logger:
        """Configure process-specific logging"""
        logger = logging.getLogger(f"AgentProcess.{self.type_id}.{self.instance_id}")
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                f"[Agent {self.instance_id}] %(asctime)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(handler)
        return logger

    def _setup_grpc(self):
        """Establish gRPC connection back to controller."""
        target = os.getenv("DAF_CONTROLLER_ADDRESS", "localhost:50055")
        use_tls = os.getenv("GRPC_USE_TLS", "").lower() in ("1", "true")
        # Default to TLS for non-localhost targets to prevent MITM
        is_localhost = target.startswith("localhost:") or target.startswith("127.0.0.1:")
        if use_tls or not is_localhost:
            self.channel = grpc.secure_channel(target, grpc.ssl_channel_credentials())
        else:
            self.channel = grpc.insecure_channel(target)
        self.daf_stub = daf_pb2_grpc.DynamicAgentFabricControllerServiceStub(
            self.channel
        )
        self.wil_stub = wil_pb2_grpc.WorldInteractionLayerServiceStub(self.channel)

    def _setup_unified_memory(self):
        """Initialize the shared unified memory store when available."""
        try:
            try:
                from predacore.memory.store import UnifiedMemoryStore
            except ImportError:
                from src.predacore.memory.store import UnifiedMemoryStore  # type: ignore

            predacore_home = Path(
                os.getenv("PREDACORE_HOME")
                or os.getenv("PREDACORE_HOME")
                or (Path.home() / ".predacore")
            )
            db_path = predacore_home / "memory" / "unified_memory.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            return UnifiedMemoryStore(str(db_path))
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            self.logger.warning("Unified memory unavailable in DAF agent: %s", exc)
            return None

    def _resolve_home_dir(self, context_data: dict[str, Any], params: dict[str, Any]) -> str:
        explicit = str(
            context_data.get("home_dir")
            or params.get("home_dir")
            or self.config.get("home_dir")
            or ""
        ).strip()
        if explicit:
            return explicit
        return str(Path.home() / ".predacore")

    def _ensure_rich_runtime(self, home_dir: str) -> bool:
        """Bootstrap the same direct-provider collaboration runtime used by PredaCore."""
        if self._rich_runtime_ready:
            return True
        if self._rich_runtime_failed:
            return False
        try:
            try:
                from predacore.config import load_config
                from predacore.tools.subsystem_init import SubsystemFactory
                from predacore.tools.handlers import HANDLER_MAP, ToolContext
                from predacore.agents.engine import AgentEngine
            except ImportError:
                from src.predacore.config import load_config  # type: ignore
                from src.predacore.tools.subsystem_init import SubsystemFactory  # type: ignore
                from src.predacore.tools.handlers import HANDLER_MAP, ToolContext  # type: ignore
                from src.predacore.agents.engine import AgentEngine  # type: ignore

            config_path = Path(home_dir) / "config.yaml"
            cfg = load_config(str(config_path) if config_path.exists() else None)
            bundle = SubsystemFactory.create_all(
                cfg,
                skip_cli_providers=True,
                home_dir=home_dir,
            )
            if bundle.llm_for_collab is None:
                raise RuntimeError("No direct collaboration LLM available")

            tool_ctx = ToolContext(
                config=cfg,
                memory={},
                desktop_operator=bundle.desktop_operator,
                unified_memory=bundle.unified_memory,
                memory_service=bundle.memory_service,
                mcts_planner=bundle.mcts_planner,
                llm_for_collab=bundle.llm_for_collab,
                voice=bundle.voice,
                openclaw_runtime=bundle.openclaw_runtime,
                openclaw_enabled=bundle.openclaw_enabled,
                sandbox=bundle.sandbox,
                docker_sandbox=bundle.docker_sandbox,
                sandbox_pool=bundle.sandbox_pool,
                resolve_user_id=lambda args: str(args.get("user_id") or "default"),
            )
            self._rich_config = cfg
            self._tool_ctx = tool_ctx
            self._agent_engine = AgentEngine(
                llm=bundle.llm_for_collab,
                tool_ctx=tool_ctx,
                handler_map=HANDLER_MAP,
            )
            self._rich_runtime_ready = True
            self.logger.info("Rich DAF runtime initialized")
            return True
        except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as exc:
            self._rich_runtime_failed = True
            self.logger.warning("Rich DAF runtime unavailable: %s", exc)
            return False

    def _run_async(self, coro):
        """Run a small async helper from the synchronous worker loop."""
        return asyncio.run(coro)

    def _persist_team_memory(
        self,
        *,
        user_id: str,
        team_id: str,
        session_id: str | None,
        content: str,
        stage: str,
        role: str,
        pattern: str = "",
        memory_type: str = "task",
        importance: int = 2,
    ) -> None:
        if self._unified_memory is None or not team_id or not content.strip():
            return
        try:
            self._run_async(
                self._unified_memory.store(
                    content=content[:4000],
                    memory_type=memory_type,
                    importance=importance,
                    source="predacore.daf.agent_process",
                    tags=["daf", "team_memory", self.type_id],
                    metadata={
                        "source": "predacore.daf.agent_process",
                        "stage": stage,
                        "role": role,
                        "pattern": pattern,
                        "agent_type": self.type_id,
                    },
                    user_id=user_id,
                    session_id=session_id,
                    memory_scope="team",
                    team_id=team_id,
                    agent_id=self.instance_id,
                )
            )
        except (RuntimeError, OSError, ValueError, TypeError):
            self.logger.debug("Failed to persist DAF team memory", exc_info=True)

    def _recall_shared_memory(
        self,
        *,
        query: str,
        user_id: str,
        team_id: str,
        top_k: int = 4,
    ) -> str:
        if (
            self._unified_memory is None
            or not query.strip()
            or not team_id
            or not hasattr(self._unified_memory, "recall")
        ):
            return ""
        try:
            recalls = self._run_async(
                self._unified_memory.recall(
                    query=query,
                    user_id=user_id,
                    top_k=top_k,
                    scopes=["global", "team"],
                    team_id=team_id,
                )
            )
        except (RuntimeError, OSError, ValueError, TypeError):
            self.logger.debug("Failed to recall DAF shared memory", exc_info=True)
            return ""
        if not recalls:
            return ""
        lines = ["Shared memory:"]
        for mem, _score in recalls:
            meta = mem.get("metadata", {}) if isinstance(mem, dict) else {}
            label = (
                meta.get("stage")
                or meta.get("key")
                or mem.get("memory_type", "memory")
            )
            content = str(mem.get("content") or "").strip()
            if content:
                lines.append(f"- {label}: {content[:220]}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def _run_rich_agent_task(
        self,
        *,
        spec_payload: dict[str, Any],
        original_task: str,
        shared_memory: str,
        home_dir: str,
        role: str,
    ) -> str | None:
        """Execute a DAF task via the richer in-process typed agent loop."""
        if not self._ensure_rich_runtime(home_dir):
            return None
        if self._agent_engine is None:
            return None
        try:
            try:
                from predacore.agents.engine import (
                    compile_dynamic_agent_prompt,
                    create_dynamic_agent_spec,
                )
            except ImportError:
                from src.predacore.agents.engine import (  # type: ignore
                    compile_dynamic_agent_prompt,
                    create_dynamic_agent_spec,
                )

            spec = create_dynamic_agent_spec(
                base_type=str(spec_payload.get("base_type") or self.type_id),
                collaboration_role=str(
                    spec_payload.get("collaboration_role")
                    or spec_payload.get("role")
                    or role
                    or self.type_id
                ),
                specialization=str(
                    spec_payload.get("specialization")
                    or spec_payload.get("collaboration_role")
                    or role
                    or self.type_id
                ),
                mission=str(spec_payload.get("mission") or original_task or role),
                success_criteria=spec_payload.get("success_criteria") or (),
                output_schema=spec_payload.get("output_schema") or {},
                memory_scopes=spec_payload.get("memory_scopes") or ("global", "team"),
                allowed_tools=spec_payload.get("allowed_tools") or (),
                max_steps=spec_payload.get("max_steps"),
                temperature=spec_payload.get("temperature"),
            )
            system_prompt, compiled_user_prompt = compile_dynamic_agent_prompt(
                spec,
                shared_context=shared_memory,
                original_task=original_task,
            )
            result = self._run_async(
                self._agent_engine.run_task(
                    prompt=original_task,
                    agent_type=spec.base_type,
                    dynamic_spec=spec,
                    system_prompt_override=system_prompt,
                    prompt_override=compiled_user_prompt,
                )
            )
            output = str(result.get("output") or "").strip()
            if output:
                return output
            error = str(result.get("error") or "").strip()
            return error or None
        except (ImportError, OSError, RuntimeError, ValueError, TypeError, AttributeError):
            self.logger.debug("Rich DAF task execution failed", exc_info=True)
            return None

    def run(self):
        """Main process loop with signal handling"""
        self.running = True
        signal.signal(signal.SIGTERM, self._handle_termination)

        try:
            self.logger.info(f"Starting agent process (Type: {self.type_id})")
            self._register_with_controller()

            while self.running:
                self._heartbeat()
                self._process_tasks()
                time.sleep(1)  # Main loop delay

        except Exception as e:
            self.logger.error(f"Agent process failed: {e}", exc_info=True)
        finally:
            self._cleanup()

    def _register_with_controller(self):
        """Register this process instance with the DAF controller"""
        try:
            req = daf_pb2.RegisterAgentInstanceRequest(
                agent_instance=daf_pb2.AgentInstanceMessage(
                    agent_instance_id=self.instance_id,
                    agent_type_id=self.type_id,
                    status=daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_IDLE,
                )
            )
            response = self.daf_stub.RegisterAgentInstance(req)
            if not response.success:
                raise RuntimeError(f"Registration failed: {response.error_message}")
            self.logger.info("Successfully registered with DAF controller")
        except grpc.RpcError as e:
            raise RuntimeError(f"gRPC registration error: {e.details()}")

    def _heartbeat(self):
        """Send periodic heartbeat to controller"""
        try:
            self.daf_stub.AgentHeartbeat(
                daf_pb2.AgentHeartbeatMessage(
                    agent_instance_id=self.instance_id,
                    pid=os.getpid(),
                )
            )
        except grpc.RpcError as e:
            self.logger.warning(f"Heartbeat failed: {e.details()}")

    def _process_tasks(self):
        """Check for and process assigned tasks"""
        try:
            resp = self.daf_stub.GetNextTask(
                daf_pb2.GetNextTaskRequest(agent_instance_id=self.instance_id)
            )
            if not resp.has_task:
                return
            task = resp.task
            params = _struct_to_dict(task.parameters)
            context_data = _struct_to_dict(task.context)
            prompt = str(params.get("prompt") or params.get("query") or "").strip()
            role = str(context_data.get("role") or params.get("role") or self.type_id).strip()
            pattern = str(context_data.get("pattern") or params.get("pattern") or "").strip()
            user_id = str(context_data.get("user_id") or params.get("user_id") or "default").strip()
            team_id = str(context_data.get("team_id") or params.get("team_id") or "").strip()
            session_id_raw = str(context_data.get("session_id") or params.get("session_id") or "").strip()
            session_id = session_id_raw or None
            home_dir = self._resolve_home_dir(context_data, params)
            shared_memory = self._recall_shared_memory(
                query=prompt,
                user_id=user_id,
                team_id=team_id,
            )
            if team_id:
                kickoff_lines = [
                    f"DAF agent {self.instance_id} ({role}/{self.type_id}) picked up task {task.task_id}."
                ]
                if prompt:
                    kickoff_lines.append(f"Prompt: {prompt[:500]}")
                if shared_memory:
                    kickoff_lines.append(shared_memory)
                self._persist_team_memory(
                    user_id=user_id,
                    team_id=team_id,
                    session_id=session_id,
                    content="\n".join(kickoff_lines),
                    stage="task_start",
                    role=role,
                    pattern=pattern,
                    importance=2,
                )

            spec_payload: dict[str, Any] = {}
            raw_spec = str(params.get("agent_spec_json") or "").strip()
            if raw_spec:
                try:
                    parsed_spec = json.loads(raw_spec)
                    if isinstance(parsed_spec, dict):
                        spec_payload = parsed_spec
                except (TypeError, ValueError, json.JSONDecodeError):
                    self.logger.debug("Invalid DAF agent spec payload", exc_info=True)

            rich_output = None
            if spec_payload:
                rich_output = self._run_rich_agent_task(
                    spec_payload=spec_payload,
                    original_task=str(
                        params.get("original_task")
                        or params.get("query")
                        or prompt
                        or spec_payload.get("mission")
                        or ""
                    ),
                    shared_memory=shared_memory,
                    home_dir=home_dir,
                    role=role,
                )

            if rich_output:
                result = daf_pb2.TaskResultMessage(
                    task_id=task.task_id,
                    plan_step_id=task.plan_step_id,
                    status=wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS,
                    output=Value(string_value=rich_output),
                    agent_type_id_used=self.type_id,
                    agent_instance_id_used=self.instance_id,
                )
            elif "code" in params:
                code = str(params.get("code") or "")
                try:
                    ctx = task.context if hasattr(task, "context") else None
                    wil_resp = self.wil_stub.ExecuteCode(
                        wil_pb2.CodeExecutionRequestMessage(
                            request_id=task.task_id,
                            code=code,
                            timeout_seconds=int(params.get("timeout_seconds") or 60),
                            context=ctx,
                        )
                    )
                    result = daf_pb2.TaskResultMessage(
                        task_id=task.task_id,
                        plan_step_id=task.plan_step_id,
                        status=wil_resp.status,
                        output=wil_resp.result,
                        error_message=wil_resp.error_message,
                        agent_type_id_used=self.type_id,
                        agent_instance_id_used=self.instance_id,
                    )
                except grpc.RpcError as e:
                    result = daf_pb2.TaskResultMessage(
                        task_id=task.task_id,
                        plan_step_id=task.plan_step_id,
                        status=wil_pb2.InteractionStatus.INTERACTION_STATUS_ERROR,
                        error_message=f"WIL error: {e.details()}",
                        agent_type_id_used=self.type_id,
                        agent_instance_id_used=self.instance_id,
                    )
            elif self.type_id == "web_searcher":
                query = str(params.get("query") or prompt).strip()
                if not query:
                    result = daf_pb2.TaskResultMessage(
                        task_id=task.task_id,
                        plan_step_id=task.plan_step_id,
                        status=wil_pb2.InteractionStatus.INTERACTION_STATUS_ERROR,
                        error_message="web_searcher requires a query or prompt",
                        agent_type_id_used=self.type_id,
                        agent_instance_id_used=self.instance_id,
                    )
                else:
                    try:
                        tool_params = Struct()
                        tool_params.update({"query": query, "role": role})
                        wil_resp = self.wil_stub.ExecuteTool(
                            wil_pb2.InteractionRequestMessage(
                                request_id=task.task_id,
                                tool_id="google_search_api",
                                parameters=tool_params,
                                context=task.context if hasattr(task, "context") else None,
                            )
                        )
                        result = daf_pb2.TaskResultMessage(
                            task_id=task.task_id,
                            plan_step_id=task.plan_step_id,
                            status=wil_resp.status,
                            output=wil_resp.output,
                            error_message=wil_resp.error_message,
                            agent_type_id_used=self.type_id,
                            agent_instance_id_used=self.instance_id,
                        )
                    except grpc.RpcError as e:
                        result = daf_pb2.TaskResultMessage(
                            task_id=task.task_id,
                            plan_step_id=task.plan_step_id,
                            status=wil_pb2.InteractionStatus.INTERACTION_STATUS_ERROR,
                            error_message=f"WIL error: {e.details()}",
                            agent_type_id_used=self.type_id,
                            agent_instance_id_used=self.instance_id,
                        )
            else:
                fallback_lines = [
                    f"DAF agent {role} ({self.type_id}) received task {task.task_id}.",
                ]
                if prompt:
                    fallback_lines.append(f"Prompt: {prompt}")
                if shared_memory:
                    fallback_lines.append(shared_memory)
                fallback_lines.append(
                    "No dedicated executor is configured for this DAF agent type yet."
                )
                result = daf_pb2.TaskResultMessage(
                    task_id=task.task_id,
                    plan_step_id=task.plan_step_id,
                    status=wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS,
                    output=Value(string_value="\n\n".join(fallback_lines)),
                    agent_type_id_used=self.type_id,
                    agent_instance_id_used=self.instance_id,
                )

            result_text = _value_to_text(result.output) or str(result.error_message or "")
            if team_id and result_text:
                self._persist_team_memory(
                    user_id=user_id,
                    team_id=team_id,
                    session_id=session_id,
                    content=result_text,
                    stage=(
                        "task_error"
                        if result.status
                        != wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS
                        else "task_result"
                    ),
                    role=role,
                    pattern=pattern,
                    importance=2,
                    memory_type=(
                        "task"
                        if result.status
                        == wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS
                        else "conversation"
                    ),
                )
            self.daf_stub.ReportTaskResult(
                daf_pb2.ReportTaskResultRequest(result=result)
            )
        except grpc.RpcError as e:
            self.logger.warning(f"Polling failed: {e.details()}")

    def _handle_termination(self, signum, frame):
        """Handle graceful shutdown signals"""
        self.logger.info(f"Received termination signal {signum}")
        self.running = False

    def _cleanup(self):
        """Cleanup resources before exit"""
        self.logger.info("Cleaning up before shutdown")
        self.channel.close()


def agent_process_main(instance_id: str, type_id: str, config: dict[str, Any]):
    """
    Entry point for spawned agent processes.
    """
    agent = AgentProcess(instance_id, type_id, config)
    agent.run()


if __name__ == "__main__":
    # For testing individual agent processes
    test_config = {"test_param": "value", "timeout": 30}
    agent_process_main("test_instance", "test_agent", test_config)
