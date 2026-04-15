"""
JARVIS Agent Engine — Unified agent orchestration.

Merges the best of three systems:
- Claude Code:  Typed agents with scoped tool access + specialized prompts
- DAF:          Agent registry, capability routing, task store, self-optimization
- JARVIS:       32 real tools, memory, screen vision, Android, voice

Every agent type gets:
    1. A specialized system prompt (defines its personality/role)
    2. A scoped tool allowlist (can only use tools relevant to its job)
    3. Capability tags (for automatic routing)
    4. Performance tracking (latency, success rate)

Usage:
    engine = AgentEngine(llm=llm_interface, tool_ctx=tool_context)
    result = await engine.run_task("Research Python async patterns", agent_type="researcher")
    result = await engine.run_team(
        "Build a web scraper",
        pattern="fan_out",
        agent_types=["researcher", "coder"],
    )
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent Type Definitions — Real tools, real prompts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentType:
    """Immutable definition of an agent type."""

    type_id: str
    description: str
    system_prompt: str
    allowed_tools: frozenset[str]
    capabilities: tuple[str, ...]
    max_steps: int = 5  # Max LLM turns per task
    temperature: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        return {
            "type_id": self.type_id,
            "description": self.description,
            "allowed_tools": sorted(self.allowed_tools),
            "capabilities": list(self.capabilities),
            "max_steps": self.max_steps,
        }


@dataclass(frozen=True)
class DynamicAgentSpec:
    """Task-specific specialization layered on top of a stable base agent type."""

    spec_id: str
    base_type: str
    collaboration_role: str
    specialization: str
    mission: str
    success_criteria: tuple[str, ...] = ()
    output_schema: dict[str, Any] = field(default_factory=dict)
    memory_scopes: tuple[str, ...] = ("global", "team")
    allowed_tools: tuple[str, ...] = ()
    max_steps: int | None = None
    temperature: float | None = None

    def to_transport_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "base_type": self.base_type,
            "collaboration_role": self.collaboration_role,
            "specialization": self.specialization,
            "mission": self.mission,
            "success_criteria": list(self.success_criteria),
            "output_schema": dict(self.output_schema),
            "memory_scopes": list(self.memory_scopes),
            "allowed_tools": list(self.allowed_tools),
            "max_steps": self.max_steps,
            "temperature": self.temperature,
        }


# ── The registry: real agent types with real JARVIS tools ────────────

AGENT_TYPES: dict[str, AgentType] = {}


def _register(
    type_id: str,
    description: str,
    system_prompt: str,
    tools: list[str],
    capabilities: list[str],
    max_steps: int = 5,
    temperature: float = 0.7,
) -> None:
    AGENT_TYPES[type_id] = AgentType(
        type_id=type_id,
        description=description,
        system_prompt=system_prompt,
        allowed_tools=frozenset(tools),
        capabilities=tuple(capabilities),
        max_steps=max_steps,
        temperature=temperature,
    )


_register(
    "researcher",
    "Web research and information gathering specialist",
    (
        "You are a research specialist. Your job is to find accurate, comprehensive "
        "information from the web and memory. Always cite sources. Prefer deep_search "
        "for complex topics, web_search for quick lookups. Cross-reference multiple "
        "sources before presenting findings."
    ),
    ["web_search", "deep_search", "web_scrape", "memory_recall", "pdf_reader", "semantic_search"],
    ["search_web", "find_information", "research", "fact_check", "read_documents"],
    max_steps=8,
)

_register(
    "coder",
    "Software development and code execution specialist",
    (
        "You are an expert software developer. Write clean, efficient, well-tested code. "
        "Use python_exec for Python tasks, execute_code for other languages. Read existing "
        "files before modifying them. Prefer small, focused changes over large rewrites. "
        "Always validate your code by running it."
    ),
    ["python_exec", "execute_code", "read_file", "write_file", "run_command", "list_directory"],
    ["write_code", "execute_code", "debug", "refactor", "review_code"],
    max_steps=10,
    temperature=0.3,
)

_register(
    "desktop_agent",
    "macOS desktop automation and screen interaction",
    (
        "You are a desktop automation agent controlling a macOS computer. "
        "Use screen_vision to understand what's on screen before acting. "
        "Prefer clicking UI elements by label over coordinates. "
        "Be careful with destructive actions — always verify before proceeding."
    ),
    ["desktop_control", "screen_vision"],
    ["control_desktop", "click_button", "type_text", "open_app", "screenshot"],
    max_steps=15,
)

_register(
    "mobile_agent",
    "Android device automation via ADB",
    (
        "You are a mobile automation agent controlling an Android device via ADB. "
        "Use android_control to interact with the device. Read the UI tree before "
        "tapping elements. Be precise with touch coordinates and text input."
    ),
    ["android_control"],
    ["control_android", "tap", "swipe", "type_text", "install_app", "screenshot"],
    max_steps=15,
)

_register(
    "creative",
    "Content creation — images, diagrams, voice",
    (
        "You are a creative specialist. Generate images with image_gen, "
        "create diagrams with the diagram tool (Mermaid syntax), "
        "and handle voice with voice_note. Focus on quality and clarity. "
        "Ask for specifics when the request is vague."
    ),
    ["image_gen", "diagram", "voice_note", "write_file"],
    ["generate_image", "create_diagram", "record_audio", "transcribe", "create_content"],
)

_register(
    "analyst",
    "Data analysis, document reading, and knowledge extraction",
    (
        "You are a data analyst. Read and analyze documents with pdf_reader, "
        "search knowledge bases with semantic_search and memory_recall. "
        "Execute analysis code with python_exec. Present findings clearly "
        "with supporting data."
    ),
    ["pdf_reader", "semantic_search", "memory_recall", "python_exec", "read_file"],
    ["analyze_data", "read_pdf", "search_knowledge", "summarize", "extract_entities"],
    max_steps=8,
)

_register(
    "planner",
    "Strategic planning and task decomposition",
    (
        "You are a strategic planner. Break complex goals into actionable steps. "
        "Consider risks, dependencies, and resource constraints. Use memory to "
        "recall past plans and outcomes. Present plans as structured, prioritized lists."
    ),
    ["strategic_plan", "memory_store", "memory_recall", "semantic_search"],
    ["plan", "decompose_task", "prioritize", "estimate", "strategize"],
    temperature=0.5,
)

_register(
    "devops",
    "Git operations, CI/CD, and system administration",
    (
        "You are a DevOps engineer. Handle git operations, run system commands, "
        "manage files and directories. Follow best practices: commit messages "
        "should be descriptive, always check status before destructive operations, "
        "prefer safe operations over forced ones."
    ),
    ["git_context", "git_diff_summary", "git_commit_suggest", "git_find_files",
     "run_command", "read_file", "write_file", "list_directory"],
    ["git_status", "git_commit", "git_diff", "run_command", "deploy", "manage_files"],
    max_steps=8,
)

_register(
    "communicator",
    "Voice interaction, speech synthesis, and messaging",
    (
        "You are a communication specialist. Handle voice recording, transcription, "
        "and text-to-speech. Keep messages clear and concise. Adapt tone to context."
    ),
    ["speak", "voice_note", "memory_store"],
    ["speak", "transcribe", "record_audio", "send_message"],
)

_register(
    "memory_agent",
    "Knowledge management — store, recall, and organize information",
    (
        "You are a knowledge manager. Store important information with proper tags "
        "and importance levels. Recall relevant memories efficiently. Use semantic_search "
        "for fuzzy matching. Keep the knowledge base clean — update outdated entries, "
        "merge duplicates, and maintain consistent tagging."
    ),
    ["memory_store", "memory_recall", "semantic_search"],
    ["store_memory", "recall_memory", "search_knowledge", "organize_knowledge"],
)

_register(
    "generalist",
    "General-purpose agent with broad tool access",
    (
        "You are a capable general-purpose assistant. Use the right tool for "
        "each task. Prefer reading before writing, searching before assuming, "
        "and asking before guessing. Be efficient — don't use tools unnecessarily."
    ),
    [
        "read_file", "write_file", "run_command", "list_directory",
        "web_search", "web_scrape", "memory_store", "memory_recall",
        "python_exec", "semantic_search", "pdf_reader",
    ],
    ["general", "any_task"],
    max_steps=10,
    )


def create_dynamic_agent_spec(
    *,
    base_type: str,
    collaboration_role: str | None = None,
    specialization: str | None = None,
    mission: str | None = None,
    success_criteria: list[str] | tuple[str, ...] | None = None,
    output_schema: dict[str, Any] | None = None,
    memory_scopes: list[str] | tuple[str, ...] | None = None,
    allowed_tools: list[str] | tuple[str, ...] | None = None,
    max_steps: int | None = None,
    temperature: float | None = None,
) -> DynamicAgentSpec:
    """Create a validated dynamic agent spec anchored to a stable archetype."""
    archetype = AGENT_TYPES.get(base_type) or AGENT_TYPES["generalist"]
    clean_role = str(collaboration_role or base_type).strip().lower() or archetype.type_id
    clean_specialization = (
        str(specialization or clean_role.replace("_", " ")).strip()
        or archetype.type_id.replace("_", " ")
    )
    clean_mission = str(mission or "").strip() or (
        f"Advance the task as a {clean_specialization} specialist."
    )
    requested_tools = [str(tool).strip() for tool in (allowed_tools or []) if str(tool).strip()]
    if requested_tools:
        allowed = tuple(tool for tool in requested_tools if tool in archetype.allowed_tools)
        if not allowed:
            allowed = tuple(archetype.allowed_tools)
    else:
        allowed = tuple(archetype.allowed_tools)
    requested_scopes = [str(scope).strip().lower() for scope in (memory_scopes or ()) if str(scope).strip()]
    scopes = tuple(requested_scopes or ("global", "team"))
    criteria = tuple(
        str(item).strip() for item in (success_criteria or ()) if str(item).strip()
    )[:5]
    schema = dict(output_schema or {})
    steps = max_steps or archetype.max_steps
    temp = archetype.temperature if temperature is None else float(temperature)
    return DynamicAgentSpec(
        spec_id=f"dyn-{uuid4().hex[:8]}",
        base_type=archetype.type_id,
        collaboration_role=clean_role,
        specialization=clean_specialization,
        mission=clean_mission,
        success_criteria=criteria,
        output_schema=schema,
        memory_scopes=scopes,
        allowed_tools=allowed,
        max_steps=max(1, int(steps)),
        temperature=temp,
    )


def compile_dynamic_agent_prompt(
    spec: DynamicAgentSpec,
    *,
    shared_context: str = "",
    original_task: str = "",
) -> tuple[str, str]:
    """Compile a dynamic spec into executable system/user prompts."""
    archetype = AGENT_TYPES.get(spec.base_type) or AGENT_TYPES["generalist"]
    tools_text = ", ".join(spec.allowed_tools) if spec.allowed_tools else "no tools"
    scopes_text = ", ".join(spec.memory_scopes) if spec.memory_scopes else "global"
    criteria_lines = "\n".join(
        f"- {item}" for item in spec.success_criteria
    ) or "- Produce the strongest useful contribution for this role."
    if spec.output_schema:
        schema_lines = "\n".join(
            f"- {key}: {value}" for key, value in spec.output_schema.items()
        )
    else:
        schema_lines = "- findings: concise task-relevant output"
    system_prompt = (
        f"{archetype.system_prompt}\n\n"
        f"You are the specialized collaboration role '{spec.collaboration_role}'.\n"
        f"Specialization: {spec.specialization}\n"
        f"Mission: {spec.mission}\n"
        f"Memory scopes available: {scopes_text}\n"
        f"Allowed tools for this run: {tools_text}\n"
        "Work from current repo/runtime truth first. Treat memory as supporting context, "
        "not as authority over live files, tests, or tool output.\n"
        "Write concrete, high-signal intermediate results that other agents can build on."
    )
    user_sections = [
        f"Role objective: {spec.mission}",
        "Success criteria:\n" + criteria_lines,
        "Expected output shape:\n" + schema_lines,
    ]
    if original_task.strip():
        user_sections.append(f"Original task:\n{original_task.strip()}")
    if shared_context.strip():
        user_sections.append(f"Shared collaboration context:\n{shared_context.strip()}")
    return system_prompt, "\n\n".join(user_sections)


# ---------------------------------------------------------------------------
# Capability Router — match tasks to agent types
# ---------------------------------------------------------------------------


class CapabilityRouter:
    """Route tasks to the best agent type based on keywords and capabilities."""

    # Keyword → agent type mapping for fast routing
    _KEYWORD_MAP: dict[str, str] = {
        # Researcher
        "search": "researcher", "find": "researcher", "lookup": "researcher",
        "research": "researcher", "google": "researcher", "investigate": "researcher",
        # Coder
        "code": "coder", "program": "coder", "script": "coder", "debug": "coder",
        "function": "coder", "implement": "coder", "compile": "coder", "refactor": "coder",
        # Desktop
        "click": "desktop_agent", "screenshot": "desktop_agent", "open app": "desktop_agent",
        "desktop": "desktop_agent", "window": "desktop_agent", "screen": "desktop_agent",
        # Mobile
        "android": "mobile_agent", "phone": "mobile_agent", "adb": "mobile_agent",
        "tap": "mobile_agent", "swipe": "mobile_agent",
        # Creative
        "image": "creative", "diagram": "creative", "draw": "creative",
        "generate image": "creative", "flowchart": "creative", "picture": "creative",
        # Analyst
        "analyze": "analyst", "pdf": "analyst", "report": "analyst",
        "data": "analyst", "statistics": "analyst", "extract": "analyst",
        # Planner
        "plan": "planner", "strategy": "planner", "roadmap": "planner",
        "prioritize": "planner", "decompose": "planner",
        # DevOps
        "git": "devops", "commit": "devops", "deploy": "devops",
        "branch": "devops", "merge": "devops", "ci": "devops",
        # Communicator
        "speak": "communicator", "voice": "communicator", "transcribe": "communicator",
        "record": "communicator", "say": "communicator",
        # Memory
        "remember": "memory_agent", "recall": "memory_agent", "forget": "memory_agent",
        "store memory": "memory_agent",
    }

    @classmethod
    def route(cls, task_description: str) -> str:
        """Pick the best agent type for a task description."""
        desc_lower = task_description.lower()

        # Score each agent type by keyword matches
        scores: dict[str, int] = {}
        for keyword, agent_type in cls._KEYWORD_MAP.items():
            if keyword in desc_lower:
                scores[agent_type] = scores.get(agent_type, 0) + 1

        if scores:
            return max(scores, key=lambda k: scores[k])

        # Fallback: capability matching against all agent types
        for type_id, agent_type in AGENT_TYPES.items():
            for cap in agent_type.capabilities:
                if cap.replace("_", " ") in desc_lower:
                    return type_id

        return "generalist"

    @classmethod
    def route_multi(cls, task_description: str, n: int = 3) -> list[str]:
        """Pick the best N agent types for a complex task."""
        desc_lower = task_description.lower()

        scores: dict[str, int] = {}
        for keyword, agent_type in cls._KEYWORD_MAP.items():
            if keyword in desc_lower:
                scores[agent_type] = scores.get(agent_type, 0) + 1

        if not scores:
            return ["generalist"]

        sorted_types = sorted(scores, key=lambda k: scores[k], reverse=True)
        return sorted_types[:n]


# ---------------------------------------------------------------------------
# Agent Instance — a running agent with state
# ---------------------------------------------------------------------------

@dataclass
class AgentInstance:
    """A live agent instance with its type, state, and performance."""

    instance_id: str = field(default_factory=lambda: uuid4().hex[:12])
    type_id: str = "generalist"
    status: str = "idle"  # idle, busy, done, failed
    created_at: float = field(default_factory=time.time)
    task_count: int = 0
    total_latency_ms: float = 0.0
    success_count: int = 0
    error_count: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.task_count, 1)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.error_count
        return self.success_count / max(total, 1)


# ---------------------------------------------------------------------------
# Task Store — track tasks and results (in-memory, bounded)
# ---------------------------------------------------------------------------

@dataclass
class TaskRecord:
    """A tracked task with its lifecycle."""

    task_id: str
    agent_instance_id: str
    agent_type: str
    prompt: str
    status: str = "pending"  # pending, running, completed, failed
    output: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    latency_ms: float = 0.0


class TaskStore:
    """Bounded in-memory task store with optional history."""

    _MAX_TASKS = 500

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self, task_id: str, agent_id: str, agent_type: str, prompt: str) -> TaskRecord:
        async with self._lock:
            record = TaskRecord(
                task_id=task_id,
                agent_instance_id=agent_id,
                agent_type=agent_type,
                prompt=prompt,
                status="running",
            )
            self._tasks[task_id] = record
            # Evict oldest completed tasks if over capacity
            if len(self._tasks) > self._MAX_TASKS:
                self._evict_old()
            return record

    async def complete(self, task_id: str, output: str) -> None:
        async with self._lock:
            if task_id in self._tasks:
                t = self._tasks[task_id]
                t.status = "completed"
                t.output = output[:50000]
                t.completed_at = time.time()
                t.latency_ms = (t.completed_at - t.created_at) * 1000

    async def fail(self, task_id: str, error: str) -> None:
        async with self._lock:
            if task_id in self._tasks:
                t = self._tasks[task_id]
                t.status = "failed"
                t.error = error[:5000]
                t.completed_at = time.time()
                t.latency_ms = (t.completed_at - t.created_at) * 1000

    async def get(self, task_id: str) -> TaskRecord | None:
        async with self._lock:
            return self._tasks.get(task_id)

    async def recent(self, limit: int = 20) -> list[TaskRecord]:
        async with self._lock:
            sorted_tasks = sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)
            return sorted_tasks[:limit]

    def _evict_old(self) -> None:
        completed = [
            (tid, t) for tid, t in self._tasks.items()
            if t.status in ("completed", "failed")
        ]
        completed.sort(key=lambda x: x[1].created_at)
        to_remove = len(self._tasks) - self._MAX_TASKS
        for tid, _ in completed[:max(to_remove, 0)]:
            self._tasks.pop(tid, None)


# ---------------------------------------------------------------------------
# Performance Tracker — lightweight self-optimization
# ---------------------------------------------------------------------------


class PerformanceTracker:
    """Track per-agent-type performance for routing optimization."""

    def __init__(self, history_size: int = 100) -> None:
        self._history: dict[str, deque[dict[str, Any]]] = {}
        self._history_size = history_size

    def record(self, agent_type: str, latency_ms: float, success: bool) -> None:
        if agent_type not in self._history:
            self._history[agent_type] = deque(maxlen=self._history_size)
        self._history[agent_type].append({
            "latency_ms": latency_ms,
            "success": success,
            "timestamp": time.time(),
        })

    def stats(self, agent_type: str) -> dict[str, Any]:
        """Get performance stats for an agent type."""
        history = self._history.get(agent_type, deque())
        if not history:
            return {"tasks": 0, "avg_latency_ms": 0, "success_rate": 1.0}

        latencies = [h["latency_ms"] for h in history]
        successes = sum(1 for h in history if h["success"])
        return {
            "tasks": len(history),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "success_rate": successes / len(history),
        }

    def all_stats(self) -> dict[str, dict[str, Any]]:
        return {at: self.stats(at) for at in self._history}

    def best_type_for(self, candidates: list[str]) -> str:
        """Pick the best agent type from candidates based on historical performance."""
        if not candidates:
            return "generalist"

        scored = []
        for c in candidates:
            s = self.stats(c)
            # Score = success_rate * (1 / normalized_latency)
            # Higher is better
            avg_lat = max(s["avg_latency_ms"], 1)
            score = s["success_rate"] * (10000 / avg_lat)
            scored.append((c, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]


# ---------------------------------------------------------------------------
# Agent Engine — the main orchestrator
# ---------------------------------------------------------------------------


class AgentEngine:
    """
    Unified agent orchestration engine.

    Combines typed agents (Claude Code style), capability routing (DAF style),
    and JARVIS's real tool execution into one in-process system.
    """

    def __init__(
        self,
        llm: Any,  # LLMInterface — must have .chat(messages, tools=) method
        tool_ctx: Any,  # ToolContext from tool_handlers
        handler_map: dict[str, Any] | None = None,  # HANDLER_MAP
    ) -> None:
        self.llm = llm
        self.tool_ctx = tool_ctx
        self._handler_map = handler_map or {}
        self.task_store = TaskStore()
        self.performance = PerformanceTracker()
        self._instances: dict[str, AgentInstance] = {}
        self._lock = asyncio.Lock()
        logger.info(
            "AgentEngine initialized with %d agent types, %d tools",
            len(AGENT_TYPES), len(self._handler_map),
        )

    # ── Single agent execution ─────────────────────────────────────

    async def run_task(
        self,
        prompt: str,
        agent_type: str | None = None,
        max_steps: int | None = None,
        dynamic_spec: DynamicAgentSpec | None = None,
        system_prompt_override: str | None = None,
        prompt_override: str | None = None,
    ) -> dict[str, Any]:
        """Run a task with a single typed agent.

        If agent_type is None, auto-routes based on the prompt.
        """
        if dynamic_spec is not None:
            agent_type = dynamic_spec.base_type
        if not agent_type:
            agent_type = CapabilityRouter.route(prompt)

        atype = AGENT_TYPES.get(agent_type)
        if not atype:
            atype = AGENT_TYPES["generalist"]
            agent_type = "generalist"

        system_prompt = system_prompt_override
        task_prompt = prompt_override or prompt
        if dynamic_spec is not None and (system_prompt is None or prompt_override is None):
            compiled_system, compiled_user = compile_dynamic_agent_prompt(
                dynamic_spec,
                original_task=prompt,
            )
            if system_prompt is None:
                system_prompt = compiled_system
            if prompt_override is None:
                task_prompt = compiled_user
        if system_prompt is None:
            system_prompt = atype.system_prompt

        steps = max_steps or (dynamic_spec.max_steps if dynamic_spec else None) or atype.max_steps

        # Create instance
        instance = AgentInstance(type_id=agent_type)
        async with self._lock:
            self._instances[instance.instance_id] = instance
            instance.status = "busy"

        # Create task record
        task_id = f"task-{uuid4().hex[:8]}"
        await self.task_store.create(task_id, instance.instance_id, agent_type, prompt)

        t0 = time.monotonic()
        try:
            # Build scoped tool definitions
            scoped_tools = self._get_scoped_tools(atype, dynamic_spec=dynamic_spec)

            # Run agent loop
            output = await self._agent_loop(
                prompt=task_prompt,
                system_prompt=system_prompt,
                tools=scoped_tools,
                handler_map={t["name"]: self._handler_map[t["name"]] for t in scoped_tools if t["name"] in self._handler_map},
                max_steps=steps,
                temperature=(
                    dynamic_spec.temperature
                    if dynamic_spec is not None and dynamic_spec.temperature is not None
                    else atype.temperature
                ),
            )

            latency = (time.monotonic() - t0) * 1000
            async with self._lock:
                instance.task_count += 1
                instance.total_latency_ms += latency
                instance.success_count += 1
                instance.status = "done"

            await self.task_store.complete(task_id, output)
            self.performance.record(agent_type, latency, True)

            return {
                "task_id": task_id,
                "agent_type": agent_type,
                "instance_id": instance.instance_id,
                "output": output,
                "latency_ms": round(latency, 1),
                "status": "completed",
            }

        except Exception as e:
            latency = (time.monotonic() - t0) * 1000
            async with self._lock:
                instance.error_count += 1
                instance.status = "failed"

            await self.task_store.fail(task_id, str(e))
            self.performance.record(agent_type, latency, False)

            return {
                "task_id": task_id,
                "agent_type": agent_type,
                "instance_id": instance.instance_id,
                "output": "",
                "error": str(e),
                "latency_ms": round(latency, 1),
                "status": "failed",
            }

    # ── Multi-agent patterns ───────────────────────────────────────

    async def run_team(
        self,
        prompt: str,
        pattern: str = "fan_out",
        agent_types: list[str] | None = None,
        num_agents: int = 3,
    ) -> dict[str, Any]:
        """Run a multi-agent collaboration with typed agents."""
        t0 = time.monotonic()

        # Auto-pick agent types if not specified
        if not agent_types:
            agent_types = CapabilityRouter.route_multi(prompt, n=num_agents)
            # Pad with generalists if needed
            while len(agent_types) < num_agents:
                agent_types.append("generalist")

        if pattern == "fan_out":
            results = await self._fan_out(prompt, agent_types)
        elif pattern == "pipeline":
            results = await self._pipeline(prompt, agent_types)
        elif pattern == "consensus":
            results = await self._consensus(prompt, agent_types)
        elif pattern == "supervisor":
            results = await self._supervisor(prompt, agent_types)
        else:
            results = await self._fan_out(prompt, agent_types)

        total_ms = (time.monotonic() - t0) * 1000

        return {
            "pattern": pattern,
            "agent_types": agent_types,
            "results": results,
            "total_latency_ms": round(total_ms, 1),
            "success_count": sum(1 for r in results if r.get("status") == "completed"),
            "error_count": sum(1 for r in results if r.get("status") == "failed"),
        }

    async def _fan_out(self, prompt: str, agent_types: list[str]) -> list[dict[str, Any]]:
        """All agents work on the same task in parallel."""
        coros = [self.run_task(prompt, agent_type=at) for at in agent_types]
        raw_results = await asyncio.gather(*coros, return_exceptions=True)
        results = []
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                logger.warning("Fan-out agent %s failed: %s", agent_types[i], r)
                results.append({
                    "agent_type": agent_types[i],
                    "status": "failed",
                    "error": str(r),
                    "output": "",
                })
            else:
                results.append(r)
        return results

    async def _pipeline(self, prompt: str, agent_types: list[str]) -> list[dict[str, Any]]:
        """Chain agents — each gets previous agent's output as input."""
        results = []
        current_prompt = prompt

        for at in agent_types:
            result = await self.run_task(current_prompt, agent_type=at)
            results.append(result)
            if result.get("status") == "failed":
                break
            # Feed output to next agent
            current_prompt = (
                f"Previous agent ({at}) produced this output:\n\n"
                f"{result.get('output', '')}\n\n"
                f"Original task: {prompt}\n\n"
                f"Continue building on this work."
            )

        return results

    async def _consensus(self, prompt: str, agent_types: list[str]) -> list[dict[str, Any]]:
        """All agents work independently, majority output wins."""
        results = await self._fan_out(prompt, agent_types)

        # Find majority by voting on output similarity
        outputs = [r.get("output", "") for r in results if r.get("status") == "completed"]
        if outputs:
            # Vote: count how many other outputs share >50% token overlap
            from collections import Counter
            votes: Counter[int] = Counter()
            for i, out_a in enumerate(outputs):
                words_a = set(out_a.lower().split())
                for j, out_b in enumerate(outputs):
                    if i == j or not words_a:
                        continue
                    words_b = set(out_b.lower().split())
                    overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
                    if overlap > 0.3:
                        votes[i] += 1
            # Winner = most votes (most agreed-upon), tiebreak by length
            if votes:
                best_idx = max(votes, key=lambda i: (votes[i], len(outputs[i])))
            else:
                best_idx = 0
            for r in results:
                if r.get("output") == outputs[best_idx]:
                    r["consensus_winner"] = True

        return results

    async def _supervisor(self, prompt: str, agent_types: list[str]) -> list[dict[str, Any]]:
        """First agent does the work, last agent reviews it."""
        if len(agent_types) < 2:
            return await self._fan_out(prompt, agent_types)

        worker_type = agent_types[0]
        supervisor_type = agent_types[-1]

        # Worker
        worker_result = await self.run_task(prompt, agent_type=worker_type)

        # Supervisor reviews
        review_prompt = (
            f"Review and improve this output. Fix any errors, fill gaps, "
            f"and ensure quality.\n\n"
            f"Original task: {prompt}\n\n"
            f"Worker output:\n{worker_result.get('output', '')}"
        )
        sup_result = await self.run_task(review_prompt, agent_type=supervisor_type)
        sup_result["role"] = "supervisor"
        worker_result["role"] = "worker"

        return [worker_result, sup_result]

    # ── Agent loop — the core execution engine ─────────────────────

    async def _agent_loop(
        self,
        prompt: str,
        system_prompt: str,
        tools: list[dict[str, Any]],
        handler_map: dict[str, Any],
        max_steps: int = 5,
        temperature: float = 0.7,
    ) -> str:
        """Run an agent loop: LLM thinks → calls tools → collects results → repeats.

        This is the core execution engine. The LLM gets:
        - A specialized system prompt (agent type personality)
        - Only the tools allowed for this agent type
        - Conversation history including tool results
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        last_text_output = ""

        for step in range(max_steps):
            # Call LLM with scoped tools
            try:
                response = await self.llm.chat(
                    messages=messages,
                    tools=tools if tools else None,
                )
            except Exception as e:
                logger.error("LLM call failed at step %d/%d: %s", step + 1, max_steps, e)
                if last_text_output:
                    return last_text_output
                raise

            content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])

            # If LLM produced text and no tool calls, we're done
            if content and not tool_calls:
                return content

            # If LLM wants to call tools, execute them
            if tool_calls:
                # Add assistant message with tool calls
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

                for tc in tool_calls:
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("arguments", {})

                    # Execute only if tool is in this agent's allowed set
                    if tool_name in handler_map:
                        try:
                            result = await handler_map[tool_name](tool_args, self.tool_ctx)
                        except (TimeoutError, asyncio.TimeoutError) as e:
                            result = f"[Tool timeout (retriable): {e}]"
                            logger.warning(
                                "Retriable tool error in %s: %s", tool_name, e,
                            )
                        except (ConnectionError, OSError) as e:
                            result = f"[Tool connection error (retriable): {e}]"
                            logger.warning(
                                "Retriable tool error in %s: %s", tool_name, e,
                            )
                        except Exception as e:
                            result = f"[Tool error (fatal): {e}]"
                            logger.error(
                                "Fatal tool error in %s: %s", tool_name, e,
                            )
                    else:
                        result = f"[Tool '{tool_name}' not available for this agent type]"

                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "content": str(result)[:30000],
                        "tool_call_id": tc.get("id", ""),
                    })

                last_text_output = content
            elif content:
                return content
            else:
                # No content and no tool calls — done
                break

        return last_text_output or "[Agent completed without producing output]"

    # ── Tool scoping ───────────────────────────────────────────────

    def _get_scoped_tools(
        self,
        atype: AgentType,
        *,
        dynamic_spec: DynamicAgentSpec | None = None,
    ) -> list[dict[str, Any]]:
        """Build tool definitions scoped to this agent type's allowlist.

        Only includes tools that:
        1. Are in the agent type's allowed_tools set
        2. Have a handler in HANDLER_MAP
        """
        scoped = []
        allowed_names = set(dynamic_spec.allowed_tools) if dynamic_spec else set(atype.allowed_tools)
        # Lazy import to avoid circular dep: agents ↔ tools
        try:
            from jarvis.tools.registry import BUILTIN_TOOLS_RAW
            for tool_raw in BUILTIN_TOOLS_RAW:
                tool_def = tool_raw[0]  # First element is the dict
                name = tool_def.get("name", "")
                if name in allowed_names and name in self._handler_map:
                    scoped.append({
                        "name": name,
                        "description": tool_def.get("description", ""),
                        "parameters": tool_def.get("parameters", {}),
                    })
        except ImportError:
            logger.warning("Could not import tool_registry for scoped tools")

        return scoped

    # ── Pool maintenance ────────────────────────────────────────────

    async def cleanup_stale_instances(self, max_age_seconds: float = 3600) -> int:
        """Remove agent instances that are done/failed or have been idle too long.

        Returns the number of instances removed.
        """
        now = time.time()
        to_remove: list[str] = []
        async with self._lock:
            for iid, inst in self._instances.items():
                if inst.status in ("done", "failed"):
                    to_remove.append(iid)
                elif inst.status == "idle" and (now - inst.created_at) > max_age_seconds:
                    to_remove.append(iid)
            for iid in to_remove:
                self._instances.pop(iid, None)
        if to_remove:
            logger.info("Cleaned up %d stale agent instances", len(to_remove))
        return len(to_remove)

    # ── Introspection ──────────────────────────────────────────────

    def list_agent_types(self) -> list[dict[str, Any]]:
        """List all available agent types."""
        return [at.to_dict() for at in AGENT_TYPES.values()]

    def get_agent_type(self, type_id: str) -> AgentType | None:
        return AGENT_TYPES.get(type_id)

    async def get_performance(self) -> dict[str, Any]:
        """Get performance stats for all agent types."""
        return {
            "agent_stats": self.performance.all_stats(),
            "active_instances": len([i for i in self._instances.values() if i.status == "busy"]),
            "total_instances": len(self._instances),
            "recent_tasks": [
                {
                    "task_id": t.task_id,
                    "agent_type": t.agent_type,
                    "status": t.status,
                    "latency_ms": round(t.latency_ms, 1),
                }
                for t in await self.task_store.recent(10)
            ],
        }

    async def route_task(self, description: str) -> dict[str, Any]:
        """Show how a task would be routed (for debugging)."""
        single = CapabilityRouter.route(description)
        multi = CapabilityRouter.route_multi(description, n=3)
        return {
            "description": description,
            "recommended_single": single,
            "recommended_multi": multi,
            "agent_type_info": AGENT_TYPES[single].to_dict() if single in AGENT_TYPES else None,
        }
