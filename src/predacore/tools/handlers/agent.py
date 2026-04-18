"""Multi-agent, strategic planning, and OpenClaw delegation handlers."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from ._context import (
    _DELEGATION_DEPTH,
    _MAX_DELEGATION_DEPTH,
    ToolContext,
    ToolError,
    ToolErrorKind,
    missing_param,
    subsystem_unavailable,
)

logger = logging.getLogger(__name__)


def _truncate_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 18)].rstrip() + "\n...[truncated]..."


def _format_explicit_context(raw_context: Any, max_chars: int = 1200) -> str:
    if not raw_context:
        return ""
    if isinstance(raw_context, str):
        return _truncate_text(raw_context, max_chars)
    try:
        return _truncate_text(
            json.dumps(raw_context, indent=2, default=str),
            max_chars,
        )
    except (TypeError, ValueError):
        return _truncate_text(str(raw_context), max_chars)


def _format_recent_tool_memory(
    memory_store: dict[str, dict[str, Any]],
    limit: int = 5,
    max_chars: int = 1600,
) -> str:
    if not memory_store:
        return ""

    items = sorted(
        memory_store.items(),
        key=lambda item: float(item[1].get("stored_at", 0.0)),
        reverse=True,
    )
    lines: list[str] = []
    used = 0
    for key, item in items[:limit]:
        content = _truncate_text(str(item.get("content") or ""), 240)
        if not content:
            continue
        tags = item.get("tags") or []
        tag_text = f" ({', '.join(str(t) for t in tags[:3])})" if tags else ""
        line = f"- {key}{tag_text}: {content}"
        if lines and used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)

    return "\n".join(lines)


def _resolve_multi_agent_user_id(args: dict[str, Any], ctx: ToolContext) -> str:
    if callable(ctx.resolve_user_id):
        try:
            return str(ctx.resolve_user_id(args) or "default")
        except (TypeError, ValueError, RuntimeError):
            logger.debug("Failed to resolve multi-agent user_id", exc_info=True)
    return "default"


def _resolve_multi_agent_team_id(args: dict[str, Any]) -> str:
    explicit = str(args.get("team_id") or "").strip()
    if explicit:
        return explicit
    return f"team-{uuid4().hex[:10]}"


def _team_memory_expiry(hours: Any) -> str | None:
    try:
        from predacore.memory.store import future_iso_from_ttl
    except ImportError:
        from src.predacore.memory.store import future_iso_from_ttl  # type: ignore

    try:
        ttl_hours = max(int(hours), 1)
    except (TypeError, ValueError):
        ttl_hours = 72
    return future_iso_from_ttl(ttl_hours * 3600)


def _format_recalled_memories(
    title: str,
    recalls: list[tuple[dict[str, Any], float]],
    max_chars: int = 1600,
) -> str:
    if not recalls:
        return ""
    lines = [f"## {title}"]
    used = len(lines[0])
    for mem, score in recalls:
        meta = mem.get("metadata", {}) if isinstance(mem.get("metadata"), dict) else {}
        label = (
            meta.get("key")
            or meta.get("agent_role")
            or meta.get("stage")
            or mem.get("memory_type")
            or "memory"
        )
        line = f"- {label} (score={score:.3f}): {_truncate_text(mem.get('content', ''), 220)}"
        if len(lines) > 1 and used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines) if len(lines) > 1 else ""


async def _recall_team_memories(
    prompt: str,
    *,
    ctx: ToolContext,
    user_id: str,
    team_id: str,
    top_k: int = 4,
) -> str:
    if ctx.unified_memory is None or not hasattr(ctx.unified_memory, "recall"):
        return ""
    try:
        recalls = await ctx.unified_memory.recall(
            query=prompt,
            user_id=user_id,
            top_k=top_k,
            scopes=["team"],
            team_id=team_id,
        )
    except (RuntimeError, OSError, ValueError, TypeError):
        logger.debug("Team memory recall failed", exc_info=True)
        return ""
    return _format_recalled_memories("Existing Team Memory", recalls, max_chars=1400)


async def _persist_team_memory(
    *,
    ctx: ToolContext,
    user_id: str,
    team_id: str,
    session_id: str | None,
    content: str,
    stage: str,
    pattern: str,
    importance: int = 2,
    agent_role: str | None = None,
    agent_id: str | None = None,
    memory_type: str = "task",
    ttl_hours: Any = 72,
) -> None:
    if ctx.unified_memory is None or not hasattr(ctx.unified_memory, "store"):
        return
    text = _truncate_text(content, 4000)
    if not text:
        return
    metadata = {
        "source": "predacore.multi_agent",
        "stage": stage,
        "pattern": pattern,
    }
    if agent_role:
        metadata["agent_role"] = agent_role
    try:
        await ctx.unified_memory.store(
            content=text,
            memory_type=memory_type,
            importance=importance,
            source="predacore.multi_agent",
            tags=["multi_agent", "team_memory", pattern],
            metadata=metadata,
            user_id=user_id,
            session_id=session_id,
            expires_at=_team_memory_expiry(ttl_hours),
            memory_scope="team",
            team_id=team_id,
            agent_id=agent_id,
        )
    except (RuntimeError, OSError, ValueError, TypeError):
        logger.debug("Persisting team memory failed", exc_info=True)


def _extract_json_payload(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            candidates.append("\n".join(lines[1:-1]).strip())
    start_positions = [idx for idx in (raw.find("{"), raw.find("[")) if idx >= 0]
    if start_positions:
        start = min(start_positions)
        end = max(raw.rfind("}"), raw.rfind("]"))
        if end > start:
            candidates.append(raw[start : end + 1].strip())
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def _default_dynamic_agent_spec_inputs(
    roles: list[str],
    *,
    num_agents: int,
    prompt: str,
    pattern_name: str,
) -> list[dict[str, Any]]:
    try:
        from predacore.agents.engine import AGENT_TYPES, CapabilityRouter
    except ImportError:
        from src.predacore.agents.engine import AGENT_TYPES, CapabilityRouter  # type: ignore

    prompt_lower = prompt.lower()
    defaults = [
        "researcher",
        "analyst",
        "critic",
        "planner",
        "coder",
        "synthesizer",
        "generalist",
    ]
    suggested = CapabilityRouter.route_multi(prompt, n=max(num_agents, 2))
    chosen_roles = [str(role).strip().lower() for role in roles if str(role).strip()]
    while len(chosen_roles) < num_agents:
        if suggested:
            chosen_roles.append(suggested.pop(0))
        else:
            chosen_roles.append(defaults[len(chosen_roles) % len(defaults)])

    specs: list[dict[str, Any]] = []
    for idx in range(num_agents):
        role = chosen_roles[idx]
        base_type = role if role in AGENT_TYPES else CapabilityRouter.route(role)
        if base_type not in AGENT_TYPES:
            base_type = "generalist"
        if role == "critic":
            mission = "Stress-test the task, spot weak assumptions, and identify the biggest risks."
            criteria = [
                "Call out concrete risks or contradictions",
                "Cite the most important missing evidence",
                "Suggest the smallest correction path",
            ]
            schema = {"risks": "array", "evidence_gaps": "array", "corrections": "array"}
        elif role in {"researcher", "analyst"}:
            mission = "Gather the highest-signal evidence and surface the most relevant facts."
            criteria = [
                "Extract the most relevant facts",
                "Keep findings concise and evidence-based",
                "Avoid speculation beyond repo/runtime truth",
            ]
            schema = {"findings": "array", "evidence": "array", "unknowns": "array"}
        elif role == "planner":
            mission = "Turn the task into a concrete execution plan with dependencies and priorities."
            criteria = [
                "Break the task into executable steps",
                "Name dependencies and sequencing",
                "Call out the highest-risk step",
            ]
            schema = {"plan": "array", "dependencies": "array", "risks": "array"}
        elif role == "coder" or "code" in prompt_lower:
            mission = "Propose the smallest technically sound implementation path grounded in the real code."
            criteria = [
                "Name the most likely files or modules involved",
                "Prefer minimal safe changes",
                "State how to verify the change",
            ]
            schema = {"files": "array", "approach": "array", "verification": "array"}
        elif role == "synthesizer":
            mission = "Merge team findings into one coherent answer or decision."
            criteria = [
                "Integrate the strongest findings from the team",
                "Resolve contradictions explicitly",
                "Produce a final concise answer",
            ]
            schema = {"summary": "string", "integrated_findings": "array", "next_steps": "array"}
        else:
            mission = "Advance the task with a distinct useful perspective."
            criteria = [
                "Produce one concrete contribution",
                "Avoid duplicating obvious work from other agents",
            ]
            schema = {"contribution": "string", "evidence": "array"}
        if pattern_name == "pipeline":
            mission += " Build on prior work instead of starting from scratch."
        specs.append(
            {
                "base_type": base_type,
                "collaboration_role": role,
                "specialization": role.replace("_", " "),
                "mission": mission,
                "success_criteria": criteria,
                "output_schema": schema,
            }
        )
    return specs


async def _generate_dynamic_agent_specs(
    *,
    prompt: str,
    roles: list[str],
    num_agents: int,
    pattern_name: str,
    shared_prompt: str,
    ctx: ToolContext,
) -> list[Any]:
    try:
        from predacore.agents.engine import AGENT_TYPES, create_dynamic_agent_spec
    except ImportError:
        from src.predacore.agents.engine import AGENT_TYPES, create_dynamic_agent_spec  # type: ignore

    defaults_raw = _default_dynamic_agent_spec_inputs(
        roles,
        num_agents=num_agents,
        prompt=prompt,
        pattern_name=pattern_name,
    )

    def _build_specs(raw_specs: list[dict[str, Any]]) -> list[Any]:
        built = []
        for raw in raw_specs[:num_agents]:
            base_type = str(raw.get("base_type") or "generalist").strip().lower()
            if base_type not in AGENT_TYPES:
                base_type = "generalist"
            built.append(
                create_dynamic_agent_spec(
                    base_type=base_type,
                    collaboration_role=str(
                        raw.get("collaboration_role") or raw.get("role") or base_type
                    ).strip().lower(),
                    specialization=str(
                        raw.get("specialization") or raw.get("collaboration_role") or base_type
                    ).strip(),
                    mission=str(raw.get("mission") or "").strip(),
                    success_criteria=raw.get("success_criteria") or (),
                    output_schema=raw.get("output_schema") or {},
                    memory_scopes=raw.get("memory_scopes") or ("global", "team"),
                    allowed_tools=raw.get("allowed_tools") or (),
                    max_steps=raw.get("max_steps"),
                    temperature=raw.get("temperature"),
                )
            )
        while len(built) < num_agents:
            fallback = defaults_raw[len(built)]
            built.append(
                create_dynamic_agent_spec(
                    base_type=fallback["base_type"],
                    collaboration_role=fallback["collaboration_role"],
                    specialization=fallback["specialization"],
                    mission=fallback["mission"],
                    success_criteria=fallback["success_criteria"],
                    output_schema=fallback["output_schema"],
                )
            )
        return built

    default_specs = _build_specs(defaults_raw)
    if ctx.llm_for_collab is None:
        return default_specs

    allowed_types = ", ".join(sorted(AGENT_TYPES))
    seed_payload = {
        "agents": [
            {
                "base_type": raw["base_type"],
                "collaboration_role": raw["collaboration_role"],
                "specialization": raw["specialization"],
                "mission": raw["mission"],
                "success_criteria": raw["success_criteria"],
                "output_schema": raw["output_schema"],
            }
            for raw in defaults_raw
        ]
    }
    meta_messages = [
        {
            "role": "system",
            "content": (
                "You are PredaCore's agent architect. Produce strict JSON only. "
                "Design a small team of specialized agents for the task. "
                "Return an object with key 'agents', whose value is an array of "
                f"{num_agents} objects. Allowed base_type values: {allowed_types}. "
                "Each object must include: base_type, collaboration_role, specialization, "
                "mission, success_criteria, output_schema. "
                "Keep roles distinct and complementary. No markdown fences."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task:\n{prompt}\n\n"
                f"Pattern: {pattern_name}\n"
                f"Requested roles: {roles[:num_agents]}\n\n"
                "Shared context excerpt:\n"
                f"{_truncate_text(shared_prompt, 2200)}\n\n"
                "Seed shape to improve consistency:\n"
                f"{json.dumps(seed_payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]
    try:
        response = await ctx.llm_for_collab.chat(messages=meta_messages, tools=None)
        parsed = _extract_json_payload(response.get("content", ""))
        if isinstance(parsed, dict):
            parsed_agents = parsed.get("agents")
        elif isinstance(parsed, list):
            parsed_agents = parsed
        else:
            parsed_agents = None
        if isinstance(parsed_agents, list) and parsed_agents:
            return _build_specs(
                [item for item in parsed_agents if isinstance(item, dict)]
            )
    except (RuntimeError, OSError, ValueError, TypeError, json.JSONDecodeError):
        logger.debug("Dynamic agent meta-planning failed; using defaults", exc_info=True)
    return default_specs


async def _build_shared_multi_agent_prompt(
    prompt: str,
    args: dict[str, Any],
    ctx: ToolContext,
    *,
    user_id: str,
    team_id: str,
) -> str:
    sections: list[str] = []

    # Project root — critical for scaffolding tasks. The main agent pins this
    # in ctx.memory["_session_project_root"]; sub-agents must inherit it so
    # they don't re-invent the path and create stray folders mid-session.
    project_root_entry = ctx.memory.get("_session_project_root") if ctx.memory else None
    if project_root_entry and isinstance(project_root_entry, dict):
        project_root = project_root_entry.get("content", "")
        if project_root:
            sections.append(
                "## Session Project Root (HARD CONSTRAINT)\n"
                f"Active project root for this session: `{project_root}`\n\n"
                "ALL file operations MUST use this exact path. Do NOT create "
                "new top-level project folders (~/Projects/xxx, ~/Developer/xxx, "
                "~/xxx) — the main agent has already chosen a location. If the "
                "task implies creating a new project, STOP and ask the user to "
                "confirm the path switch first. Never drift silently."
            )

    explicit_context = _format_explicit_context(args.get("context"))
    if explicit_context:
        sections.append(f"## Explicit Task Context\n{explicit_context}")

    session_id = str(args.get("session_id") or "").strip() or None
    memory_tokens = max(400, min(int(args.get("memory_tokens") or 1200), 2500))
    if ctx.unified_memory is not None:
        try:
            try:
                from predacore.memory import MemoryRetriever
            except ImportError:
                from src.predacore.memory import MemoryRetriever  # type: ignore

            retriever = MemoryRetriever(store=ctx.unified_memory)
            retrieved = await retriever.build_context(
                query=prompt,
                user_id=user_id,
                session_id=session_id,
                max_tokens=memory_tokens,
                scopes=["global"],
            )
            if retrieved:
                sections.append(f"## Retrieved Unified Memory\n{retrieved}")
        except (ImportError, RuntimeError, OSError, ValueError, TypeError):
            logger.debug("Shared multi-agent memory retrieval failed", exc_info=True)

    team_memory = await _recall_team_memories(
        prompt,
        ctx=ctx,
        user_id=user_id,
        team_id=team_id,
    )
    if team_memory:
        sections.append(team_memory)

    recent_tool_memory = _format_recent_tool_memory(ctx.memory)
    if recent_tool_memory:
        sections.append(f"## Recent Tool Memory\n{recent_tool_memory}")

    if not sections:
        return prompt

    return (
        f"{prompt}\n\n---\n\n"
        "Shared context for all collaborating agents. Use it only when relevant. "
        "If it conflicts with the latest task wording, trust the latest task wording "
        "and note the conflict.\n\n"
        f"{chr(10).join(chr(10).join(section.splitlines()) for section in sections)}"
    )


async def handle_multi_agent(args: dict[str, Any], ctx: ToolContext) -> str:
    """Run a task using multi-agent collaboration patterns.

    Supports three optional budget parameters for autonomous runs:

    - ``max_runtime_seconds`` — wall-clock kill switch (hard enforced via
      ``asyncio.wait_for``; clamped 10s..6h). If the team overruns, the
      whole run is cancelled.
    - ``max_iterations_per_agent`` — advisory hint passed into the agent
      prompts; the core-level ``max_tool_iterations`` cap still applies.
    - ``max_cost_usd`` — advisory hint, same treatment.

    Without budgets this behaves exactly as before. With them, the tool
    becomes a safe vehicle for "run autonomously, stop yourself if you
    blow past these limits."
    """
    import asyncio

    current_depth = _DELEGATION_DEPTH.get()
    if current_depth >= _MAX_DELEGATION_DEPTH:
        raise ToolError(
            f"Multi-agent blocked: recursion depth {current_depth} exceeds max {_MAX_DELEGATION_DEPTH}",
            kind=ToolErrorKind.LIMIT_EXCEEDED,
            tool_name="multi_agent",
            recoverable=False,
        )
    _DELEGATION_DEPTH.set(current_depth + 1)

    # Pull the wall-clock cap — clamp to a sane range so a rogue caller
    # can't ask for a 30-day run. 10 seconds floor, 6 hour ceiling.
    raw_timeout = args.get("max_runtime_seconds")
    timeout_s: float | None = None
    if raw_timeout is not None:
        try:
            timeout_s = max(10.0, min(float(raw_timeout), 21_600.0))
        except (TypeError, ValueError):
            timeout_s = None

    try:
        coro = _multi_agent_inner(args, ctx)
        if timeout_s is not None:
            return await asyncio.wait_for(coro, timeout=timeout_s)
        return await coro
    except asyncio.TimeoutError:
        raise ToolError(
            f"multi_agent wall-clock exceeded ({timeout_s:.0f}s) — team cancelled",
            kind=ToolErrorKind.TIMEOUT,
            tool_name="multi_agent",
        ) from None
    finally:
        _DELEGATION_DEPTH.set(current_depth)


async def _multi_agent_inner(args: dict[str, Any], ctx: ToolContext) -> str:
    if ctx.llm_for_collab is None:
        raise subsystem_unavailable("Multi-agent collaboration LLM", tool="multi_agent")
    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        raise missing_param("prompt", tool="multi_agent")

    # Prepend budget hints (advisory — the sub-agents see them but nothing
    # enforces iterations/cost here; the wall-clock cap in handle_multi_agent
    # is the only hard enforcement). Only injected when a caller sets them.
    budget_hints: list[str] = []
    if args.get("max_iterations_per_agent"):
        budget_hints.append(
            f"- Target ≤ {int(args['max_iterations_per_agent'])} tool rounds per agent."
        )
    if args.get("max_runtime_seconds"):
        budget_hints.append(
            f"- Wall-clock cap: {int(args['max_runtime_seconds'])}s total. "
            "If you can't finish cleanly, stop and summarize partial findings."
        )
    if args.get("max_cost_usd"):
        budget_hints.append(
            f"- Estimated spend budget: ${float(args['max_cost_usd']):.2f}. "
            "Favor concise reasoning, avoid redundant tool calls."
        )
    if budget_hints:
        prompt = (
            "[BUDGET — autonomous run, no human waiting]\n"
            + "\n".join(budget_hints)
            + "\n\nTASK:\n"
            + prompt
        )

    try:
        from predacore.agents.collaboration import AgentSpec, AgentTeam, TaskPayload
        from predacore.agents.engine import compile_dynamic_agent_prompt
    except ImportError:
        from src.predacore.agents.collaboration import AgentSpec, AgentTeam, TaskPayload  # type: ignore
        from src.predacore.agents.engine import compile_dynamic_agent_prompt  # type: ignore

    pattern_name = str(args.get("pattern", "fan_out")).lower()
    num_agents = min(int(args.get("num_agents") or 3), 5)
    roles = args.get("agent_roles") or []
    use_daf = bool(args.get("use_daf", False))
    team_id = _resolve_multi_agent_team_id(args)
    user_id = _resolve_multi_agent_user_id(args, ctx)
    session_id = str(args.get("session_id") or "").strip() or None
    team_ttl_hours = args.get("team_ttl_hours") or 72
    default_roles = ["analyst", "creative", "critic", "researcher", "synthesizer"]
    while len(roles) < num_agents:
        roles.append(default_roles[len(roles) % len(default_roles)])
    shared_prompt = await _build_shared_multi_agent_prompt(
        prompt,
        args,
        ctx,
        user_id=user_id,
        team_id=team_id,
    )
    dynamic_specs = await _generate_dynamic_agent_specs(
        prompt=prompt,
        roles=roles[:num_agents],
        num_agents=num_agents,
        pattern_name=pattern_name,
        shared_prompt=shared_prompt,
        ctx=ctx,
    )
    ctx.memory["multi_agent:last_team_id"] = {
        "content": team_id,
        "tags": ["multi_agent", "team_id"],
        "stored_at": time.time(),
    }
    ctx.memory[f"multi_agent:team:{team_id}"] = {
        "content": prompt[:400],
        "tags": ["multi_agent", "team", pattern_name],
        "stored_at": time.time(),
    }
    await _persist_team_memory(
        ctx=ctx,
        user_id=user_id,
        team_id=team_id,
        session_id=session_id,
        content=(
            f"Multi-agent kickoff\n"
            f"Prompt: {prompt}\n"
            f"Pattern: {pattern_name}\n"
            f"Roles: {', '.join(str(role) for role in roles[:num_agents])}\n"
            "Dynamic specs:\n"
            + json.dumps(
                [spec.to_transport_dict() for spec in dynamic_specs],
                ensure_ascii=False,
                indent=2,
            )
        ),
        stage="kickoff",
        pattern=pattern_name,
        importance=3,
        ttl_hours=team_ttl_hours,
    )

    if use_daf:
        try:
            from predacore.agents.daf_bridge import DAFBridge, DAFBridgeConfig
            bridge = DAFBridge(DAFBridgeConfig())
            if bridge.should_use_daf(num_agents, pattern_name):
                daf_results = await bridge.dispatch_multi_agent(
                    shared_prompt,
                    roles[:num_agents],
                    pattern=pattern_name,
                    team_id=team_id,
                    user_id=user_id,
                    session_id=session_id,
                    home_dir=getattr(getattr(ctx, "config", None), "home_dir", None),
                    agent_specs=[
                        {
                            **spec.to_transport_dict(),
                            "compiled_prompt": compile_dynamic_agent_prompt(
                                spec,
                                shared_context=shared_prompt,
                                original_task=prompt,
                            )[1],
                        }
                        for spec in dynamic_specs
                    ],
                )
                parts = [f"Pattern: {pattern_name} (DAF) | Team: {team_id} | Agents: {num_agents}"]
                for r in daf_results:
                    status = "OK" if r.status == "completed" else f"{r.status}: {r.error}"
                    await _persist_team_memory(
                        ctx=ctx,
                        user_id=user_id,
                        team_id=team_id,
                        session_id=session_id,
                        content=r.output or r.error,
                        stage="daf_agent_output",
                        pattern=pattern_name,
                        importance=2,
                        agent_role=r.agent_type,
                        agent_id=r.agent_instance_id or r.agent_type,
                        ttl_hours=team_ttl_hours,
                    )
                    parts.append(
                        f"\n--- Agent {r.agent_type} ({status}, {r.latency_ms:.0f}ms) ---\n{r.output or r.error}"
                    )
                await _persist_team_memory(
                    ctx=ctx,
                    user_id=user_id,
                    team_id=team_id,
                    session_id=session_id,
                    content="\n".join(parts),
                    stage="daf_final_output",
                    pattern=pattern_name,
                    importance=3,
                    ttl_hours=team_ttl_hours,
                )
                return "\n".join(parts)
        except (RuntimeError, ImportError, OSError, ConnectionError) as e:
            logger.warning("DAF dispatch failed, falling back to in-process: %s", e)

    team = AgentTeam()

    for dyn_spec in dynamic_specs:
        role = dyn_spec.collaboration_role
        spec = AgentSpec(
            agent_id=dyn_spec.spec_id,
            agent_type=dyn_spec.base_type,
            capabilities=[role, dyn_spec.base_type],
        )

        async def _make_fn(dynamic_spec: Any) -> Callable:
            async def _agent_fn(task: TaskPayload) -> str:
                worker_context = task.prompt
                live_team_memory = await _recall_team_memories(
                    task.prompt,
                    ctx=ctx,
                    user_id=user_id,
                    team_id=team_id,
                    top_k=3,
                )
                if live_team_memory:
                    worker_context = f"{worker_context}\n\n---\n\n{live_team_memory}"
                system_prompt, user_prompt = compile_dynamic_agent_prompt(
                    dynamic_spec,
                    shared_context=worker_context,
                    original_task=prompt,
                )
                msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                resp = await ctx.llm_for_collab.chat(messages=msgs, tools=None)
                content = resp.get("content", "")
                await _persist_team_memory(
                    ctx=ctx,
                    user_id=user_id,
                    team_id=team_id,
                    session_id=session_id,
                    content=content,
                    stage="agent_output",
                    pattern=pattern_name,
                    importance=2,
                    agent_role=dynamic_spec.collaboration_role,
                    agent_id=dynamic_spec.spec_id,
                    ttl_hours=team_ttl_hours,
                )
                return content
            return _agent_fn

        team.add_agent(spec, await _make_fn(dyn_spec))

    task = TaskPayload(
        prompt=shared_prompt,
        timeout=60.0,
        context={"original_prompt": prompt},
    )

    try:
        if pattern_name == "pipeline":
            result = await team.pipeline(task)
        elif pattern_name == "consensus":
            result = await team.consensus(task)
        elif pattern_name == "supervisor" and num_agents >= 2:
            agents = list(team._agents.keys())
            result = await team.supervise(task, worker_id=agents[0], supervisor_id=agents[1])
        else:
            result = await team.fan_out(task)
    except (RuntimeError, OSError, ConnectionError, asyncio.TimeoutError) as e:
        raise ToolError(
            f"Multi-agent collaboration failed: {e}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="multi_agent",
        ) from e

    parts = [
        f"Pattern: {result.pattern.value} | Team: {team_id} | Agents: {num_agents} | Time: {result.total_latency_ms:.0f}ms"
    ]
    for r in result.results:
        status = "OK" if r.success else f"ERR: {r.error}"
        parts.append(
            f"\n--- Agent {r.agent_id} ({status}, {r.latency_ms:.0f}ms) ---\n{r.output or r.error}"
        )
        if not r.success:
            await _persist_team_memory(
                ctx=ctx,
                user_id=user_id,
                team_id=team_id,
                session_id=session_id,
                content=r.error,
                stage="agent_error",
                pattern=pattern_name,
                importance=2,
                agent_id=r.agent_id,
                ttl_hours=team_ttl_hours,
            )
    if result.final_output and result.final_output != [r.output for r in result.results]:
        parts.append(f"\n=== Final Output ===\n{result.final_output}")
    await _persist_team_memory(
        ctx=ctx,
        user_id=user_id,
        team_id=team_id,
        session_id=session_id,
        content="\n".join(parts),
        stage="final_output",
        pattern=pattern_name,
        importance=3,
        ttl_hours=team_ttl_hours,
    )
    return "\n".join(parts)


async def handle_strategic_plan(args: dict[str, Any], ctx: ToolContext) -> str:
    """Generate a strategic plan for a complex goal."""
    if ctx.mcts_planner is None:
        raise subsystem_unavailable("Strategic planner (MCTS)", tool="strategic_plan")
    goal = str(args.get("goal") or "").strip()
    if not goal:
        raise missing_param("goal", tool="strategic_plan")

    context = args.get("context") or {}
    from uuid import uuid4 as _uuid4
    goal_id = _uuid4()

    try:
        plan = await ctx.mcts_planner.create_plan(goal_id, goal, context)
    except (RuntimeError, OSError, ValueError, asyncio.TimeoutError) as e:
        raise ToolError(
            f"Planning failed: {e}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="strategic_plan",
        ) from e

    if plan is None:
        return "[Planner returned no plan]"

    steps_out = []
    for i, step in enumerate(plan.steps):
        steps_out.append({
            "step": i + 1, "action": step.action_type,
            "description": step.description,
            "parameters": step.parameters if step.parameters else {},
        })

    result = {
        "goal_id": str(goal_id),
        "status": plan.status.value if hasattr(plan.status, "value") else str(plan.status),
        "steps": steps_out,
        "justification": getattr(plan, "justification", ""),
    }
    return json.dumps(result, indent=2, default=str)


async def handle_openclaw_delegate(args: dict[str, Any], ctx: ToolContext) -> str:
    """Delegate a task to an OpenClaw bridge endpoint."""
    current_depth = _DELEGATION_DEPTH.get()
    if current_depth >= _MAX_DELEGATION_DEPTH:
        raise ToolError(
            f"Delegation blocked: recursion depth {current_depth} exceeds max {_MAX_DELEGATION_DEPTH}",
            kind=ToolErrorKind.LIMIT_EXCEEDED,
            tool_name="openclaw_delegate",
            recoverable=False,
        )
    _DELEGATION_DEPTH.set(current_depth + 1)
    try:
        return await _openclaw_delegate_inner(args, ctx)
    finally:
        _DELEGATION_DEPTH.set(current_depth)


async def _openclaw_delegate_inner(args: dict[str, Any], ctx: ToolContext) -> str:
    if not ctx.openclaw_enabled:
        raise ToolError(
            "OpenClaw bridge is disabled in current launch profile",
            kind=ToolErrorKind.UNAVAILABLE,
            tool_name="openclaw_delegate",
            suggestion="Enable openclaw_bridge in your launch profile",
        )
    if not (ctx.config.openclaw.base_url or "").strip():
        raise ToolError(
            "OpenClaw bridge is enabled but OPENCLAW_BRIDGE_URL is not configured",
            kind=ToolErrorKind.UNAVAILABLE,
            tool_name="openclaw_delegate",
            suggestion="Set OPENCLAW_BRIDGE_URL in your .env file",
        )
    if not ctx.openclaw_runtime:
        raise subsystem_unavailable("OpenClaw runtime", tool="openclaw_delegate")

    from predacore.agents.autonomy import KillSwitchActivatedError

    task = str(args.get("task", "")).strip()
    if not task:
        raise missing_param("task", tool="openclaw_delegate")

    mode = str(args.get("mode") or "oneshot").strip() or "oneshot"
    context = args.get("context", {})
    if not isinstance(context, dict):
        context = {"raw_context": context}

    # Inherit the session project root so delegated agents don't re-invent
    # the project path. Same constraint as in _build_shared_multi_agent_prompt.
    project_root_entry = ctx.memory.get("_session_project_root") if ctx.memory else None
    if project_root_entry and isinstance(project_root_entry, dict):
        project_root = project_root_entry.get("content", "")
        if project_root and "project_root" not in context:
            context["project_root"] = project_root
            context["project_root_constraint"] = (
                f"Active project root: {project_root}. All file operations "
                "MUST use this exact path. Do NOT create new top-level project "
                "folders — the main agent has already chosen a location."
            )

    timeout_seconds = int(
        args.get("timeout_seconds") or ctx.config.openclaw.timeout_seconds or 180
    )
    timeout_seconds = max(5, min(timeout_seconds, 3600))
    await_completion = bool(args.get("await_completion", True))
    idempotency_key = str(args.get("idempotency_key") or "").strip()

    meta = {
        "source": "prometheus_predacore",
        "profile": ctx.config.launch.profile,
        "self_evolution": bool(ctx.config.launch.enable_self_evolution),
        "approvals_required": bool(ctx.config.launch.approvals_required),
        "egm_mode": ctx.config.launch.egm_mode,
    }

    try:
        result = await ctx.openclaw_runtime.delegate(
            task=task, context=context, mode=mode,
            timeout_seconds=timeout_seconds,
            await_completion=await_completion,
            idempotency_key=idempotency_key, extra_meta=meta,
        )
    except KillSwitchActivatedError as exc:
        raise ToolError(
            f"OpenClaw delegation blocked by kill switch: {exc}",
            kind=ToolErrorKind.BLOCKED,
            tool_name="openclaw_delegate",
            recoverable=False,
        ) from exc
    except (RuntimeError, OSError, ConnectionError, TimeoutError) as exc:
        raise ToolError(
            f"OpenClaw bridge call failed: {exc}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="openclaw_delegate",
        ) from exc

    memory_key = f"openclaw:{result.get('idempotency_key', '')[:12]}"
    ctx.memory[memory_key] = {
        "content": json.dumps({
            "task": task[:400], "status": result.get("status"),
            "cache_hit": result.get("cache_hit"),
            "request_id": result.get("request_id"),
        }, default=str),
        "tags": ["openclaw", "delegate", ctx.config.launch.profile],
        "stored_at": time.time(),
    }

    out = json.dumps(result, indent=2, default=str)
    return out[:50000] if len(out) > 50000 else out
