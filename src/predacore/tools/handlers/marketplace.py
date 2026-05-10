"""Marketplace handlers: marketplace_list_skills, marketplace_install_skill, marketplace_invoke_skill."""
from __future__ import annotations

import json
import logging
from typing import Any

from ._context import (
    ToolContext,
    ToolError,
    ToolErrorKind,
    invalid_param,
    missing_param,
    subsystem_unavailable,
)

logger = logging.getLogger(__name__)


async def handle_marketplace_list_skills(args: dict[str, Any], ctx: ToolContext) -> str:
    """List available and installed marketplace skills."""
    if ctx.skill_marketplace is None:
        raise subsystem_unavailable("Skill marketplace", tool="marketplace_list_skills")
    if ctx.resolve_user_id is None:
        raise ToolError(
            "resolve_user_id not configured",
            kind=ToolErrorKind.UNAVAILABLE,
            tool_name="marketplace_list_skills",
        )
    user_id = ctx.resolve_user_id(args)
    search = str(args.get("search") or "").strip() or None
    available = ctx.skill_marketplace.list_available(search=search)
    installed = ctx.skill_marketplace.list_installed(user_id=user_id)
    installed_ids = {s.definition.id for s in installed}

    lines = [f"user_id={user_id}"]
    lines.append(f"installed={len(installed)} available={len(available)}")
    for skill in available:
        status = "installed" if skill.id in installed_ids else "available"
        lines.append(f"- {skill.id} ({status}) :: {skill.description}")
    return "\n".join(lines)


async def handle_marketplace_install_skill(
    args: dict[str, Any], ctx: ToolContext
) -> str:
    """Install a marketplace skill for the current user."""
    if ctx.skill_marketplace is None:
        raise subsystem_unavailable("Skill marketplace", tool="marketplace_install_skill")
    skill_id = str(args.get("skill_id") or "").strip()
    if not skill_id:
        raise missing_param("skill_id", tool="marketplace_install_skill")
    if ctx.resolve_user_id is None:
        raise ToolError(
            "resolve_user_id not configured",
            kind=ToolErrorKind.UNAVAILABLE,
            tool_name="marketplace_install_skill",
        )
    user_id = ctx.resolve_user_id(args)
    config = args.get("config")
    if config is not None and not isinstance(config, dict):
        raise invalid_param("config", "must be an object", tool="marketplace_install_skill")
    ok = ctx.skill_marketplace.install(skill_id, user_id=user_id, config=config)
    if not ok:
        raise ToolError(
            f"Failed to install skill: {skill_id}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="marketplace_install_skill",
            suggestion="Check that the skill_id exists in the marketplace catalog",
        )
    return f"Installed skill '{skill_id}' for user '{user_id}'"


async def handle_marketplace_invoke_skill(
    args: dict[str, Any], ctx: ToolContext
) -> str:
    """Invoke an installed marketplace skill with parameters.

    Phase 18 (2026-05-08): peer-published Flame skills (TrustLevel below
    TRUSTED) auto-isolate via DAFRunner. The orchestrator runs the skill
    in a separate process so a buggy/malicious skill can't compromise
    the daemon. TRUSTED skills run in-process (fast path).

    Detection: peeks at the installed skill's trust_level (Flame skills
    expose this via `genome.trust.level`). Predacore-internal skills
    without a trust field default to in-process — they're first-party.
    """
    if ctx.skill_marketplace is None:
        raise subsystem_unavailable("Skill marketplace", tool="marketplace_invoke_skill")
    skill_id = str(args.get("skill_id") or "").strip()
    params = args.get("params")
    if not skill_id:
        raise missing_param("skill_id", tool="marketplace_invoke_skill")
    if not isinstance(params, dict):
        raise invalid_param("params", "must be an object", tool="marketplace_invoke_skill")
    if ctx.resolve_user_id is None:
        raise ToolError(
            "resolve_user_id not configured",
            kind=ToolErrorKind.UNAVAILABLE,
            tool_name="marketplace_invoke_skill",
        )
    user_id = ctx.resolve_user_id(args)

    # Phase 18: trust-level isolation check.
    trust_level = _resolve_skill_trust(ctx, skill_id, user_id)
    if trust_level is not None:
        try:
            from ...agents import needs_isolation

            if needs_isolation(trust_level):
                logger.info(
                    "marketplace skill %s has trust_level=%s — isolating via DAFRunner",
                    skill_id, trust_level,
                )
                isolated = await _invoke_skill_isolated(
                    ctx=ctx, skill_id=skill_id, user_id=user_id,
                    params=params, trust_level=trust_level,
                )
                if isolated is not None:
                    return isolated
                # fall through to in-process if isolation declined
        except ImportError as exc:
            logger.debug("agents.trust_routing import failed: %s", exc)

    # In-process path — fast, default for TRUSTED skills + first-party
    response = await ctx.skill_marketplace.invoke(
        skill_id=skill_id,
        user_id=user_id,
        params=params,
    )

    # PR1 (W3): close the trust-feedback loop. Every Flame skill
    # invocation feeds (genome_id, success) back to the SkillCrystallizer
    # so TrustScore.local_successes / local_failures accumulate from real
    # use. Auto-quarantine fires when network_score < 30 or local
    # failure rate spikes (see _vendor/common/skill_genome.py:should_quarantine).
    # No-op if crystallizer not wired (stub-ctx tests) or skill_id isn't
    # a Flame genome (predacore-internal skills don't track trust).
    crystallizer = getattr(ctx, "skill_crystallizer", None)
    if crystallizer is not None:
        try:
            success = bool(
                response.get("success", True)
                if isinstance(response, dict) else True
            )
            crystallizer.record_execution(skill_id, success)
        except (AttributeError, KeyError, RuntimeError) as exc:
            logger.debug(
                "skill_crystallizer.record_execution failed for %s: %s",
                skill_id, exc,
            )

    out = json.dumps(response, ensure_ascii=False, indent=2, default=str)
    return out[:50000] if len(out) > 50000 else out


def _resolve_skill_trust(ctx: ToolContext, skill_id: str, user_id: str) -> Any:
    """Look up the skill's trust_level. Returns None if unknown
    (predacore-internal skills don't track trust)."""
    try:
        installed_map = getattr(ctx.skill_marketplace, "_installed", None)
        if not isinstance(installed_map, dict):
            return None
        per_user = installed_map.get(user_id) or {}
        installed = per_user.get(skill_id)
        if installed is None:
            return None
        # Flame skills expose .definition.metadata.trust or .genome.trust.level;
        # internal skills won't have either. Best-effort traversal.
        defn = getattr(installed, "definition", None)
        meta = getattr(defn, "metadata", None) or {}
        if isinstance(meta, dict):
            t = meta.get("trust") or meta.get("trust_level")
            if t is not None:
                return t
        genome = getattr(defn, "genome", None) or meta.get("genome") if isinstance(meta, dict) else None
        if genome is not None:
            trust_obj = getattr(genome, "trust", None)
            if trust_obj is not None:
                return getattr(trust_obj, "level", None)
        return None
    except (AttributeError, KeyError, TypeError):
        return None


async def _invoke_skill_isolated(
    *,
    ctx: ToolContext,
    skill_id: str,
    user_id: str,
    params: dict[str, Any],
    trust_level: Any,
) -> str | None:
    """Run the skill via the orchestrator with `runner="daf"` forced.

    Returns the result string on success, None if isolation declined
    (e.g. orchestrator unavailable) so caller can fall back to in-process.
    """
    try:
        from ...agents import OrchestrationBudget, Orchestrator, OrchestratorConfig
    except ImportError:
        return None

    if ctx.llm_for_collab is None:
        return None  # orchestrator needs an LLM for the lead path

    orch = Orchestrator(
        llm=ctx.llm_for_collab,
        memory=getattr(ctx, "unified_memory", None),
        handler_map=getattr(ctx, "_handler_map", None) or {},
        tool_executor=ctx,
        config=OrchestratorConfig(),
    )
    # The "task" framing for the skill — orchestrator's autonomous
    # pattern wraps execution; the actual call still goes through
    # ctx.skill_marketplace.invoke under DAF process isolation.
    import os as _os

    budget = OrchestrationBudget(
        max_total_tokens=int(_os.getenv("PREDACORE_ORCH_MAX_TOKENS", "100000")),
        max_total_dollars=float(_os.getenv("PREDACORE_ORCH_MAX_DOLLARS", "2.5")),
        max_wall_seconds=int(_os.getenv("PREDACORE_SKILL_MAX_WALL_SECONDS", "300")),
        max_subagents=3,  # skills are single-task units; cap fan-out
    )
    task = (
        f"Execute marketplace skill {skill_id!r} with params:\n"
        f"{json.dumps(params, ensure_ascii=False, indent=2)}\n\n"
        f"This skill has trust_level={trust_level} and must run in "
        f"process isolation. Return the skill's output verbatim."
    )
    try:
        result = await orch.run(
            task=task,
            user_id=user_id,
            session_id="",
            budget=budget,
            runner="daf",  # forced — Phase 18 isolation guarantee
        )
    except Exception as exc:  # noqa: BLE001 — isolation boundary
        logger.warning(
            "isolated skill invocation failed (%s) — falling back to in-process",
            exc, exc_info=True,
        )
        return None

    if not result.success or not result.output.strip():
        return None
    return result.output
