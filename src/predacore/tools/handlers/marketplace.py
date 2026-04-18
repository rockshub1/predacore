"""Marketplace handlers: marketplace_list_skills, marketplace_install_skill, marketplace_invoke_skill."""
from __future__ import annotations

import json
from typing import Any

from ._context import (
    ToolContext,
    ToolError,
    ToolErrorKind,
    missing_param,
    invalid_param,
    subsystem_unavailable,
)


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
    """Invoke an installed marketplace skill with parameters."""
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
    response = await ctx.skill_marketplace.invoke(
        skill_id=skill_id,
        user_id=user_id,
        params=params,
    )
    out = json.dumps(response, ensure_ascii=False, indent=2, default=str)
    return out[:50000] if len(out) > 50000 else out
