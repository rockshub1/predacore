"""Identity handlers: identity_read, identity_update, journal_append."""
from __future__ import annotations

import json
from typing import Any

from ._context import (
    ToolContext,
    ToolError,
    ToolErrorKind,
    invalid_param,
    missing_param,
    resource_not_found,
)


async def handle_identity_read(args: dict[str, Any], ctx: ToolContext) -> str:
    """Read an identity file from the agent's workspace."""
    from ...identity.engine import get_identity_engine

    file_key = args.get("file", "").upper()
    filename = f"{file_key}.md"

    engine = get_identity_engine(ctx.config)

    loaders = {
        "IDENTITY.md": engine.load_identity,
        "SOUL.md": engine.load_soul,
        "USER.md": engine.load_user,
        "JOURNAL.md": engine.load_journal,
        "TOOLS.md": engine.load_tools,
        "MEMORY.md": engine.load_memory,
        "HEARTBEAT.md": engine.load_heartbeat_config,
        "REFLECTION.md": engine.load_reflection_rules,
        "BELIEFS.md": engine.load_beliefs,
        "DECISIONS.md": engine.load_decisions,
        "EVOLUTION.md": engine.load_evolution,
    }

    loader = loaders.get(filename)
    if loader is None:
        raise invalid_param(
            "file",
            "unknown identity file: "
            f"{file_key}. Valid: IDENTITY, SOUL, USER, JOURNAL, TOOLS, MEMORY, "
            "HEARTBEAT, REFLECTION, BELIEFS, DECISIONS, EVOLUTION",
            tool="identity_read",
        )

    content = loader()
    if not content:
        raise resource_not_found(filename, "", tool="identity_read")

    return content


async def handle_identity_update(args: dict[str, Any], ctx: ToolContext) -> str:
    """Write or update an identity file in the agent's workspace."""
    from ...identity.engine import get_identity_engine

    file_key = args.get("file", "").upper()
    filename = f"{file_key}.md"
    content = args.get("content", "")
    reason = args.get("reason", "")

    if not content.strip():
        raise missing_param("content", tool="identity_update")

    engine = get_identity_engine(ctx.config)
    result = engine.write_identity_file(filename, content, reason=reason)

    if result["status"] == "error":
        raise ToolError(
            str(result.get("error", "Unknown error")),
            kind=ToolErrorKind.EXECUTION,
            tool_name="identity_update",
        )

    return json.dumps(result, indent=2)


async def handle_journal_append(args: dict[str, Any], ctx: ToolContext) -> str:
    """Append a timestamped entry to the identity journal."""
    from ...identity.engine import get_identity_engine

    entry = args.get("entry", "")
    if not entry.strip():
        raise missing_param("entry", tool="journal_append")

    engine = get_identity_engine(ctx.config)
    result = engine.append_journal(entry)

    if result["status"] == "error":
        raise ToolError(
            str(result.get("error", "Unknown error")),
            kind=ToolErrorKind.EXECUTION,
            tool_name="journal_append",
        )

    return json.dumps(result, indent=2)
