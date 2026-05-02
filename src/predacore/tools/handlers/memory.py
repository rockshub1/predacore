"""Memory handlers: memory_store, memory_recall, plus bulk-index pair."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from ._context import (
    ToolContext,
    ToolError,
    ToolErrorKind,
    _lazy_memory_types,
    missing_param,
    subsystem_unavailable,
)

logger = logging.getLogger(__name__)

_MEMORY_KEY_MAX = 256
_MEMORY_CONTENT_MAX = 100_000
_MEMORY_TAGS_MAX = 20


def _normalize_scope(scope: str | None) -> str:
    try:
        from predacore.memory.store import normalize_memory_scope
    except ImportError:
        from src.predacore.memory.store import normalize_memory_scope  # type: ignore
    return normalize_memory_scope(scope)


def _normalize_scope_strict(scope: str | None, *, tool: str) -> str:
    """Strict scope validation for write/read request boundaries.

    Unlike the lenient ``_normalize_scope`` (which silently downgrades
    unknown values to ``"global"`` so legacy rows still surface), this one
    REJECTS unknown scopes so a typo like ``"teaam"`` doesn't masquerade
    as global and accidentally write team-intended data into the global
    bucket. Only use at input boundaries — never on stored metadata.
    """
    try:
        from predacore.memory.store import _VALID_MEMORY_SCOPES
    except ImportError:
        from src.predacore.memory.store import _VALID_MEMORY_SCOPES  # type: ignore
    value = scope.strip().lower() if isinstance(scope, str) else "global"
    if not value:
        return "global"
    if value not in _VALID_MEMORY_SCOPES:
        valid = ", ".join(sorted(_VALID_MEMORY_SCOPES))
        raise ToolError(
            f"Unknown memory scope: {scope!r}. Valid scopes: {valid}",
            kind=ToolErrorKind.INVALID_PARAM,
            tool_name=tool,
            suggestion="Pass one of: global, agent, team, scratch.",
        )
    return value


def _ttl_to_expires_at(ttl_seconds: Any) -> str | None:
    try:
        from predacore.memory.store import future_iso_from_ttl
    except ImportError:
        from src.predacore.memory.store import future_iso_from_ttl  # type: ignore
    return future_iso_from_ttl(ttl_seconds)


def _resolve_team_id(args: dict[str, Any], ctx: ToolContext) -> str:
    explicit = str(args.get("team_id") or "").strip()
    if explicit:
        return explicit
    cached = ctx.memory.get("multi_agent:last_team_id", {})
    return str(cached.get("content") or "").strip()


def _extract_key(mem: dict[str, Any]) -> str:
    """Extract display key from a memory dict, handling metadata as str or dict."""
    import json as _json
    meta = mem.get("metadata", {})
    if isinstance(meta, str):
        try:
            meta = _json.loads(meta)
        except (ValueError, _json.JSONDecodeError):
            meta = {}
    if isinstance(meta, dict):
        return str(meta.get("key") or mem.get("id", "?"))
    return str(mem.get("id", "?"))


async def handle_memory_store(args: dict[str, Any], ctx: ToolContext) -> str:
    """Persist a memory entry across sessions.

    Stores to the unified 3-tier memory system (vector + graph + episode)
    with automatic entity extraction for knowledge-type memories. Falls back
    to legacy MemoryService, then in-memory session cache.

    Args:
        args: ``{"key": str, "content": str, "tags": list[str],
               "memory_type": str, "importance": str, "user_id": str}``
        ctx: Tool context with memory backends.

    Raises:
        ToolError(MISSING_PARAM): if key or content is empty.
        ToolError(LIMIT_EXCEEDED): if key or content exceeds size limits.
    """
    key = str(args.get("key") or "").strip()
    if not key:
        raise missing_param("key", tool="memory_store")
    content = str(args.get("content") or "").strip()
    if not content:
        raise missing_param("content", tool="memory_store")
    tags = args.get("tags", [])

    if len(key) > _MEMORY_KEY_MAX:
        raise ToolError(
            f"Memory key too long: {len(key)} chars, max {_MEMORY_KEY_MAX}",
            kind=ToolErrorKind.LIMIT_EXCEEDED,
            tool_name="memory_store",
        )
    if len(content) > _MEMORY_CONTENT_MAX:
        raise ToolError(
            f"Memory content too large: {len(content)} chars, max {_MEMORY_CONTENT_MAX}",
            kind=ToolErrorKind.LIMIT_EXCEEDED,
            tool_name="memory_store",
        )
    if not isinstance(tags, list):
        tags = []
    tags = [str(t)[:64] for t in tags[:_MEMORY_TAGS_MAX]]
    memory_scope = _normalize_scope_strict(args.get("scope") or "global", tool="memory_store")
    team_id = _resolve_team_id(args, ctx)
    agent_id = str(args.get("agent_id") or "").strip() or None
    if memory_scope in ("team", "scratch") and not team_id:
        raise missing_param("team_id", tool="memory_store")
    if memory_scope == "agent" and not agent_id:
        raise missing_param("agent_id", tool="memory_store")
    session_key = key
    if memory_scope != "global":
        session_key = f"{memory_scope}:{team_id}:{key}"
    ctx.memory[session_key] = {"content": content, "tags": tags, "stored_at": time.time()}

    user_id = str(args.get("user_id") or os.getenv("USER") or "default")
    expires_at = str(args.get("expires_at") or "").strip() or None
    if expires_at is None:
        expires_at = _ttl_to_expires_at(args.get("ttl_seconds"))

    # L5 — auto-tag with current project_id unless caller passed explicit value.
    # Sentinel "all" is invalid for store (a memory belongs to ONE project);
    # treat it as auto-detect.
    project_id_arg = str(args.get("project_id") or "").strip()
    if project_id_arg and project_id_arg.lower() != "all":
        store_project_id = project_id_arg
    else:
        try:
            from predacore.memory.project_id import default_project
            store_project_id = default_project()
        except ImportError:
            store_project_id = "default"

    # Unified memory store (primary)
    if ctx.unified_memory:
        memory_type_raw = str(args.get("memory_type") or "fact").strip().lower()
        importance_raw = str(args.get("importance") or "medium").strip().lower()
        importance_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        importance = importance_map.get(importance_raw, 2)
        try:
            memory_id = await ctx.unified_memory.store(
                content=content, memory_type=memory_type_raw,
                importance=importance, tags=tags, user_id=user_id,
                metadata={
                    "key": key,
                    "source": "predacore.memory_store",
                },
                expires_at=expires_at,
                memory_scope=memory_scope,
                team_id=team_id or None,
                agent_id=agent_id,
                project_id=store_project_id,  # L5
                # Explicit agent/user invocation of the memory_store tool —
                # treat as user_stated with high confidence. Ranking weight
                # is user_stated (0.90) per TRUST_MULTIPLIERS. Callers that
                # want a different classification can instead call
                # ctx.unified_memory.store() directly with trust_source set.
                trust_source="user_stated",
                confidence=0.9,
            )
            entities_msg = ""
            if memory_type_raw == "knowledge" and hasattr(ctx.unified_memory, "extract_entities"):
                try:
                    entities = await ctx.unified_memory.extract_entities(content)
                    if entities:
                        entities_msg = f", entities={len(entities)}"
                except (RuntimeError, ValueError, TypeError) as ent_exc:
                    logger.debug("Entity extraction skipped: %s", ent_exc)
            return (
                f"Stored memory (unified+session): '{key}' "
                f"(id={memory_id}, user={user_id}, scope={memory_scope}, {len(content)} chars{entities_msg})"
            )
        except (RuntimeError, OSError, ValueError, ConnectionError) as exc:
            logger.warning("Unified memory store failed, trying legacy: %s", exc)

    # Legacy MemoryService fallback
    if not ctx.memory_service:
        return f"Stored memory (session): '{key}' ({len(content)} chars)"

    MemoryType, ImportanceLevel = _lazy_memory_types()

    memory_type_raw = str(args.get("memory_type") or "fact").strip().lower()
    memory_type_map = {
        "fact": MemoryType.FACT, "conversation": MemoryType.CONVERSATION,
        "task": MemoryType.TASK, "preference": MemoryType.PREFERENCE,
        "context": MemoryType.CONTEXT, "skill": MemoryType.SKILL,
        "entity": MemoryType.ENTITY, "knowledge": MemoryType.FACT,
    }
    memory_type = memory_type_map.get(memory_type_raw, MemoryType.FACT)

    importance_raw = str(args.get("importance") or "medium").strip().lower()
    importance_map_legacy = {
        "low": ImportanceLevel.LOW, "medium": ImportanceLevel.MEDIUM,
        "high": ImportanceLevel.HIGH, "critical": ImportanceLevel.CRITICAL,
    }
    importance_level = importance_map_legacy.get(importance_raw, ImportanceLevel.MEDIUM)

    try:
        mem = await ctx.memory_service.store(
            content=content, user_id=user_id,
            memory_type=memory_type, importance=importance_level,
            tags=tags, metadata={"key": key, "source": "predacore.memory_store"},
            source="predacore.memory_store",
        )
        return (
            f"Stored memory (persistent+session): '{key}' "
            f"(id={mem.id}, user={user_id}, {len(content)} chars)"
        )
    except (RuntimeError, OSError, ValueError, ConnectionError) as exc:
        logger.warning("Persistent memory store failed, kept session memory: %s", exc)
        return (
            f"Stored memory (session fallback): '{key}' ({len(content)} chars). "
            f"[persistent store failed: {exc}]"
        )


async def handle_memory_recall(args: dict[str, Any], ctx: ToolContext) -> str:
    """Recall memories matching a query via semantic, keyword, or entity search.

    Searches the unified memory system first (vector similarity + BM25 hybrid),
    falls back to legacy MemoryService, then to in-memory session cache with
    simple substring matching.

    Args:
        args: ``{"query": str, "user_id": str, "search_mode": str, "top_k": int}``
              search_mode: "semantic" (default), "keyword", or "entity"
        ctx: Tool context with memory backends.

    Raises:
        ToolError(MISSING_PARAM): if query is empty.
    """
    query = str(args.get("query") or "").strip()
    if not query:
        raise missing_param("query", tool="memory_recall")
    query_lower = query.lower()
    user_id = str(args.get("user_id") or os.getenv("USER") or "default")
    search_mode = str(args.get("search_mode") or "semantic").strip().lower()
    memory_scope = _normalize_scope_strict(args.get("scope") or "global", tool="memory_recall")
    team_id = _resolve_team_id(args, ctx)
    if memory_scope in ("team", "scratch") and not team_id:
        raise missing_param("team_id", tool="memory_recall")
    # Default top_k=10 (was 5). Production benchmark on the predacore source
    # tree showed R@10=1.000 and R@5=0.967 — the engine reliably finds the
    # right file in top-10 even when top-5 misses. Bumping the agent-facing
    # default from 5→10 gets agents the precision the engine already delivers.
    # Cap stays at 20 to bound prompt growth.
    top_k_raw = args.get("top_k") or 10
    try:
        top_k = max(1, min(int(top_k_raw), 20))
    except (ValueError, TypeError):
        top_k = 10

    # L5 — auto-default project filter to current project unless caller
    # passed an explicit value. Pass "all" to disable filter (cross-project
    # recall). Pass a specific project name to query that project's
    # memories. Default = "auto-detect from cwd/git".
    project_arg = str(args.get("project_id") or "").strip()
    if not project_arg:
        try:
            from predacore.memory.project_id import default_project
            recall_project_id: str | None = default_project()
        except ImportError:
            recall_project_id = None
    elif project_arg.lower() == "all":
        recall_project_id = None  # explicit cross-project
    else:
        recall_project_id = project_arg

    # Unified memory store (primary)
    if ctx.unified_memory:
        try:
            if search_mode == "entity" and hasattr(ctx.unified_memory, "get_entity_context"):
                entity_ctx = await ctx.unified_memory.get_entity_context(query)
                if entity_ctx:
                    return entity_ctx
                return f"[No entity context found for: {query}]"

            if search_mode == "keyword" and hasattr(ctx.unified_memory, "_recall_keyword"):
                recalls = await ctx.unified_memory._recall_keyword(
                    query=query,
                    user_id=user_id,
                    top_k=top_k,
                    scopes=[memory_scope],
                    team_id=team_id or None,
                    project_id=recall_project_id,
                )
                if recalls:
                    out: list[str] = []
                    for mem, score in recalls:
                        key = _extract_key(mem)
                        out.append(
                            f"**{key}** (score={score:.3f}, type={mem['memory_type']}): "
                            f"{mem['content'][:300]}"
                        )
                    return "\n\n".join(out)
                return f"[No keyword matches found for: {query}]"

            recalls = await ctx.unified_memory.recall(
                query=query,
                user_id=user_id,
                top_k=top_k,
                scopes=[memory_scope],
                team_id=team_id or None,
                project_id=recall_project_id,
            )
            if recalls:
                out_sem: list[str] = []
                for mem, score in recalls:
                    key = _extract_key(mem)
                    out_sem.append(
                        f"**{key}** (score={score:.3f}, type={mem['memory_type']}): "
                        f"{mem['content'][:300]}"
                    )
                return "\n\n".join(out_sem)
        except (RuntimeError, OSError, ValueError, ConnectionError) as exc:
            logger.warning("Unified recall failed, trying legacy: %s", exc)

    # Legacy MemoryService fallback
    if ctx.memory_service:
        try:
            recalls = await ctx.memory_service.recall(
                query=query, user_id=user_id, top_k=top_k,
            )
            if recalls:
                out_leg = []
                for mem, score in recalls:
                    key = str(mem.metadata.get("key") or mem.id)
                    out_leg.append(
                        f"**{key}** (score={score:.3f}, type={mem.memory_type.value}): "
                        f"{mem.content[:300]}"
                    )
                return "\n\n".join(out_leg)
        except (RuntimeError, OSError, ValueError, ConnectionError) as exc:
            logger.warning("Persistent recall failed, falling back to session cache: %s", exc)

    # Session-memory fallback
    results: list[str] = []
    for key, data in ctx.memory.items():
        if memory_scope != "global" and not key.startswith(f"{memory_scope}:{team_id}:"):
            continue
        if memory_scope == "global" and key.startswith(("team:", "scratch:")):
            continue
        if query_lower in key.lower() or query_lower in data["content"].lower():
            results.append(f"**{key}**: {data['content'][:300]}")
        elif any(query_lower in tag.lower() for tag in data.get("tags", [])):
            results.append(f"**{key}**: {data['content'][:300]}")
    return "\n\n".join(results) if results else f"[No memories found matching: {query}]"


# ---------------------------------------------------------------------------
# Phase W3/W4 — additional memory tool handlers
# ---------------------------------------------------------------------------


async def handle_memory_get(args: dict[str, Any], ctx: ToolContext) -> str:
    """Fetch a single memory row by ID. Returns the row's content + metadata
    formatted as a readable string, or "not found" if the ID doesn't exist."""
    mem_id = str(args.get("id") or "").strip()
    if not mem_id:
        raise missing_param("id", tool="memory_get")

    if not ctx.unified_memory:
        return "[memory subsystem not available]"

    try:
        mem = await ctx.unified_memory.get(mem_id)
    except (RuntimeError, OSError, ValueError, ConnectionError) as exc:
        logger.warning("memory_get failed for id=%s: %s", mem_id, exc)
        return f"[memory_get error: {exc}]"

    if mem is None:
        return f"[no memory with id={mem_id}]"

    # Pull canonical fields (defensive .get() since adapter shapes can vary).
    tags = mem.get("tags") or []
    if isinstance(tags, str):
        try:
            import json as _json
            tags = _json.loads(tags)
        except (ValueError, _json.JSONDecodeError):
            tags = [tags]
    tags_str = ", ".join(str(t) for t in tags) if tags else "(none)"

    lines = [
        f"**id:** {mem.get('id', mem_id)}",
        f"**type:** {mem.get('memory_type', '?')}    "
        f"**importance:** {mem.get('importance', '?')}    "
        f"**trust:** {mem.get('trust_source', '?')}    "
        f"**state:** {mem.get('verification_state', '?')}",
    ]
    src = mem.get("source_path")
    if src:
        lines.append(f"**source:** {src}    chunk_ordinal={mem.get('chunk_ordinal', 0)}")
    if mem.get("created_at"):
        lines.append(f"**created:** {mem['created_at']}")
    lines.append(f"**tags:** {tags_str}")
    lines.append("")
    lines.append(str(mem.get("content", "")))
    return "\n".join(lines)


async def handle_memory_delete(args: dict[str, Any], ctx: ToolContext) -> str:
    """Delete a single memory row by ID. Destructive — no undo. The
    ``requires_confirmation`` flag in the registry means the executor
    should already have asked the user before calling this."""
    mem_id = str(args.get("id") or "").strip()
    if not mem_id:
        raise missing_param("id", tool="memory_delete")

    if not ctx.unified_memory:
        return "[memory subsystem not available]"

    # Capture a preview before we delete so we can report what was removed
    # (helpful for audit + lets the agent describe the action to the user).
    preview = ""
    try:
        mem = await ctx.unified_memory.get(mem_id)
        if mem:
            content = str(mem.get("content", ""))
            preview = content[:80].replace("\n", " ")
            if len(content) > 80:
                preview += "..."
    except (RuntimeError, OSError, ValueError, ConnectionError):
        # Non-fatal — proceed with delete attempt
        pass

    try:
        ok = await ctx.unified_memory.delete(mem_id)
    except (RuntimeError, OSError, ValueError, ConnectionError) as exc:
        logger.warning("memory_delete failed for id=%s: %s", mem_id, exc)
        return f"[memory_delete error: {exc}]"

    if not ok:
        return f"[no memory deleted — id={mem_id} not found]"

    if preview:
        return f"Deleted memory {mem_id} (was: {preview!r})"
    return f"Deleted memory {mem_id}"


async def handle_memory_stats(args: dict[str, Any], ctx: ToolContext) -> str:
    """Return memory subsystem health as a formatted summary."""
    if not ctx.unified_memory:
        return "[memory subsystem not available]"

    try:
        stats = await ctx.unified_memory.get_stats()
    except (RuntimeError, OSError, ValueError, ConnectionError) as exc:
        logger.warning("memory_stats failed: %s", exc)
        return f"[memory_stats error: {exc}]"

    if not isinstance(stats, dict):
        return f"[memory_stats unexpected response shape: {type(stats).__name__}]"

    lines = ["**Memory Stats**", ""]

    # Headline counts
    total = stats.get("total_memories", stats.get("total", "?"))
    lines.append(f"- **total memories:** {total}")

    by_type = stats.get("by_type") or {}
    if by_type:
        type_pairs = ", ".join(f"{k}={v}" for k, v in sorted(by_type.items()))
        lines.append(f"- **by type:** {type_pairs}")

    # Schema / embedding versions
    if "schema_version" in stats:
        lines.append(
            f"- **schema:** v{stats['schema_version']}    "
            f"**embedding:** {stats.get('embedding_version', '?')}"
        )

    # Verification health
    by_vstate = stats.get("by_verification_state") or {}
    if by_vstate:
        vstate_pairs = ", ".join(f"{k}={v}" for k, v in sorted(by_vstate.items()))
        lines.append(f"- **verification states:** {vstate_pairs}")

    # Safety
    safety = stats.get("safety") or {}
    if safety:
        blocked = safety.get("secrets_blocked", 0)
        ignored = safety.get("ignored_paths", 0)
        if blocked or ignored:
            lines.append(
                f"- **safety:** {blocked} secrets blocked, "
                f"{ignored} ignored paths"
            )

    # Healer (if surfaced)
    healer = stats.get("healer") or stats.get("healer_status")
    if isinstance(healer, dict):
        paused = healer.get("paused", False)
        last_snap = healer.get("last_snapshot_at") or "(never)"
        lines.append(
            f"- **healer:** {'PAUSED' if paused else 'running'}, "
            f"last snapshot: {last_snap}"
        )

    return "\n".join(lines)


async def handle_memory_explain(args: dict[str, Any], ctx: ToolContext) -> str:
    """Run recall_explain for a query and return a per-stage trace."""
    query = str(args.get("query") or "").strip()
    if not query:
        raise missing_param("query", tool="memory_explain")

    if not ctx.unified_memory:
        return "[memory subsystem not available]"

    top_k_raw = args.get("top_k") or 10
    try:
        top_k = max(1, min(int(top_k_raw), 50))
    except (ValueError, TypeError):
        top_k = 10

    try:
        trace = await ctx.unified_memory.recall_explain(query=query, top_k=top_k)
    except AttributeError:
        return (
            "[memory_explain unavailable — UnifiedMemoryStore.recall_explain not defined "
            "in this build]"
        )
    except (RuntimeError, OSError, ValueError, ConnectionError) as exc:
        logger.warning("memory_explain failed for query=%r: %s", query, exc)
        return f"[memory_explain error: {exc}]"

    if not isinstance(trace, dict):
        return f"[memory_explain unexpected response shape: {type(trace).__name__}]"

    lines = [
        f"**Memory recall trace for:** {query!r}",
        "",
        f"**Results:** {trace.get('results_count', 0)} hit(s)",
    ]

    for i, hit in enumerate(trace.get("results", []), start=1):
        preview = (hit.get("preview") or "").replace("\n", " ")
        if len(preview) > 100:
            preview = preview[:100] + "..."
        lines.append(
            f"  {i}. id={(hit.get('id') or '?')[:8]}... "
            f"score={hit.get('score', 0):.3f} "
            f"type={hit.get('memory_type', '?')} "
            f"trust={hit.get('trust_source', '?')} | {preview}"
        )

    vstates = trace.get("verification_state_counts") or {}
    if vstates:
        lines.append("")
        lines.append("**Verification state counts (whole DB):**")
        for state, count in sorted(vstates.items()):
            lines.append(f"  - {state}: {count}")

    hidden = trace.get("filtered_by_invariants") or {}
    if hidden:
        total_hidden = hidden.get("total_hidden", 0)
        if total_hidden:
            lines.append("")
            lines.append(
                f"**Filtered by invariants (hidden from recall):** "
                f"{total_hidden} row(s)"
            )
            for state, count in (hidden.get("by_state") or {}).items():
                if count:
                    lines.append(f"  - {state}: {count}")
            lines.append("  _(call with show_stale=True to inspect)_")

    if "embedding_version" in trace:
        lines.append("")
        lines.append(
            f"_schema v{trace.get('schema_version', '?')} · "
            f"embedding {trace.get('embedding_version', '?')}_"
        )
    return "\n".join(lines)


# ── Bulk indexing handlers (PR T5a) ────────────────────────────────────


async def handle_memory_scan_directory(args: dict[str, Any], ctx: ToolContext) -> str:
    """Cheap dry-run: count what bulk_index_directory WOULD walk.

    Use BEFORE memory_bulk_index when the directory is unfamiliar — gives
    you a count and time estimate so you can decide between full bulk,
    scoped bulk (just one subpath), or skipping.

    Args:
        args: ``{"path": str, "ignore": list[str] | None}``
        ctx: Tool context (uses ctx.unified_memory).
    """
    if ctx.unified_memory is None:
        raise subsystem_unavailable(
            "Unified memory store", tool="memory_scan_directory"
        )
    path = str(args.get("path") or "").strip()
    if not path:
        raise missing_param("path", tool="memory_scan_directory")
    ignore_patterns = args.get("ignore")
    if ignore_patterns is not None and not isinstance(ignore_patterns, list):
        raise ToolError(
            "`ignore` must be a list of patterns",
            kind=ToolErrorKind.INVALID_PARAM,
            tool_name="memory_scan_directory",
        )
    try:
        result = await ctx.unified_memory.scan_directory(
            path, ignore_patterns=ignore_patterns,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ToolError(
            f"Scan failed: {exc}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="memory_scan_directory",
        ) from exc
    return json.dumps(result, indent=2, default=str)


async def handle_memory_bulk_index(args: dict[str, Any], ctx: ToolContext) -> str:
    """Walk a directory and reindex every file that survives the filter.

    WHEN TO CALL:
      - User wants you to deeply analyze, refactor, or work on a project
        you don't have indexed (e.g. *"help me refactor the auth layer
        in /path/to/repo"*). One bulk-index up front pays for itself in
        every subsequent recall during the session.
      - User explicitly says *"index this folder"* / *"learn this codebase"*.
      - First time predacore is invoked in a fresh project and the user
        plans to stay there.

    WHEN NOT TO CALL:
      - User just glanced at one file (touch-index handles single edits).
      - Brief project visit — touch-index covers what you actually open.
      - Directory is huge (>5000 files) — call ``memory_scan_directory``
        first, then call this with a scoped ``path`` (e.g. just ``src/auth``).

    Args:
        args: ``{
          "path": str,           # required: directory to walk
          "ignore": list[str],   # optional: extra ignore patterns (gitignore syntax)
          "skip_unchanged": bool # optional: default True; set False to force
                                 #   reindex even when mtime matches
        }``
        ctx: Tool context (uses ctx.unified_memory).

    Returns JSON with files_indexed, files_skipped_unchanged, files_ignored,
    files_failed, chunks_added, elapsed_sec.
    """
    if ctx.unified_memory is None:
        raise subsystem_unavailable(
            "Unified memory store", tool="memory_bulk_index"
        )
    path = str(args.get("path") or "").strip()
    if not path:
        raise missing_param("path", tool="memory_bulk_index")
    ignore_patterns = args.get("ignore")
    if ignore_patterns is not None and not isinstance(ignore_patterns, list):
        raise ToolError(
            "`ignore` must be a list of patterns",
            kind=ToolErrorKind.INVALID_PARAM,
            tool_name="memory_bulk_index",
        )
    skip_unchanged = bool(args.get("skip_unchanged", True))
    user_id = str(args.get("user_id") or os.getenv("USER") or "default")

    # Wire the global status + abort token. memory_index_status reads them;
    # memory_bulk_abort writes the abort flag. The walker polls the token
    # between files and updates status as it goes.
    try:
        from predacore.memory.workspace import (
            get_global_abort_token,
            get_global_status,
            mark_bulk_indexed,
        )
    except ImportError:
        from src.predacore.memory.workspace import (  # type: ignore
            get_global_abort_token,
            get_global_status,
            mark_bulk_indexed,
        )

    abort_token = get_global_abort_token()
    abort_token.reset()  # clean start; user has to abort during THIS run
    status = get_global_status()

    try:
        result = await ctx.unified_memory.bulk_index_directory(
            path,
            user_id=user_id,
            ignore_patterns=ignore_patterns,
            skip_unchanged=skip_unchanged,
            abort_token=abort_token,
            status=status,
        )
    except (OSError, RuntimeError, ValueError, ConnectionError) as exc:
        raise ToolError(
            f"Bulk index failed: {exc}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="memory_bulk_index",
        ) from exc

    if "error" in result:
        raise ToolError(
            f"Bulk index error: {result['error']}",
            kind=ToolErrorKind.INVALID_PARAM,
            tool_name="memory_bulk_index",
        )

    # Write the first-touch marker on a successful (non-aborted) bulk so
    # subsequent daemon starts in this project don't re-prompt.
    if not result.get("aborted") and result.get("files_indexed", 0) > 0:
        try:
            home_dir = getattr(getattr(ctx, "config", None), "home_dir", None)
            mark_bulk_indexed(
                project_id=result.get("project_id", ""),
                files_indexed=result.get("files_indexed", 0),
                chunks_added=result.get("chunks_added", 0),
                home_dir=home_dir,
            )
        except (OSError, ValueError) as exc:
            logger.debug("Failed to write first-touch marker: %s", exc)

    return json.dumps(result, indent=2, default=str)


async def handle_memory_bulk_abort(args: dict[str, Any], ctx: ToolContext) -> str:
    """Request graceful abort of an in-flight bulk index.

    The walker polls the abort token between files; the active call
    returns at the next file boundary with ``aborted=True``. Files
    already indexed remain in the DB. Use this when:

      - User says "stop indexing" / "cancel" mid-run
      - You realize you scoped wrong (was indexing a 50K-file repo when
        user only needed src/auth)
      - Any non-fatal reason to halt cleanly

    Args:
        args: ``{"reason": str}`` (optional reason for the audit trail).
        ctx: Tool context (uses workspace.AbortToken).
    """
    try:
        from predacore.memory.workspace import get_global_abort_token, get_global_status
    except ImportError:
        from src.predacore.memory.workspace import (  # type: ignore
            get_global_abort_token,
            get_global_status,
        )

    reason = str(args.get("reason") or "user requested").strip() or "user requested"
    token = get_global_abort_token()
    status = get_global_status()
    if status.state != "indexing":
        return json.dumps({
            "status": "no_op",
            "message": f"No bulk index is currently running (state={status.state}).",
        }, indent=2)
    token.request_abort(reason=reason)
    return json.dumps({
        "status": "abort_requested",
        "reason": reason,
        "note": "Walker will exit at the next file boundary; partial index is preserved.",
    }, indent=2)


async def handle_memory_index_status(args: dict[str, Any], ctx: ToolContext) -> str:
    """Return the current bulk-index status (state, progress, errors).

    Read-only. Useful when:

      - User asks "is it still indexing?" mid-run
      - You want to check progress before deciding to abort
      - `predacore status` surfaces this same data via the same global
    """
    try:
        from predacore.memory.workspace import get_global_status
    except ImportError:
        from src.predacore.memory.workspace import get_global_status  # type: ignore
    return json.dumps(get_global_status().to_dict(), indent=2, default=str)
