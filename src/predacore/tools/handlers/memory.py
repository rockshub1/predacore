"""Memory handlers: memory_store, memory_recall."""
from __future__ import annotations

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
    memory_scope = _normalize_scope(args.get("scope") or "global")
    team_id = _resolve_team_id(args, ctx)
    agent_id = str(args.get("agent_id") or "").strip() or None
    if memory_scope in ("team", "scratch") and not team_id:
        raise missing_param("team_id", tool="memory_store")
    session_key = key
    if memory_scope != "global":
        session_key = f"{memory_scope}:{team_id}:{key}"
    ctx.memory[session_key] = {"content": content, "tags": tags, "stored_at": time.time()}

    user_id = str(args.get("user_id") or os.getenv("USER") or "default")
    expires_at = str(args.get("expires_at") or "").strip() or None
    if expires_at is None:
        expires_at = _ttl_to_expires_at(args.get("ttl_seconds"))

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
    memory_scope = _normalize_scope(args.get("scope") or "global")
    team_id = _resolve_team_id(args, ctx)
    if memory_scope in ("team", "scratch") and not team_id:
        raise missing_param("team_id", tool="memory_recall")
    top_k_raw = args.get("top_k") or 5
    try:
        top_k = max(1, min(int(top_k_raw), 20))
    except (ValueError, TypeError):
        top_k = 5

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
