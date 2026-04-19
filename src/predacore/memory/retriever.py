"""
PredaCore Memory Retriever — smart multi-source retrieval with context budgeting.

Builds a rich memory context for each LLM call in 5 sections, each with its
own token budget:

1. Preferences       — user preferences, always first (500 tokens)
2. Entity context    — entities mentioned in query + their relations (800)
3. Semantic search   — vector similarity via predacore_core SIMD (1200)
4. Fuzzy matches     — trigram typo-tolerant via predacore_core (400)
5. Recent episodes   — session summaries for temporal continuity (remaining)

predacore_core (Rust) is a HARD dependency — no Python fallbacks.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import predacore_core  # HARD dependency — no fallback

from ..utils.cache import TTLCache, hash_key

logger = logging.getLogger(__name__)

# Rough estimate: 1 token ≈ 4 chars
_CHARS_PER_TOKEN = 4


class MemoryRetriever:
    """
    Builds rich memory context for each LLM call.

    Token budgets (default max_tokens=3000):
        Preferences:      500 tokens  — always included first
        Entity context:    800 tokens  — entities found in the query
        Semantic results: 1200 tokens  — embedding similarity search
        Episode summaries: 500 tokens  — recent session summaries
    """

    def __init__(
        self,
        store: Any,  # UnifiedMemoryStore
        embedding_client: Any = None,
    ) -> None:
        self._store = store
        self._embed = embedding_client
        # Caches to reduce per-message latency
        self._semantic_cache = TTLCache()       # hash(query+user_id) -> results, 60s
        self._entity_cache = TTLCache()         # "all_entities" -> list, 5min
        self._preferences_cache = TTLCache()    # user_id -> prefs, 5min

    async def build_context(
        self,
        query: str,
        user_id: str = "default",
        max_tokens: int = 3000,
        session_id: str | None = None,
        scopes: list[str] | None = None,
        team_id: str | None = None,
    ) -> str:
        """
        Build memory context string for injection into the LLM system prompt.

        Returns a formatted text block or empty string if no relevant memories.
        """
        if not query or not query.strip():
            return ""

        max_chars = max_tokens * _CHARS_PER_TOKEN
        sections: list[str] = []
        chars_used = 0

        # ── 1. Preferences (always included first) ───────────────────
        pref_budget = min(500 * _CHARS_PER_TOKEN, max_chars // 4)
        pref_section = await self._build_preferences_section(user_id, pref_budget)
        if pref_section:
            sections.append(pref_section)
            chars_used += len(pref_section)

        # ── 2. Entity context ────────────────────────────────────────
        entity_budget = min(800 * _CHARS_PER_TOKEN, (max_chars - chars_used) // 3)
        entity_section = await self._build_entity_section(query, entity_budget)
        if entity_section:
            sections.append(entity_section)
            chars_used += len(entity_section)

        # ── 3. Semantic search results ───────────────────────────────
        semantic_budget = min(1200 * _CHARS_PER_TOKEN, (max_chars - chars_used) // 2)
        semantic_section = await self._build_semantic_section(
            query,
            user_id,
            semantic_budget,
            scopes=scopes,
            team_id=team_id,
        )
        if semantic_section:
            sections.append(semantic_section)
            chars_used += len(semantic_section)

        # ── 4. Fuzzy matches (typo-tolerant, via Rust) ───────────────
        fuzzy_budget = min(400 * _CHARS_PER_TOKEN, (max_chars - chars_used) // 3)
        if fuzzy_budget > 200:
            fuzzy_section = await self._build_fuzzy_section(
                query, user_id, fuzzy_budget
            )
            if fuzzy_section:
                sections.append(fuzzy_section)
                chars_used += len(fuzzy_section)

        # ── 5. Recent episode summaries ──────────────────────────────
        episode_budget = max_chars - chars_used
        if episode_budget > 200:
            episode_section = await self._build_episode_section(episode_budget)
            if episode_section:
                sections.append(episode_section)

        if not sections:
            return ""

        return "\n\n".join(sections)

    # ── Section builders ──────────────────────────────────────────────

    async def _build_preferences_section(self, user_id: str, budget: int) -> str:
        """Load user preferences (always included). Cached for 5 minutes."""
        cache_key = hash_key("prefs", user_id)
        cached = self._preferences_cache.get(cache_key)
        if cached is not None:
            results = cached
        else:
            try:
                prefs = await self._store.get_all_memories(
                    memory_type="preference",
                    limit=50,  # Fetch more to account for multi-user filtering
                )
                results = [(p, 1.0) for p in prefs if p.get("user_id") == user_id][:10]
                self._preferences_cache.set(cache_key, results, ttl_seconds=300)
            except (RuntimeError, OSError, ValueError, ConnectionError):
                return ""

        if not results:
            return ""

        lines = ["## User Preferences"]
        chars = len(lines[0])
        for mem, _ in results:
            line = f"- {_truncate(mem['content'], 200)}"
            if chars + len(line) > budget:
                break
            lines.append(line)
            chars += len(line)

        return "\n".join(lines) if len(lines) > 1 else ""

    async def _build_entity_section(self, query: str, budget: int) -> str:
        """Find entities mentioned in the query and load their context."""
        entity_names = await self._extract_entities_from_query(query)
        if not entity_names:
            return ""

        lines = ["## Relevant Entities"]
        chars = len(lines[0])

        for name in entity_names[:5]:  # Max 5 entities
            try:
                ctx = await self._store.get_entity_context(name)
            except (RuntimeError, OSError, ValueError, KeyError):
                continue

            entity = ctx.get("entity")
            if entity is None:
                continue

            # Entity header
            props = entity.get("properties", {})
            prop_str = (
                ", ".join(f"{k}={v}" for k, v in list(props.items())[:3])
                if props
                else ""
            )
            header = f"**{entity['name']}** ({entity['entity_type']})"
            if prop_str:
                header += f" [{prop_str}]"
            header += f" (mentioned {entity['mention_count']}x)"

            if chars + len(header) > budget:
                break
            lines.append(header)
            chars += len(header)

            # Relations
            for rel in ctx.get("relations", [])[:5]:
                if rel["source_entity_id"] == entity["id"]:
                    rel_line = f"  - {rel['relation_type']} → {rel['target_name']}"
                else:
                    rel_line = f"  - ← {rel['relation_type']} {rel['source_name']}"
                if chars + len(rel_line) > budget:
                    break
                lines.append(rel_line)
                chars += len(rel_line)

            # Recent memories about this entity (max 2)
            for mem in ctx.get("memories", [])[:2]:
                snippet = _truncate(mem["content"], 150)
                mem_line = f"  - [{mem['memory_type']}] {snippet}"
                if chars + len(mem_line) > budget:
                    break
                lines.append(mem_line)
                chars += len(mem_line)

        return "\n".join(lines) if len(lines) > 1 else ""

    async def _build_semantic_section(
        self,
        query: str,
        user_id: str,
        budget: int,
        scopes: list[str] | None = None,
        team_id: str | None = None,
    ) -> str:
        """Semantic search for memories related to the query. Cached for 60s."""
        cache_key = hash_key("semantic", query, user_id)
        results = self._semantic_cache.get(cache_key)
        if results is None:
            try:
                results = await self._store.recall(
                    query=query,
                    user_id=user_id,
                    top_k=15,
                    min_importance=1,
                    scopes=scopes,
                    team_id=team_id,
                )
                self._semantic_cache.set(cache_key, results, ttl_seconds=60)
            except (RuntimeError, OSError, ValueError, ConnectionError):
                return ""

        if not results:
            return ""

        # Rerank: combine similarity score with recency and importance
        import math as _math
        import time as _time
        now = _time.time()
        reranked = []
        for mem, sim_score in results:
            if mem.get("memory_type") == "preference":
                continue  # Already in preferences section
            created_at = _to_unix_timestamp(mem.get("created_at"), default=now)
            age_days = max(0.01, (now - created_at) / 86400)
            recency_boost = _math.exp(-0.05 * age_days)  # Half-life ~14 days
            importance = mem.get("importance", 2) / 5.0
            final_score = 0.6 * sim_score + 0.25 * recency_boost + 0.15 * importance
            reranked.append((mem, final_score))
        reranked.sort(key=lambda x: x[1], reverse=True)

        lines = ["## Relevant Memories"]
        chars = len(lines[0])

        for mem, score in reranked:
            snippet = _truncate(mem["content"], 220)
            line = f"- ({score:.2f}) [{mem['memory_type']}] {snippet}"
            if chars + len(line) > budget:
                break
            lines.append(line)
            chars += len(line)

        return "\n".join(lines) if len(lines) > 1 else ""

    async def _build_episode_section(self, budget: int) -> str:
        """Load recent episode summaries for temporal context."""
        try:
            episodes = await self._store.get_recent_episodes(limit=3)
        except (RuntimeError, OSError, ValueError, ConnectionError):
            return ""

        if not episodes:
            return ""

        lines = ["## Recent Sessions"]
        chars = len(lines[0])

        for ep in episodes:
            summary = _truncate(ep["summary"], 200)
            outcome = ep.get("outcome") or "unknown"
            line = f"- [{outcome}] {summary}"
            if chars + len(line) > budget:
                break
            lines.append(line)
            chars += len(line)

        return "\n".join(lines) if len(lines) > 1 else ""

    async def _build_fuzzy_section(
        self, query: str, user_id: str, budget: int
    ) -> str:
        """Fuzzy search for typo-tolerant matching via predacore_core (Rust trigrams)."""
        memories = await self._store.get_all_memories(limit=200)
        if not memories:
            return ""

        user_memories = [m for m in memories if m.get("user_id") == user_id]
        if not user_memories:
            return ""

        contents = [m["content"] for m in user_memories]
        results = predacore_core.fuzzy_search(query, contents, top_k=5, threshold=0.2)
        if not results:
            return ""

        lines = ["## Fuzzy Matches"]
        chars = len(lines[0])

        for idx, score in results:
            mem = user_memories[idx]
            snippet = _truncate(mem["content"], 180)
            line = f"- ({score:.2f}) [{mem.get('memory_type', 'unknown')}] {snippet}"
            if chars + len(line) > budget:
                break
            lines.append(line)
            chars += len(line)

        return "\n".join(lines) if len(lines) > 1 else ""

    # ── Entity extraction ─────────────────────────────────────────────

    async def _extract_entities_from_query(self, query: str) -> list[str]:
        """
        Find known entity names in the query text.

        Simple string matching against the entities table — no LLM call needed.
        The consolidator handles LLM-based entity extraction from conversations.
        """
        cached_entities = self._entity_cache.get("all_entities")
        if cached_entities is not None:
            all_entities = cached_entities
        else:
            try:
                all_entities = await self._store.list_entities(limit=500)
                self._entity_cache.set("all_entities", all_entities, ttl_seconds=300)
            except (RuntimeError, OSError, ValueError, ConnectionError):
                return []

        query_lower = query.lower()
        found = []
        for entity in all_entities:
            name = entity.get("name", "")
            if len(name) >= 2 and name.lower() in query_lower:
                found.append(name)

        # Sort by mention_count (most important first)
        found.sort(
            key=lambda n: next(
                (e["mention_count"] for e in all_entities if e["name"] == n), 0
            ),
            reverse=True,
        )
        return found


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len chars, adding ellipsis if needed."""
    text = " ".join(str(text).split())  # Collapse whitespace
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _to_unix_timestamp(value: Any, default: float) -> float:
    """Normalize float/int/ISO-8601 timestamps for recency calculations."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return default
    return default
