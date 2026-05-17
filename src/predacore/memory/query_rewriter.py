"""
Multi-query rewriter — closes the bi-encoder recall ceiling that HyDE
can't reach.

Single-phrasing semantic recall fails when the user's question is worded
differently from how the stored memory was written. Bi-encoder + BM25 +
fuzzy fusion catches a lot of this; HyDE catches more (it generates a
hypothetical answer and re-recalls). But HyDE only fires on low-confidence
queries — most queries score above its 0.55 threshold and never trigger.

Multi-query rewriter is HyDE done differently: ALWAYS rewrite the query
into N variants via a cheap LLM, fan out parallel recalls, union by id
taking the max score per id. Closes #11 (temporal), #12 (preferences),
#15 (HyDE rarely fires) with one mechanism.

Cost: +1 LLM call per recall (Haiku-class), +N parallel bi-encoder recalls.
Total added latency: ~200-400 ms in practice.

Cached per-query for 5 minutes — repeated identical queries don't pay
the LLM cost twice.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from ..utils.cache import TTLCache, hash_key

logger = logging.getLogger(__name__)


_REWRITE_PROMPT = """You are a query rewriter for a memory search system.

Given a user's question, produce {n} alternative phrasings that might match
different ways the answer could be stored. Vary the form:
- One statement-form (declarative, what the answer looks like)
- One keyword-emphasized (terse, content words only)
- For temporal queries ("yesterday", "last week", "in May"), include the
  time scope explicitly

Output EXACTLY {n} rewrites, ONE per line. No numbering, no quotes, no
explanation. Each rewrite preserves the original intent.

Question: {query}

Rewrites:"""


# Sentinel matched against an LLM response that returned nothing usable.
_EMPTY_REWRITE = []


class QueryRewriter:
    """LLM-backed query rewriter with TTL cache.

    Pass in an llm with an async ``chat(messages=...)`` method (same shape
    HyDE uses). Returns up to ``n`` alternative phrasings of the query;
    falls back to an empty list on any failure (caller treats as "no
    rewrites, single-query recall").
    """

    def __init__(self, llm: Any, *, cache_ttl_seconds: int = 300) -> None:
        self._llm = llm
        self._cache: TTLCache = TTLCache()
        self._cache_ttl = cache_ttl_seconds

    async def rewrite(self, query: str, n: int = 3) -> list[str]:
        """Generate up to ``n`` rewrites of ``query``. Cached for 5 min.

        Returns a list of strings (NOT including the original query —
        the caller is responsible for adding the original when fanning
        out). May return fewer than ``n`` if the LLM produced duplicates
        or empty lines.
        """
        if not query or not query.strip() or self._llm is None or n < 1:
            return []
        cache_key = hash_key("rewrite", query.strip(), str(n))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return list(cached)

        prompt = _REWRITE_PROMPT.format(n=n, query=query.strip())
        try:
            response = await asyncio.wait_for(
                self._llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                ),
                timeout=10.0,
            )
        except (asyncio.TimeoutError, ConnectionError, RuntimeError, ValueError) as exc:
            logger.debug("Query rewriter LLM call failed: %s", exc)
            self._cache.set(cache_key, _EMPTY_REWRITE, ttl_seconds=self._cache_ttl)
            return []

        if isinstance(response, dict):
            content = str(response.get("content") or "").strip()
        else:
            content = str(response or "").strip()
        if not content:
            self._cache.set(cache_key, _EMPTY_REWRITE, ttl_seconds=self._cache_ttl)
            return []

        rewrites = _parse_rewrites(content, n=n, original=query)
        self._cache.set(cache_key, rewrites, ttl_seconds=self._cache_ttl)
        return rewrites


def _parse_rewrites(content: str, *, n: int, original: str) -> list[str]:
    """Extract one rewrite per non-empty line.

    Defensive: strips numbering ("1.", "1)"), bullets ("-", "*"), and
    quotes the LLM may have ignored instructions on. Dedupes against
    the original query and against each other. Returns at most ``n``.
    """
    lines = [line.strip() for line in content.splitlines()]
    cleaned: list[str] = []
    seen = {original.strip().lower()}
    for raw in lines:
        if not raw:
            continue
        # Strip leading list markers
        stripped = re.sub(r"^\s*(?:\d+[\.\)]|\-|\*|•)\s*", "", raw)
        # Strip surrounding quotes
        stripped = stripped.strip("'\"").strip()
        # Skip "Rewrites:" header or other meta-lines
        if not stripped or stripped.lower().endswith(":"):
            continue
        # Skip duplicates (case-insensitive)
        if stripped.lower() in seen:
            continue
        seen.add(stripped.lower())
        cleaned.append(stripped)
        if len(cleaned) >= n:
            break
    return cleaned


__all__ = ["QueryRewriter"]
