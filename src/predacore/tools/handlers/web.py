"""Web handlers: web_search, web_scrape, deep_search, semantic_search."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from ._context import (
    ToolContext,
    ToolError,
    ToolErrorKind,
    blocked,
    missing_param,
    web_cache_get,
    web_cache_put,
)

logger = logging.getLogger(__name__)


async def handle_web_search(args: dict[str, Any], ctx: ToolContext) -> str:
    """Search the web via DuckDuckGo and return formatted results.
    
    Tries the duckduckgo-search library first, falls back to the
    DuckDuckGo Instant Answer API via HTTP. Results are cached for
    5 minutes to avoid redundant network calls.
    
    Args:
        args: ``{"query": str, "max_results": int}``
        ctx: Tool context with HTTP client.
    
    Raises:
        ToolError(MISSING_PARAM): if query is empty.
        ToolError(BLOCKED): if URL fails SSRF validation.
    """
    query = args.get("query", "")
    if not query:
        raise missing_param("query", tool="web_search")
    max_results = max(1, min(int(args.get("max_results") or 5), 10))

    cache_key = f"search:{query}:{max_results}"
    cached = web_cache_get(cache_key)
    if cached:
        return cached

    try:
        from ddgs import DDGS

        results_list: list[str] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "")
                body = r.get("body", "")[:300]
                href = r.get("href", "")
                results_list.append(f"**{title}**\n   {body}\n   {href}")
        if results_list:
            result = "\n\n".join(results_list)
            web_cache_put(cache_key, result)
            return result
    except ImportError:
        logger.debug("ddgs not installed, trying fallback")
    except (ConnectionError, TimeoutError, OSError) as e:
        logger.debug("ddgs failed: %s", e)

    try:
        if ctx.http_with_retry is None:
            return f"[No results found for: {query} — HTTP client not available]"

        from predacore.auth.security import validate_url_ssrf

        search_url = "https://api.duckduckgo.com/"
        try:
            validate_url_ssrf(search_url)
        except ValueError as ssrf_err:
            raise blocked(f"SSRF: {ssrf_err}", tool="web_search") from ssrf_err

        resp = await ctx.http_with_retry(
            "get", search_url,
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
        )
        data = resp.json()
        results: list[str] = []
        if data.get("Abstract"):
            results.append(f"**{data.get('Heading', 'Result')}**: {data['Abstract']}")
        for r in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(r, dict) and r.get("Text"):
                url = r.get("FirstURL", "")
                results.append(f"- {r['Text'][:200]}" + (f"\n  {url}" if url else ""))
        result = "\n".join(results) if results else f"[No results found for: {query}]"
        web_cache_put(cache_key, result)
        return result
    except ToolError:
        raise
    except (ConnectionError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as e:
        raise ToolError(
            f"Search error: {e}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="web_search",
        ) from e


async def handle_web_scrape(args: dict[str, Any], ctx: ToolContext) -> str:
    """Scrape a web page and return cleaned text content.
    
    Fetches the URL, strips HTML tags/scripts/styles, collapses whitespace,
    and truncates to 30KB. Includes SSRF validation and result caching.
    
    Args:
        args: ``{"url": str}``
        ctx: Tool context with HTTP client.
    
    Raises:
        ToolError(MISSING_PARAM): if url is empty.
        ToolError(BLOCKED): if URL fails SSRF validation.
        ToolError(UNAVAILABLE): if HTTP client is not available.
    """
    url = args.get("url", "")
    if not url:
        raise missing_param("url", tool="web_scrape")

    cache_key = f"scrape:{url}"
    cached = web_cache_get(cache_key)
    if cached:
        return cached

    try:
        if ctx.http_with_retry is None:
            raise ToolError(
                "HTTP client not available",
                kind=ToolErrorKind.UNAVAILABLE,
                tool_name="web_scrape",
            )

        from predacore.auth.security import validate_url_ssrf

        try:
            validate_url_ssrf(url)
        except ValueError as ssrf_err:
            raise blocked(f"SSRF: {ssrf_err}", tool="web_scrape") from ssrf_err

        resp = await ctx.http_with_retry("get", url)
        text = resp.text
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 30000:
            text = text[:30000] + "...[truncated]"
        web_cache_put(cache_key, text)
        return text
    except ToolError:
        raise
    except (ConnectionError, TimeoutError, OSError, ValueError) as e:
        raise ToolError(
            f"Scrape error: {e}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="web_scrape",
        ) from e


async def handle_deep_search(args: dict[str, Any], ctx: ToolContext) -> str:
    """Deep research: search web, read top results, synthesize answer with citations."""
    query = str(args.get("query") or "").strip()
    if not query:
        raise missing_param("query", tool="deep_search")

    max_sources = max(1, min(int(args.get("max_sources") or 5), 10))

    from predacore.auth.security import validate_url_ssrf

    search_results: list[dict[str, str]] = []

    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_sources * 2):
                search_results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")[:300],
                })
    except ImportError:
        logger.debug("ddgs not installed, using Instant Answer API")
    except (ConnectionError, TimeoutError, OSError) as e:
        logger.debug("ddgs failed: %s", e)

    if not search_results and ctx.http_with_retry is not None:
        try:
            ddg_url = "https://api.duckduckgo.com/"
            resp = await ctx.http_with_retry(
                "get", ddg_url,
                params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            )
            data = resp.json()
            if data.get("Abstract"):
                search_results.append({
                    "title": data.get("Heading", ""),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data["Abstract"][:300],
                })
            for r in data.get("RelatedTopics", [])[:8]:
                if isinstance(r, dict) and r.get("Text"):
                    search_results.append({
                        "title": r.get("Text", "")[:80],
                        "url": r.get("FirstURL", ""),
                        "snippet": r.get("Text", "")[:300],
                    })
        except (ConnectionError, TimeoutError, OSError, json.JSONDecodeError) as e:
            logger.debug("DuckDuckGo Instant Answer API failed: %s", e)

    if not search_results:
        return f"[No search results found for: {query}]"

    sources_to_read = [r for r in search_results if r.get("url")][:max_sources]
    source_texts: list[dict[str, str]] = []

    async def _fetch_source(source: dict) -> dict[str, str] | None:
        url = source["url"]
        if ctx.http_with_retry is None:
            return None
        try:
            validate_url_ssrf(url)
            resp = await ctx.http_with_retry("get", url)
            text = resp.text
            text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return {"title": source.get("title", ""), "url": url, "content": text[:8000]}
        except (ConnectionError, TimeoutError, OSError, ValueError):
            return None

    tasks = [_fetch_source(s) for s in sources_to_read]
    fetched = await asyncio.gather(*tasks, return_exceptions=True)
    for result in fetched:
        if isinstance(result, dict) and result is not None:
            source_texts.append(result)

    if not source_texts:
        lines = [f"## Search Results for: {query}\n"]
        for i, r in enumerate(search_results[:5], 1):
            lines.append(f"**[{i}]** {r.get('title', 'Result')}")
            lines.append(f"   {r.get('snippet', '')}")
            if r.get("url"):
                lines.append(f"   Source: {r['url']}")
            lines.append("")
        return "\n".join(lines)

    lines = [f"## Deep Search: {query}\n"]
    lines.append(f"*Analyzed {len(source_texts)} sources*\n")
    for i, src in enumerate(source_texts, 1):
        lines.append(f"### [{i}] {src['title'][:100]}")
        lines.append(f"Source: {src['url']}")
        content = src["content"][:2000]
        if content:
            lines.append(f"\n{content}\n")

    remaining = [r for r in search_results if r.get("url") not in
                 {s["url"] for s in source_texts}]
    if remaining:
        lines.append("\n### Additional References")
        for r in remaining[:5]:
            lines.append(f"- {r.get('title', 'Link')}: {r.get('url', '')}")

    output = "\n".join(lines)
    return output[:50000] if len(output) > 50000 else output


def _bm25_score(
    query_terms: list[str],
    doc: str,
    k1: float = 1.5,
    b: float = 0.75,
    avg_dl: float = 500,
    n_docs: int = 1,
    doc_freqs: dict[str, int] | None = None,
) -> float:
    """BM25 scoring with proper IDF."""
    import math as _math
    doc_lower = doc.lower()
    dl = len(doc_lower.split())
    score = 0.0
    for term in query_terms:
        tf = len(re.findall(r'\b' + re.escape(term) + r'\b', doc_lower))
        if tf > 0:
            df = (doc_freqs or {}).get(term, 1)
            idf = _math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / max(avg_dl, 1))
            score += idf * numerator / denominator
    return score


async def handle_semantic_search(args: dict[str, Any], ctx: ToolContext) -> str:
    """Hybrid BM25 + vector search across memory and local files."""
    query = str(args.get("query") or "").strip()
    if not query:
        raise missing_param("query", tool="semantic_search")

    scope = str(args.get("scope") or "all").strip().lower()
    top_k = max(1, min(int(args.get("top_k") or 10), 30))
    search_path = str(args.get("path") or "").strip()
    file_pattern = str(args.get("file_pattern") or "").strip()

    results: list[dict[str, Any]] = []
    query_terms = query.lower().split()

    if scope in ("memory", "all") and ctx.unified_memory:
        try:
            mem_results = await ctx.unified_memory.recall(
                query=query, user_id=str(os.getenv("USER") or "default"), top_k=top_k,
            )
            for mem, vec_score in mem_results:
                bm25 = _bm25_score(query_terms, mem["content"])
                hybrid_score = 0.6 * vec_score + 0.4 * min(bm25 / 5.0, 1.0)
                results.append({
                    "source": "memory", "type": mem.get("memory_type", "fact"),
                    "content": mem["content"][:500], "score": hybrid_score,
                    "id": mem.get("id", ""),
                })
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            logger.debug("Memory search failed: %s", exc)

    if scope in ("files", "all"):
        import glob as _glob
        base_dir = search_path or os.getcwd()
        base_dir = str(Path(base_dir).expanduser().resolve())
        _BLOCKED_SEARCH_DIRS = ["/etc", "/var", "/sys", "/proc", "/dev", "/boot", "/sbin"]
        for blocked_dir in _BLOCKED_SEARCH_DIRS:
            if base_dir.startswith(blocked_dir):
                raise blocked(
                    f"cannot search in system directory: {base_dir}",
                    tool="semantic_search",
                )
        pattern = file_pattern or "*.{py,md,txt,yaml,yml,json,toml,cfg,ini,rst}"

        if "{" in pattern:
            exts = pattern.replace("*.", "").strip("{").strip("}").split(",")
            patterns = [f"**/*.{ext.strip()}" for ext in exts]
        else:
            patterns = [f"**/{pattern}"]

        files_to_search: list[str] = []
        for p in patterns:
            try:
                found = _glob.glob(os.path.join(base_dir, p), recursive=True)
                for fpath in found:
                    if Path(fpath).resolve().is_relative_to(base_dir):
                        files_to_search.append(fpath)
            except OSError:
                pass
        files_to_search = files_to_search[:200]

        for fpath in files_to_search:
            try:
                with open(fpath, errors="replace") as f:
                    content = f.read(20000)
                bm25 = _bm25_score(query_terms, content)
                if bm25 > 0:
                    lines_content = content.split("\n")
                    best_line = ""
                    best_score = 0
                    for line in lines_content:
                        ls = _bm25_score(query_terms, line)
                        if ls > best_score:
                            best_score = ls
                            best_line = line.strip()
                    results.append({
                        "source": "file", "path": fpath,
                        "content": best_line[:500] if best_line else content[:500],
                        "score": min(bm25 / 5.0, 1.0),
                    })
            except (OSError, UnicodeDecodeError):
                continue

    if not results:
        return f"[No results found for: {query}]"

    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:top_k]

    lines = [f"## Semantic Search: {query}\n"]
    for i, r in enumerate(results, 1):
        score_str = f"{r['score']:.3f}"
        if r["source"] == "memory":
            lines.append(f"**[{i}]** Memory ({r.get('type', 'fact')}) — score: {score_str}")
            lines.append(f"   {r['content'][:300]}")
        else:
            rel_path = r.get("path", "")
            try:
                rel_path = os.path.relpath(rel_path)
            except ValueError:
                pass
            lines.append(f"**[{i}]** File: `{rel_path}` — score: {score_str}")
            lines.append(f"   {r['content'][:300]}")
        lines.append("")

    return "\n".join(lines)
