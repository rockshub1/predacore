"""
PredaCore Tool Resilience — Circuit Breaker, Result Cache, and Execution History.

Phase 6.2 additions for production-grade reliability:
  - ToolCircuitBreaker: Opens after N consecutive failures, auto-resets after cooldown
  - ToolResultCache: LRU + TTL cache for idempotent tool results
  - ExecutionHistory: Ring buffer of recent tool calls for debugging/auditing

Phase 6.5: Uses ToolStatus enums throughout — zero magic strings for status codes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .enums import ToolStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class CircuitState(str, Enum):
    CLOSED = "closed"      # Normal operation — requests flow through
    OPEN = "open"          # Failures exceeded threshold — fast-fail
    HALF_OPEN = "half_open"  # Cooldown passed — allow one probe request


@dataclass
class ToolCircuitBreaker:
    """Per-tool circuit breaker with auto-recovery.

    State machine:
      CLOSED → OPEN        after ``failure_threshold`` consecutive failures
      OPEN → HALF_OPEN     after ``cooldown_seconds`` elapsed
      HALF_OPEN → CLOSED   on first success (probe passed)
      HALF_OPEN → OPEN     on first failure (probe failed, reset cooldown)

    Usage:
        cb = ToolCircuitBreaker()
        if cb.is_open("web_search"):
            return "[web_search circuit open — service unavailable]"
        try:
            result = await run_tool(...)
            cb.record_success("web_search")
        except Exception:
            cb.record_failure("web_search")
    """

    failure_threshold: int = 3
    cooldown_seconds: float = 60.0
    # Per-tool state
    _failures: dict[str, int] = field(default_factory=dict)
    _states: dict[str, CircuitState] = field(default_factory=dict)
    _opened_at: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def state(self, tool_name: str) -> CircuitState:
        """Get current circuit state for a tool."""
        with self._lock:
            return self._get_state(tool_name)

    def _get_state(self, tool_name: str) -> CircuitState:
        """Internal — must be called with lock held."""
        current = self._states.get(tool_name, CircuitState.CLOSED)
        if current == CircuitState.OPEN:
            opened = self._opened_at.get(tool_name, 0)
            if time.time() - opened >= self.cooldown_seconds:
                self._states[tool_name] = CircuitState.HALF_OPEN
                return CircuitState.HALF_OPEN
        return current

    def is_open(self, tool_name: str) -> bool:
        """Check if circuit is OPEN (should fast-fail)."""
        return self.state(tool_name) == CircuitState.OPEN

    def record_success(self, tool_name: str) -> None:
        """Record a successful execution — resets failure count and closes circuit."""
        with self._lock:
            self._failures[tool_name] = 0
            current = self._get_state(tool_name)
            if current in (CircuitState.HALF_OPEN, CircuitState.OPEN):
                self._states[tool_name] = CircuitState.CLOSED
                logger.info(
                    "Circuit breaker CLOSED for '%s' after successful probe",
                    tool_name,
                )

    def record_failure(self, tool_name: str) -> None:
        """Record a failure — may trip the circuit."""
        with self._lock:
            current = self._get_state(tool_name)
            self._failures[tool_name] = self._failures.get(tool_name, 0) + 1
            count = self._failures[tool_name]

            if current == CircuitState.HALF_OPEN:
                # Probe failed — reopen
                self._states[tool_name] = CircuitState.OPEN
                self._opened_at[tool_name] = time.time()
                logger.warning(
                    "Circuit breaker RE-OPENED for '%s' (probe failed)",
                    tool_name,
                )
            elif count >= self.failure_threshold:
                self._states[tool_name] = CircuitState.OPEN
                self._opened_at[tool_name] = time.time()
                logger.warning(
                    "Circuit breaker OPENED for '%s' after %d consecutive failures "
                    "(cooldown=%ds)",
                    tool_name,
                    count,
                    int(self.cooldown_seconds),
                )

    def status(self) -> dict[str, dict[str, Any]]:
        """Return circuit status for all tracked tools."""
        with self._lock:
            result = {}
            for tool_name in set(list(self._states.keys()) + list(self._failures.keys())):
                state = self._get_state(tool_name)
                result[tool_name] = {
                    "state": state.value,
                    "failures": self._failures.get(tool_name, 0),
                    "opened_at": self._opened_at.get(tool_name),
                }
            return result


# ---------------------------------------------------------------------------
# Tool Result Cache
# ---------------------------------------------------------------------------


class ToolResultCache:
    """LRU + TTL cache for idempotent tool results.

    Only tools marked as cacheable get cached. Cache key is derived from
    tool name + serialized arguments.

    Default TTLs:
      read_file: 30s       (file may change, but short cache avoids double-reads)
      list_directory: 30s  (same reasoning)
      web_search: 120s     (search results are fairly stable)
      web_scrape: 300s     (page content is very stable in short term)
      git_context: 15s     (status changes frequently)
      git_find_files: 30s  (repo structure is fairly stable)
    """

    DEFAULT_TTL_MAP: dict[str, int] = {
        "read_file": 30,
        "list_directory": 30,
        "web_search": 120,
        "web_scrape": 300,
        "deep_search": 300,
        "git_context": 15,
        "git_find_files": 30,
        "git_diff_summary": 15,
        "semantic_search": 60,
        "memory_recall": 30,
        "pdf_reader": 120,
    }

    def __init__(
        self,
        max_entries: int = 200,
        ttl_map: dict[str, int] | None = None,
    ):
        self._max_entries = max_entries
        self._ttl_map = ttl_map or self.DEFAULT_TTL_MAP
        self._cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def is_cacheable(self, tool_name: str) -> bool:
        """Check if a tool's results can be cached."""
        return tool_name in self._ttl_map

    def _make_key(self, tool_name: str, args: dict[str, Any]) -> str:
        """Create a cache key from tool name + arguments."""
        args_str = json.dumps(args, sort_keys=True, default=str)
        key_hash = hashlib.md5(
            f"{tool_name}:{args_str}".encode(), usedforsecurity=False
        ).hexdigest()[:16]
        return f"{tool_name}:{key_hash}"

    def get(self, tool_name: str, args: dict[str, Any]) -> str | None:
        """Retrieve cached result if fresh, else None."""
        if not self.is_cacheable(tool_name):
            return None

        key = self._make_key(tool_name, args)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            stored_at, result = entry
            ttl = self._ttl_map.get(tool_name, 30)
            if time.time() - stored_at > ttl:
                # Expired
                self._cache.pop(key, None)
                self._misses += 1
                return None

            # Move to end (LRU refresh)
            self._cache.move_to_end(key)
            self._hits += 1
            logger.debug("Cache HIT for %s (key=%s)", tool_name, key)
            return result

    @staticmethod
    def _is_error_result(result: str) -> bool:
        """Check if a result string indicates an error that should not be cached."""
        if not result:
            return False
        _error_prefixes = ("[Tool error:", "[Tool '", "[Rate limited", "[Unknown tool:")
        for prefix in _error_prefixes:
            if result.startswith(prefix):
                return True
        _error_indicators = ("Error:", "Blocked:", "timed out", "circuit breaker")
        result_lower = result.lower()
        for indicator in _error_indicators:
            if indicator.lower() in result_lower:
                return True
        return False

    def put(self, tool_name: str, args: dict[str, Any], result: str, *, never_cache: bool = False) -> None:
        """Store result in cache.

        Args:
            never_cache: If True, explicitly skip caching this result.
        """
        if never_cache:
            return
        if not self.is_cacheable(tool_name):
            return
        # Don't cache error results
        if self._is_error_result(result):
            return

        key = self._make_key(tool_name, args)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (time.time(), result)
            # Evict oldest if full
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)

    def invalidate(self, tool_name: str | None = None) -> int:
        """Invalidate cache entries. If tool_name given, only that tool's entries."""
        with self._lock:
            if tool_name is None:
                count = len(self._cache)
                self._cache.clear()
                return count
            keys_to_remove = [k for k in self._cache if k.startswith(f"{tool_name}:")]
            for k in keys_to_remove:
                self._cache.pop(k, None)
            return len(keys_to_remove)

    def invalidate_on_write(self, tool_name: str, args: dict[str, Any]) -> None:
        """Invalidate relevant cache entries when a write operation occurs."""
        # write_file invalidates read_file and list_directory for that path
        if tool_name == "write_file":
            self.invalidate("read_file")
            self.invalidate("list_directory")
        elif tool_name == "run_command":
            # Commands might change filesystem — conservative invalidation
            self.invalidate("read_file")
            self.invalidate("list_directory")
            self.invalidate("git_context")
            self.invalidate("git_diff_summary")

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._cache),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{self._hits / total * 100:.1f}%" if total > 0 else "N/A",
                "cacheable_tools": sorted(self._ttl_map.keys()),
            }


# ---------------------------------------------------------------------------
# Execution History
# ---------------------------------------------------------------------------


@dataclass
class ExecutionRecord:
    """Single tool execution record."""
    tool_name: str
    arguments: dict[str, Any]
    result_preview: str  # First 500 chars
    status: str  # ToolStatus value: "ok", "error", "timeout", "circuit_open", "cached"
    elapsed_ms: float
    timestamp: float
    origin: str = "user"
    trace_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize execution record to dictionary."""
        return {
            "tool": self.tool_name,
            "args_preview": json.dumps(self.arguments, default=str)[:200],
            "result_preview": self.result_preview[:300],
            "status": self.status,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "timestamp": self.timestamp,
            "origin": self.origin,
        }


_ERROR_STATUSES = frozenset({ToolStatus.ERROR, ToolStatus.TIMEOUT})


class ExecutionHistory:
    """Ring buffer of recent tool executions for debugging and auditing.

    Keeps the last N executions in memory. Useful for:
    - Debugging LLM tool call patterns
    - Auditing what PredaCore did in a session
    - Performance profiling (which tools are slow?)
    - Error pattern detection
    """

    def __init__(self, max_entries: int = 500):
        self._buffer: deque[ExecutionRecord] = deque(maxlen=max_entries)
        self._lock = threading.Lock()
        self._total_calls = 0
        self._total_errors = 0
        self._tool_counts: dict[str, int] = {}

    def record(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        status: str,
        elapsed_ms: float,
        origin: str = "user",
        trace_id: str = "",
    ) -> None:
        """Record a tool execution."""
        entry = ExecutionRecord(
            tool_name=tool_name,
            arguments=arguments,
            result_preview=result[:500] if result else "",
            status=status,
            elapsed_ms=elapsed_ms,
            timestamp=time.time(),
            origin=origin,
        )
        with self._lock:
            self._buffer.append(entry)
            self._total_calls += 1
            if status in _ERROR_STATUSES:
                self._total_errors += 1
            self._tool_counts[tool_name] = self._tool_counts.get(tool_name, 0) + 1

    def recent(self, n: int = 20) -> list[dict[str, Any]]:
        """Get the N most recent executions."""
        with self._lock:
            entries = list(self._buffer)[-n:]
        return [e.to_dict() for e in entries]

    def for_tool(self, tool_name: str, n: int = 10) -> list[dict[str, Any]]:
        """Get recent executions for a specific tool."""
        with self._lock:
            entries = [e for e in self._buffer if e.tool_name == tool_name][-n:]
        return [e.to_dict() for e in entries]

    def stats(self) -> dict[str, Any]:
        """Return execution statistics."""
        with self._lock:
            recent_errors = sum(
                1 for e in self._buffer if e.status in _ERROR_STATUSES
            )
            avg_latency = (
                sum(e.elapsed_ms for e in self._buffer) / len(self._buffer)
                if self._buffer
                else 0
            )
            slowest = (
                max(self._buffer, key=lambda e: e.elapsed_ms).to_dict()
                if self._buffer
                else None
            )
            top_tools = sorted(
                self._tool_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]

        return {
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "error_rate": f"{self._total_errors / self._total_calls * 100:.1f}%"
            if self._total_calls > 0
            else "N/A",
            "recent_error_count": recent_errors,
            "avg_latency_ms": round(avg_latency, 1),
            "slowest_call": slowest,
            "top_tools": dict(top_tools),
            "buffer_size": len(self._buffer),
        }

    def clear(self) -> None:
        """Clear execution history."""
        with self._lock:
            self._buffer.clear()
            self._total_calls = 0
            self._total_errors = 0
            self._tool_counts.clear()
