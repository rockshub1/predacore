"""
Comprehensive tests for JARVIS tools system — Phase 4 Capabilities Layer.

Tests: enums, registry, dispatcher, resilience, trust policy, middleware,
pipeline, MCP server, health dashboard, and handler context.

Target: 80+ tests covering all core infrastructure.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Imports under test ─────────────────────────────────────────────────

from jarvis.tools.enums import (
    ToolName,
    ToolStatus,
    WRITE_TOOLS,
    READ_ONLY_TOOLS,
    DesktopAction,
    AndroidAction,
    VisionAction,
)
from jarvis.tools.registry import (
    ToolDefinition,
    ToolRegistry,
    BUILTIN_TOOLS_RAW,
    TRUST_POLICIES,
    build_builtin_registry,
    build_full_registry,
)
from jarvis.tools.resilience import (
    CircuitState,
    ToolCircuitBreaker,
    ToolResultCache,
    ExecutionHistory,
    ExecutionRecord,
)
from jarvis.tools.trust_policy import (
    TrustPolicyEvaluator,
    ApprovalContext,
    ApprovalHistory,
    _RISK_MAP,
    _CRITICAL_PATTERNS,
)
from jarvis.tools.middleware import (
    Middleware,
    MiddlewareContext,
    MiddlewareStack,
    LoggingMiddleware,
    MetricsMiddleware,
    AuditTrailMiddleware,
    InputSanitizerMiddleware,
    OutputTruncationMiddleware,
    PerToolRateLimitMiddleware,
    create_default_stack,
    _truncate,
    _percentile,
)
from jarvis.tools.dispatcher import (
    AdaptiveTimeoutTracker,
    _check_ethical_compliance,
    _FORBIDDEN_KEYWORDS,
    _TOOL_ALIAS_MAP,
)
from jarvis.tools.pipeline import (
    PipelineStep,
    PipelineResult,
    ToolPipeline,
    _MAX_PIPELINE_STEPS,
    _MAX_PIPELINE_TIMEOUT,
)
from jarvis.tools.health import HealthDashboard
from jarvis.tools.handlers._context import (
    ToolError,
    ToolErrorKind,
    ToolContext,
    missing_param,
    invalid_param,
    subsystem_unavailable,
    resource_not_found,
    blocked,
    web_cache_get,
    web_cache_put,
    SENSITIVE_READ_PATTERNS,
    SENSITIVE_WRITE_PATHS,
    SENSITIVE_WRITE_FILES,
    _DELEGATION_DEPTH,
    _MAX_DELEGATION_DEPTH,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. ToolName Enum
# ═══════════════════════════════════════════════════════════════════════


class TestToolNameEnum:
    """Tests for ToolName enum — all members, str semantics, frozensets."""

    def test_total_member_count(self):
        """ToolName should have exactly 44 members."""
        assert len(ToolName) == 44

    @pytest.mark.parametrize(
        "member,value",
        [
            (ToolName.READ_FILE, "read_file"),
            (ToolName.WRITE_FILE, "write_file"),
            (ToolName.LIST_DIRECTORY, "list_directory"),
            (ToolName.RUN_COMMAND, "run_command"),
            (ToolName.PYTHON_EXEC, "python_exec"),
            (ToolName.EXECUTE_CODE, "execute_code"),
            (ToolName.WEB_SEARCH, "web_search"),
            (ToolName.WEB_SCRAPE, "web_scrape"),
            (ToolName.DEEP_SEARCH, "deep_search"),
            (ToolName.SEMANTIC_SEARCH, "semantic_search"),
            (ToolName.BROWSER_CONTROL, "browser_control"),
            (ToolName.MEMORY_STORE, "memory_store"),
            (ToolName.MEMORY_RECALL, "memory_recall"),
            (ToolName.SPEAK, "speak"),
            (ToolName.VOICE_NOTE, "voice_note"),
            (ToolName.DESKTOP_CONTROL, "desktop_control"),
            (ToolName.SCREEN_VISION, "screen_vision"),
            (ToolName.ANDROID_CONTROL, "android_control"),
            (ToolName.MULTI_AGENT, "multi_agent"),
            (ToolName.STRATEGIC_PLAN, "strategic_plan"),
            (ToolName.OPENCLAW_DELEGATE, "openclaw_delegate"),
            (ToolName.MARKETPLACE_LIST, "marketplace_list_skills"),
            (ToolName.MARKETPLACE_INSTALL, "marketplace_install_skill"),
            (ToolName.MARKETPLACE_INVOKE, "marketplace_invoke_skill"),
            (ToolName.GIT_CONTEXT, "git_context"),
            (ToolName.GIT_DIFF_SUMMARY, "git_diff_summary"),
            (ToolName.GIT_COMMIT_SUGGEST, "git_commit_suggest"),
            (ToolName.GIT_FIND_FILES, "git_find_files"),
            (ToolName.GIT_SEMANTIC_SEARCH, "git_semantic_search"),
            (ToolName.IMAGE_GEN, "image_gen"),
            (ToolName.PDF_READER, "pdf_reader"),
            (ToolName.DIAGRAM, "diagram"),
            (ToolName.IDENTITY_READ, "identity_read"),
            (ToolName.IDENTITY_UPDATE, "identity_update"),
            (ToolName.JOURNAL_APPEND, "journal_append"),
            (ToolName.BOOTSTRAP_COMPLETE, "bootstrap_complete"),
            (ToolName.CRON_TASK, "cron_task"),
            (ToolName.TOOL_PIPELINE, "tool_pipeline"),
            (ToolName.HIVEMIND_STATUS, "hivemind_status"),
            (ToolName.HIVEMIND_SYNC, "hivemind_sync"),
            (ToolName.SKILL_EVOLVE, "skill_evolve"),
            (ToolName.SKILL_SCAN, "skill_scan"),
            (ToolName.SKILL_ENDORSE, "skill_endorse"),
            (ToolName.TOOL_STATS, "tool_stats"),
        ],
    )
    def test_member_values(self, member, value):
        """Each ToolName member should have the correct string value."""
        assert member.value == value

    def test_str_enum_semantics(self):
        """ToolName extends (str, Enum) so plain string comparison works."""
        assert ToolName.READ_FILE == "read_file"
        assert "read_file" == ToolName.READ_FILE
        assert ToolName.RUN_COMMAND in {"run_command", "other"}

    def test_lookup_by_value(self):
        """ToolName('value') should return the correct member."""
        assert ToolName("web_search") is ToolName.WEB_SEARCH

    def test_lookup_invalid_value_raises(self):
        """ToolName with unknown value should raise ValueError."""
        with pytest.raises(ValueError):
            ToolName("nonexistent_tool")


class TestToolStatusEnum:
    """Tests for ToolStatus enum — 9 standard status codes."""

    def test_total_member_count(self):
        assert len(ToolStatus) == 9

    @pytest.mark.parametrize(
        "member,value",
        [
            (ToolStatus.OK, "ok"),
            (ToolStatus.ERROR, "error"),
            (ToolStatus.TIMEOUT, "timeout"),
            (ToolStatus.CACHED, "cached"),
            (ToolStatus.CIRCUIT_OPEN, "circuit_open"),
            (ToolStatus.RATE_LIMITED, "rate_limited"),
            (ToolStatus.BLOCKED, "blocked"),
            (ToolStatus.DENIED, "denied"),
            (ToolStatus.UNKNOWN_TOOL, "unknown_tool"),
        ],
    )
    def test_status_values(self, member, value):
        assert member.value == value

    def test_str_enum_semantics(self):
        assert ToolStatus.OK == "ok"


class TestWriteAndReadOnlySets:
    """Tests for WRITE_TOOLS and READ_ONLY_TOOLS frozensets."""

    def test_write_tools_contents(self):
        assert ToolName.WRITE_FILE in WRITE_TOOLS
        assert ToolName.RUN_COMMAND in WRITE_TOOLS
        assert ToolName.PYTHON_EXEC in WRITE_TOOLS
        assert ToolName.EXECUTE_CODE in WRITE_TOOLS
        assert len(WRITE_TOOLS) == 4

    def test_read_only_tools_count(self):
        assert len(READ_ONLY_TOOLS) == 16

    def test_read_only_does_not_intersect_write(self):
        """Read-only and write tools should be disjoint."""
        assert WRITE_TOOLS & READ_ONLY_TOOLS == frozenset()

    def test_read_only_specific_members(self):
        for tool in [
            ToolName.READ_FILE,
            ToolName.LIST_DIRECTORY,
            ToolName.WEB_SEARCH,
            ToolName.MEMORY_RECALL,
            ToolName.SCREEN_VISION,
            ToolName.TOOL_STATS,
        ]:
            assert tool in READ_ONLY_TOOLS


# ═══════════════════════════════════════════════════════════════════════
# 2. ToolDefinition + ToolRegistry
# ═══════════════════════════════════════════════════════════════════════


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_construction_minimal(self):
        td = ToolDefinition(name="test", description="A test", parameters={})
        assert td.name == "test"
        assert td.category == "general"
        assert td.cost_estimate == "free"
        assert td.parallelizable is True
        assert td.requires_confirmation is False
        assert td.timeout_default == 30

    def test_construction_full(self):
        td = ToolDefinition(
            name="risky",
            description="A risky tool",
            parameters={"type": "object"},
            category="shell",
            cost_estimate="high",
            parallelizable=False,
            requires_confirmation=True,
            timeout_default=120,
        )
        assert td.category == "shell"
        assert td.cost_estimate == "high"
        assert td.parallelizable is False
        assert td.requires_confirmation is True
        assert td.timeout_default == 120

    def test_to_openai_dict(self):
        td = ToolDefinition(
            name="demo",
            description="Demo tool",
            parameters={"type": "object", "properties": {"x": {"type": "int"}}},
        )
        d = td.to_openai_dict()
        assert d["name"] == "demo"
        assert d["description"] == "Demo tool"
        assert "x" in d["parameters"]["properties"]


class TestToolRegistry:
    """Tests for ToolRegistry methods."""

    def _make_registry(self) -> ToolRegistry:
        r = ToolRegistry()
        r.register(ToolDefinition(name="a", description="A", parameters={}, category="cat1"))
        r.register(ToolDefinition(name="b", description="B", parameters={}, category="cat2"))
        r.register(ToolDefinition(
            name="c", description="C", parameters={}, category="cat1", parallelizable=False,
        ))
        return r

    def test_register_and_get(self):
        r = self._make_registry()
        assert r.get("a") is not None
        assert r.get("a").name == "a"

    def test_get_missing_returns_none(self):
        r = self._make_registry()
        assert r.get("nonexistent") is None

    def test_has(self):
        r = self._make_registry()
        assert r.has("a") is True
        assert r.has("nonexistent") is False

    def test_list_all(self):
        r = self._make_registry()
        assert len(r.list_all()) == 3

    def test_list_names(self):
        r = self._make_registry()
        names = r.list_names()
        assert "a" in names
        assert "b" in names
        assert "c" in names

    def test_list_by_category(self):
        r = self._make_registry()
        cat1_tools = r.list_by_category("cat1")
        assert len(cat1_tools) == 2
        names = [t.name for t in cat1_tools]
        assert "a" in names
        assert "c" in names

    def test_list_by_category_empty(self):
        r = self._make_registry()
        assert r.list_by_category("nonexistent") == []

    def test_get_categories(self):
        r = self._make_registry()
        cats = r.get_categories()
        assert "cat1" in cats
        assert "cat2" in cats

    def test_get_parallelizable(self):
        r = self._make_registry()
        par = r.get_parallelizable()
        assert "a" in par
        assert "b" in par
        assert "c" not in par  # parallelizable=False

    def test_get_all_definitions_format(self):
        r = self._make_registry()
        defs = r.get_all_definitions()
        assert len(defs) == 3
        assert all("name" in d and "description" in d for d in defs)

    def test_len_and_contains(self):
        r = self._make_registry()
        assert len(r) == 3
        assert "a" in r
        assert "z" not in r

    def test_register_raw(self):
        r = ToolRegistry()
        r.register_raw(
            {"name": "raw_tool", "description": "Raw", "parameters": {}},
            category="test",
            timeout_default=60,
        )
        td = r.get("raw_tool")
        assert td is not None
        assert td.category == "test"
        assert td.timeout_default == 60

    def test_register_raw_invalid_type(self):
        r = ToolRegistry()
        with pytest.raises(TypeError, match="must be a dict"):
            r.register_raw("not_a_dict")

    def test_register_raw_missing_name(self):
        r = ToolRegistry()
        with pytest.raises(ValueError, match="non-empty 'name'"):
            r.register_raw({"description": "no name"})

    def test_register_raw_invalid_parameters_type(self):
        r = ToolRegistry()
        with pytest.raises(TypeError, match="parameters must be a dict"):
            r.register_raw({"name": "bad", "parameters": "not_a_dict"})

    def test_register_duplicate_same_category(self):
        """Re-registering same tool should overwrite but not duplicate in category."""
        r = ToolRegistry()
        r.register(ToolDefinition(name="x", description="V1", parameters={}, category="cat"))
        r.register(ToolDefinition(name="x", description="V2", parameters={}, category="cat"))
        assert r.get("x").description == "V2"
        # Category list should not have duplicates
        assert r.list_by_category("cat") == [r.get("x")]


class TestBuiltinToolsRaw:
    """Tests for the BUILTIN_TOOLS_RAW list and build functions."""

    def test_builtin_tools_raw_is_list(self):
        assert isinstance(BUILTIN_TOOLS_RAW, list)
        assert len(BUILTIN_TOOLS_RAW) > 25

    def test_builtin_tools_each_is_6tuple(self):
        for item in BUILTIN_TOOLS_RAW:
            assert len(item) == 6, f"Item should be 6-tuple: {item[0].get('name', '?')}"

    def test_build_builtin_registry(self):
        reg = build_builtin_registry()
        assert len(reg) == len(BUILTIN_TOOLS_RAW)
        assert reg.has("read_file")
        assert reg.has("web_search")
        assert reg.has("tool_stats")

    def test_build_full_registry_includes_flame(self):
        reg = build_full_registry(include_flame=True)
        assert reg.has("hivemind_status")
        assert reg.has("skill_evolve")


class TestTrustPolicies:
    """Tests for TRUST_POLICIES dict — yolo / normal / paranoid."""

    def test_three_levels_exist(self):
        assert "yolo" in TRUST_POLICIES
        assert "normal" in TRUST_POLICIES
        assert "paranoid" in TRUST_POLICIES

    def test_yolo_auto_approves_all(self):
        assert "*" in TRUST_POLICIES["yolo"]["auto_approve_tools"]
        assert TRUST_POLICIES["yolo"]["require_confirmation"] == []

    def test_normal_requires_confirmation_for_writes(self):
        confirm_list = TRUST_POLICIES["normal"]["require_confirmation"]
        assert "write_file" in confirm_list
        assert "run_command" in confirm_list
        assert "python_exec" in confirm_list

    def test_paranoid_confirms_everything(self):
        assert "*" in TRUST_POLICIES["paranoid"]["require_confirmation"]
        assert TRUST_POLICIES["paranoid"]["auto_approve_tools"] == []

    def test_each_policy_has_required_keys(self):
        required = {"description", "require_confirmation", "auto_approve_tools"}
        for name, policy in TRUST_POLICIES.items():
            for key in required:
                assert key in policy, f"Policy '{name}' missing key '{key}'"


# ═══════════════════════════════════════════════════════════════════════
# 3. ToolCircuitBreaker
# ═══════════════════════════════════════════════════════════════════════


class TestToolCircuitBreaker:
    """Tests for circuit breaker state machine."""

    def test_initial_state_is_closed(self):
        cb = ToolCircuitBreaker()
        assert cb.state("any_tool") == CircuitState.CLOSED
        assert not cb.is_open("any_tool")

    def test_closed_to_open_after_threshold(self):
        cb = ToolCircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure("web_search")
        assert cb.state("web_search") == CircuitState.OPEN
        assert cb.is_open("web_search")

    def test_failures_below_threshold_stay_closed(self):
        cb = ToolCircuitBreaker(failure_threshold=3)
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        assert cb.state("web_search") == CircuitState.CLOSED

    def test_open_to_half_open_after_cooldown(self):
        cb = ToolCircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure("tool_a")
        assert cb.state("tool_a") == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.state("tool_a") == CircuitState.HALF_OPEN

    def test_half_open_to_closed_on_success(self):
        cb = ToolCircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure("tool_a")
        time.sleep(0.02)
        assert cb.state("tool_a") == CircuitState.HALF_OPEN
        cb.record_success("tool_a")
        assert cb.state("tool_a") == CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self):
        cb = ToolCircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure("tool_a")
        time.sleep(0.02)
        assert cb.state("tool_a") == CircuitState.HALF_OPEN
        cb.record_failure("tool_a")
        assert cb.state("tool_a") == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        cb = ToolCircuitBreaker(failure_threshold=3)
        cb.record_failure("tool_a")
        cb.record_failure("tool_a")
        cb.record_success("tool_a")
        # After reset, need 3 more failures to trip
        cb.record_failure("tool_a")
        cb.record_failure("tool_a")
        assert cb.state("tool_a") == CircuitState.CLOSED

    def test_per_tool_isolation(self):
        cb = ToolCircuitBreaker(failure_threshold=2)
        cb.record_failure("tool_a")
        cb.record_failure("tool_a")
        assert cb.is_open("tool_a")
        assert not cb.is_open("tool_b")

    def test_status_returns_all_tracked(self):
        cb = ToolCircuitBreaker()
        cb.record_failure("x")
        cb.record_success("y")
        status = cb.status()
        assert "x" in status
        assert "y" in status

    def test_thread_safety(self):
        """Concurrent failures should not corrupt state."""
        cb = ToolCircuitBreaker(failure_threshold=100)
        errors = []

        def fail_many():
            try:
                for _ in range(50):
                    cb.record_failure("concurrent")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=fail_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        # 4 threads x 50 = 200 failures
        assert cb._failures.get("concurrent", 0) == 200


# ═══════════════════════════════════════════════════════════════════════
# 4. ToolResultCache
# ═══════════════════════════════════════════════════════════════════════


class TestToolResultCache:
    """Tests for LRU + TTL result cache."""

    def test_put_and_get(self):
        cache = ToolResultCache(max_entries=10)
        cache.put("read_file", {"path": "/a"}, "contents")
        result = cache.get("read_file", {"path": "/a"})
        assert result == "contents"

    def test_get_uncacheable_tool_returns_none(self):
        cache = ToolResultCache(max_entries=10)
        cache.put("write_file", {"path": "/a"}, "result")
        assert cache.get("write_file", {"path": "/a"}) is None

    def test_ttl_expiry(self):
        cache = ToolResultCache(max_entries=10, ttl_map={"read_file": 0})
        cache.put("read_file", {"path": "/a"}, "data")
        time.sleep(0.01)
        assert cache.get("read_file", {"path": "/a"}) is None

    def test_lru_eviction(self):
        cache = ToolResultCache(max_entries=2)
        cache.put("read_file", {"path": "/a"}, "a")
        cache.put("read_file", {"path": "/b"}, "b")
        cache.put("read_file", {"path": "/c"}, "c")  # evicts /a
        assert cache.get("read_file", {"path": "/a"}) is None
        assert cache.get("read_file", {"path": "/b"}) == "b"

    def test_error_results_not_cached(self):
        cache = ToolResultCache(max_entries=10)
        cache.put("read_file", {"path": "/a"}, "[Tool error: boom]")
        assert cache.get("read_file", {"path": "/a"}) is None

    def test_never_cache_flag(self):
        cache = ToolResultCache(max_entries=10)
        cache.put("read_file", {"path": "/a"}, "data", never_cache=True)
        assert cache.get("read_file", {"path": "/a"}) is None

    def test_invalidate_all(self):
        cache = ToolResultCache(max_entries=10)
        cache.put("read_file", {"path": "/a"}, "a")
        cache.put("web_search", {"query": "test"}, "results")
        count = cache.invalidate()
        assert count == 2
        assert cache.get("read_file", {"path": "/a"}) is None

    def test_invalidate_specific_tool(self):
        cache = ToolResultCache(max_entries=10)
        cache.put("read_file", {"path": "/a"}, "a")
        cache.put("web_search", {"query": "test"}, "results")
        count = cache.invalidate("read_file")
        assert count == 1
        assert cache.get("web_search", {"query": "test"}) == "results"

    def test_invalidate_on_write_file(self):
        cache = ToolResultCache(max_entries=10)
        cache.put("read_file", {"path": "/a"}, "a")
        cache.put("list_directory", {"path": "/"}, "listing")
        cache.invalidate_on_write("write_file", {"path": "/a"})
        assert cache.get("read_file", {"path": "/a"}) is None
        assert cache.get("list_directory", {"path": "/"}) is None

    def test_invalidate_on_run_command(self):
        cache = ToolResultCache(max_entries=10)
        cache.put("git_context", {}, "context")
        cache.invalidate_on_write("run_command", {"command": "ls"})
        assert cache.get("git_context", {}) is None

    def test_stats(self):
        cache = ToolResultCache(max_entries=10)
        cache.put("read_file", {"path": "/a"}, "a")
        cache.get("read_file", {"path": "/a"})  # hit
        cache.get("read_file", {"path": "/b"})  # miss
        stats = cache.stats()
        assert stats["entries"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert "hit_rate" in stats

    def test_is_cacheable(self):
        cache = ToolResultCache()
        assert cache.is_cacheable("read_file") is True
        assert cache.is_cacheable("web_search") is True
        assert cache.is_cacheable("write_file") is False
        assert cache.is_cacheable("run_command") is False

    def test_is_error_result_detection(self):
        assert ToolResultCache._is_error_result("[Tool error: something]") is True
        assert ToolResultCache._is_error_result("[Rate limited — wait]") is True
        assert ToolResultCache._is_error_result("normal result") is False
        assert ToolResultCache._is_error_result("") is False
        assert ToolResultCache._is_error_result("Contains Error: in text") is True


# ═══════════════════════════════════════════════════════════════════════
# 5. ExecutionHistory
# ═══════════════════════════════════════════════════════════════════════


class TestExecutionHistory:
    """Tests for ring buffer execution history."""

    def test_record_and_recent(self):
        h = ExecutionHistory(max_entries=10)
        h.record("web_search", {"q": "test"}, "results", "ok", 100.0)
        recent = h.recent(5)
        assert len(recent) == 1
        assert recent[0]["tool"] == "web_search"

    def test_ring_buffer_overflow(self):
        h = ExecutionHistory(max_entries=3)
        for i in range(5):
            h.record(f"tool_{i}", {}, "r", "ok", 10.0)
        recent = h.recent(10)
        assert len(recent) == 3
        # Oldest entries (tool_0, tool_1) should be evicted
        tools = [r["tool"] for r in recent]
        assert "tool_0" not in tools
        assert "tool_4" in tools

    def test_for_tool_filter(self):
        h = ExecutionHistory(max_entries=100)
        h.record("read_file", {}, "r", "ok", 10.0)
        h.record("web_search", {}, "r", "ok", 20.0)
        h.record("read_file", {}, "r", "ok", 30.0)
        results = h.for_tool("read_file")
        assert len(results) == 2

    def test_stats(self):
        h = ExecutionHistory(max_entries=100)
        h.record("a", {}, "r", "ok", 100.0)
        h.record("b", {}, "r", "error", 200.0)
        h.record("a", {}, "r", "ok", 50.0)
        stats = h.stats()
        assert stats["total_calls"] == 3
        assert stats["total_errors"] == 1
        assert stats["buffer_size"] == 3
        assert stats["top_tools"]["a"] == 2

    def test_stats_with_timeout(self):
        h = ExecutionHistory(max_entries=100)
        h.record("a", {}, "r", "timeout", 5000.0)
        stats = h.stats()
        assert stats["total_errors"] == 1

    def test_clear(self):
        h = ExecutionHistory(max_entries=100)
        h.record("a", {}, "r", "ok", 10.0)
        h.clear()
        assert h.stats()["total_calls"] == 0
        assert h.stats()["buffer_size"] == 0

    def test_execution_record_to_dict(self):
        rec = ExecutionRecord(
            tool_name="test",
            arguments={"key": "value"},
            result_preview="result",
            status="ok",
            elapsed_ms=42.5,
            timestamp=time.time(),
        )
        d = rec.to_dict()
        assert d["tool"] == "test"
        assert d["elapsed_ms"] == 42.5
        assert d["status"] == "ok"


# ═══════════════════════════════════════════════════════════════════════
# 6. AdaptiveTimeoutTracker
# ═══════════════════════════════════════════════════════════════════════


class TestAdaptiveTimeoutTracker:
    """Tests for adaptive timeout P95 computation."""

    def test_returns_ceiling_before_min_samples(self):
        tracker = AdaptiveTimeoutTracker(min_samples=3)
        tracker.record("web_search", 1.0)
        tracker.record("web_search", 2.0)
        # Only 2 samples, need 3
        assert tracker.get_timeout("web_search", 30.0) == 30.0

    def test_returns_adaptive_after_min_samples(self):
        tracker = AdaptiveTimeoutTracker(min_samples=3, multiplier=2.0, min_floor=1.0)
        for _ in range(5):
            tracker.record("tool", 1.0)
        # P95 of [1,1,1,1,1] = 1.0, adaptive = max(1.0, 1.0*2.0) = 2.0
        # effective = min(30.0, 2.0) = 2.0
        timeout = tracker.get_timeout("tool", 30.0)
        assert timeout == pytest.approx(2.0)

    def test_min_floor_enforced(self):
        tracker = AdaptiveTimeoutTracker(min_samples=3, multiplier=1.0, min_floor=5.0)
        for _ in range(5):
            tracker.record("tool", 0.1)
        # P95 * 1.0 = 0.1, but min_floor is 5.0
        timeout = tracker.get_timeout("tool", 30.0)
        assert timeout == pytest.approx(5.0)

    def test_ceiling_caps_adaptive(self):
        tracker = AdaptiveTimeoutTracker(min_samples=3, multiplier=100.0, min_floor=1.0)
        for _ in range(5):
            tracker.record("tool", 10.0)
        # adaptive = 10.0 * 100 = 1000, ceiling = 30
        timeout = tracker.get_timeout("tool", 30.0)
        assert timeout == 30.0

    def test_window_size_bounds_samples(self):
        tracker = AdaptiveTimeoutTracker(window_size=3, min_samples=3, multiplier=2.0, min_floor=1.0)
        tracker.record("tool", 100.0)
        tracker.record("tool", 100.0)
        tracker.record("tool", 100.0)
        tracker.record("tool", 1.0)
        tracker.record("tool", 1.0)
        tracker.record("tool", 1.0)
        # Window keeps last 3: [1.0, 1.0, 1.0]
        timeout = tracker.get_timeout("tool", 200.0)
        assert timeout < 10.0  # Should be based on recent 1.0s

    def test_unknown_tool_returns_ceiling(self):
        tracker = AdaptiveTimeoutTracker()
        assert tracker.get_timeout("unknown", 45.0) == 45.0


# ═══════════════════════════════════════════════════════════════════════
# 7. TrustPolicyEvaluator
# ═══════════════════════════════════════════════════════════════════════


class TestTrustPolicyEvaluator:
    """Tests for trust policy evaluation, confirmation checks, blocking."""

    def test_yolo_never_requires_confirmation(self):
        ev = TrustPolicyEvaluator(trust_level="yolo")
        assert not ev.requires_confirmation("run_command")
        assert not ev.requires_confirmation("write_file")

    def test_normal_requires_confirmation_for_writes(self):
        ev = TrustPolicyEvaluator(trust_level="normal")
        assert ev.requires_confirmation("write_file")
        assert ev.requires_confirmation("run_command")
        assert not ev.requires_confirmation("read_file")

    def test_paranoid_confirms_everything(self):
        ev = TrustPolicyEvaluator(trust_level="paranoid")
        assert ev.requires_confirmation("read_file")
        assert ev.requires_confirmation("run_command")
        assert ev.requires_confirmation("web_search")

    def test_permission_mode_ask(self):
        ev = TrustPolicyEvaluator(trust_level="yolo", permission_mode="ask")
        # ask mode overrides yolo — always ask
        assert ev.requires_confirmation("read_file")

    def test_permission_mode_deny(self):
        ev = TrustPolicyEvaluator(trust_level="yolo", permission_mode="deny")
        # deny mode blocks high-risk tools
        assert ev.requires_confirmation("run_command")

    def test_is_blocked_by_blocked_tools(self):
        ev = TrustPolicyEvaluator(trust_level="normal")
        result = ev.is_blocked("web_search", blocked_tools=["web_search"])
        assert result is not None
        assert "blocked" in result.lower()

    def test_is_blocked_by_allowed_list(self):
        ev = TrustPolicyEvaluator(trust_level="normal")
        result = ev.is_blocked("web_search", allowed_tools=["read_file"])
        assert result is not None
        assert "not in the allowed" in result

    def test_not_blocked_when_in_allowed_list(self):
        ev = TrustPolicyEvaluator(trust_level="normal")
        result = ev.is_blocked("read_file", allowed_tools=["read_file"])
        assert result is None

    def test_assess_risk_low(self):
        ev = TrustPolicyEvaluator(trust_level="normal")
        ctx = ev.assess_risk("read_file", {"path": "/tmp/safe.txt"})
        assert isinstance(ctx, ApprovalContext)
        assert ctx.risk_level == "low"
        assert ctx.reversible is True

    def test_assess_risk_critical_pattern(self):
        ev = TrustPolicyEvaluator(trust_level="normal")
        ctx = ev.assess_risk("run_command", {"command": "sudo rm -rf /"})
        assert ctx.risk_level == "critical"
        assert ctx.reversible is False

    def test_trust_level_property(self):
        ev = TrustPolicyEvaluator(trust_level="paranoid")
        assert ev.trust_level == "paranoid"

    def test_describe_impact(self):
        ev = TrustPolicyEvaluator(trust_level="normal")
        desc = ev._describe_impact("run_command", {"command": "ls"})
        assert "shell command" in desc.lower()

    def test_describe_impact_fallback(self):
        ev = TrustPolicyEvaluator(trust_level="normal")
        desc = ev._describe_impact("custom_tool", {})
        assert "custom_tool" in desc


class TestApprovalHistory:
    """Tests for SQLite-backed approval persistence."""

    def test_check_and_record(self, tmp_path):
        db_path = str(tmp_path / "approvals.db")
        ah = ApprovalHistory(db_path)
        # Initially no record
        assert ah.check("web_search", {"query": "test"}) is None
        # Record approval
        ah.record("web_search", {"query": "test"}, True)
        assert ah.check("web_search", {"query": "test"}) is True
        # Record denial
        ah.record("run_command", {"command": "rm"}, False)
        assert ah.check("run_command", {"command": "rm"}) is False

    def test_different_args_different_records(self, tmp_path):
        db_path = str(tmp_path / "approvals.db")
        ah = ApprovalHistory(db_path)
        ah.record("web_search", {"query": "a"}, True)
        ah.record("web_search", {"query": "b"}, False)
        assert ah.check("web_search", {"query": "a"}) is True
        assert ah.check("web_search", {"query": "b"}) is False

    def test_hash_args_deterministic(self):
        h1 = ApprovalHistory._hash_args({"a": 1, "b": 2})
        h2 = ApprovalHistory._hash_args({"b": 2, "a": 1})
        assert h1 == h2  # sort_keys=True

    def test_approval_context_to_message(self):
        ctx = ApprovalContext(
            tool_name="run_command",
            arguments={"command": "ls"},
            risk_level="high",
            impact_summary="Execute shell command: ls",
            cost_estimate="free",
            reversible=False,
        )
        msg = ctx.to_message()
        assert "run_command" in msg
        assert "high" in msg.lower()
        assert "not be reversible" in msg.lower()


class TestApprovalContextMessage:
    """Tests for ApprovalContext.to_message formatting."""

    def test_low_risk_message(self):
        ctx = ApprovalContext(
            tool_name="read_file",
            arguments={"path": "/tmp/x"},
            risk_level="low",
            impact_summary="Read file",
            cost_estimate="free",
            reversible=True,
        )
        msg = ctx.to_message()
        assert "read_file" in msg
        assert "yes" in msg.lower()

    def test_critical_risk_message(self):
        ctx = ApprovalContext(
            tool_name="run_command",
            arguments={"command": "sudo rm -rf /"},
            risk_level="critical",
            impact_summary="Execute shell",
            cost_estimate="free",
            reversible=False,
        )
        msg = ctx.to_message()
        assert "not be reversible" in msg.lower()


# ═══════════════════════════════════════════════════════════════════════
# 8. Middleware Stack
# ═══════════════════════════════════════════════════════════════════════


class TestMiddlewareContext:
    """Tests for MiddlewareContext dataclass."""

    def test_defaults(self):
        ctx = MiddlewareContext()
        assert ctx.tool == ""
        assert ctx.args == {}
        assert ctx.status == "pending"
        assert not ctx.skip_execution
        assert ctx.trace_id  # auto-generated

    def test_construction(self):
        ctx = MiddlewareContext(tool="web_search", args={"q": "test"}, origin="user")
        assert ctx.tool == "web_search"
        assert ctx.args["q"] == "test"
        assert ctx.origin == "user"


class TestMiddlewareStack:
    """Tests for MiddlewareStack ordering and execution."""

    @pytest.mark.asyncio
    async def test_empty_stack_runs_without_error(self):
        stack = MiddlewareStack()
        ctx = MiddlewareContext(tool="test")
        await stack.run_before(ctx)
        await stack.run_after(ctx)

    @pytest.mark.asyncio
    async def test_before_hooks_run_in_order(self):
        order = []

        class MW1(Middleware):
            @property
            def order(self):
                return 10
            async def before(self, ctx):
                order.append("mw1")

        class MW2(Middleware):
            @property
            def order(self):
                return 20
            async def before(self, ctx):
                order.append("mw2")

        stack = MiddlewareStack()
        stack.add(MW2())  # add out of order
        stack.add(MW1())
        ctx = MiddlewareContext()
        await stack.run_before(ctx)
        assert order == ["mw1", "mw2"]

    @pytest.mark.asyncio
    async def test_after_hooks_run_in_reverse(self):
        order = []

        class MW1(Middleware):
            @property
            def order(self):
                return 10
            async def after(self, ctx):
                order.append("mw1")

        class MW2(Middleware):
            @property
            def order(self):
                return 20
            async def after(self, ctx):
                order.append("mw2")

        stack = MiddlewareStack()
        stack.add(MW1())
        stack.add(MW2())
        ctx = MiddlewareContext()
        await stack.run_after(ctx)
        assert order == ["mw2", "mw1"]

    @pytest.mark.asyncio
    async def test_skip_execution_halts_before_chain(self):
        order = []

        class SkipMW(Middleware):
            @property
            def order(self):
                return 10
            async def before(self, ctx):
                ctx.skip_execution = True
                ctx.skip_reason = "blocked"
                order.append("skipper")

        class NeverReached(Middleware):
            @property
            def order(self):
                return 20
            async def before(self, ctx):
                order.append("unreachable")

        stack = MiddlewareStack()
        stack.add(SkipMW())
        stack.add(NeverReached())
        ctx = MiddlewareContext()
        await stack.run_before(ctx)
        assert order == ["skipper"]
        assert ctx.skip_execution is True

    @pytest.mark.asyncio
    async def test_middleware_exception_is_swallowed(self):
        class FailMW(Middleware):
            async def before(self, ctx):
                raise RuntimeError("boom")

        stack = MiddlewareStack()
        stack.add(FailMW())
        ctx = MiddlewareContext()
        # Should not raise
        await stack.run_before(ctx)

    def test_add_returns_self_for_chaining(self):
        stack = MiddlewareStack()
        result = stack.add(LoggingMiddleware())
        assert result is stack

    def test_remove_by_name(self):
        stack = MiddlewareStack()
        stack.add(LoggingMiddleware())
        assert len(stack) == 1
        removed = stack.remove("LoggingMiddleware")
        assert removed is True
        assert len(stack) == 0

    def test_remove_nonexistent(self):
        stack = MiddlewareStack()
        removed = stack.remove("NotThere")
        assert removed is False

    def test_len(self):
        stack = MiddlewareStack()
        assert len(stack) == 0
        stack.add(LoggingMiddleware())
        assert len(stack) == 1

    def test_middlewares_property_returns_copy(self):
        stack = MiddlewareStack()
        stack.add(LoggingMiddleware())
        mw_list = stack.middlewares
        mw_list.clear()  # mutating copy
        assert len(stack) == 1  # original untouched


class TestInputSanitizerMiddleware:
    """Tests for InputSanitizerMiddleware."""

    @pytest.mark.asyncio
    async def test_strips_whitespace(self):
        mw = InputSanitizerMiddleware()
        ctx = MiddlewareContext(args={"path": "  /tmp/file.txt  "})
        await mw.before(ctx)
        assert ctx.args["path"] == "/tmp/file.txt"

    @pytest.mark.asyncio
    async def test_removes_null_bytes(self):
        mw = InputSanitizerMiddleware()
        ctx = MiddlewareContext(args={"text": "hello\x00world"})
        await mw.before(ctx)
        assert "\x00" not in ctx.args["text"]

    @pytest.mark.asyncio
    async def test_truncates_oversized(self):
        mw = InputSanitizerMiddleware(max_arg_length=10)
        ctx = MiddlewareContext(args={"text": "a" * 100})
        await mw.before(ctx)
        assert len(ctx.args["text"]) < 100
        assert ctx.args["text"].endswith("...[truncated]")

    @pytest.mark.asyncio
    async def test_order(self):
        mw = InputSanitizerMiddleware()
        assert mw.order == 5


class TestOutputTruncationMiddleware:
    """Tests for OutputTruncationMiddleware."""

    @pytest.mark.asyncio
    async def test_short_output_unchanged(self):
        mw = OutputTruncationMiddleware(max_length=100)
        ctx = MiddlewareContext(result="short")
        await mw.after(ctx)
        assert ctx.result == "short"

    @pytest.mark.asyncio
    async def test_long_output_truncated(self):
        mw = OutputTruncationMiddleware(max_length=100, keep_head=40, keep_tail=20)
        ctx = MiddlewareContext(result="x" * 200)
        await mw.after(ctx)
        assert len(ctx.result) < 200
        assert "truncated" in ctx.result
        assert ctx.metadata["truncated"] is True
        assert ctx.metadata["original_length"] == 200

    @pytest.mark.asyncio
    async def test_order(self):
        mw = OutputTruncationMiddleware()
        assert mw.order == 90


class TestMetricsMiddleware:
    """Tests for MetricsMiddleware counters and snapshot."""

    @pytest.mark.asyncio
    async def test_tracks_calls(self):
        mw = MetricsMiddleware()
        ctx = MiddlewareContext(tool="read_file", status="ok", duration_ms=10.0)
        await mw.after(ctx)
        snap = mw.snapshot()
        assert snap["total_calls"] == 1
        assert "read_file" in snap["tools"]

    @pytest.mark.asyncio
    async def test_tracks_errors(self):
        mw = MetricsMiddleware()
        ctx = MiddlewareContext(tool="web_search", status="error", duration_ms=50.0)
        await mw.after(ctx)
        snap = mw.snapshot()
        assert snap["total_errors"] == 1

    @pytest.mark.asyncio
    async def test_latency_stats(self):
        mw = MetricsMiddleware()
        for dur in [10.0, 20.0, 30.0]:
            ctx = MiddlewareContext(tool="tool_a", status="ok", duration_ms=dur)
            await mw.after(ctx)
        snap = mw.snapshot()
        assert snap["tools"]["tool_a"]["latency_ms"]["avg"] == pytest.approx(20.0, abs=0.5)

    def test_reset(self):
        mw = MetricsMiddleware()
        mw._total_calls = 5
        mw.reset()
        assert mw._total_calls == 0
        assert mw.snapshot()["total_calls"] == 0

    @pytest.mark.asyncio
    async def test_order(self):
        mw = MetricsMiddleware()
        assert mw.order == 20


class TestAuditTrailMiddleware:
    """Tests for AuditTrailMiddleware."""

    @pytest.mark.asyncio
    async def test_records_entries(self):
        audit = AuditTrailMiddleware(max_entries=100)
        ctx = MiddlewareContext(tool="web_search", status="ok")
        await audit.after(ctx)
        assert audit.size == 1

    @pytest.mark.asyncio
    async def test_max_entries_eviction(self):
        audit = AuditTrailMiddleware(max_entries=3)
        for i in range(5):
            ctx = MiddlewareContext(tool=f"tool_{i}", status="ok")
            await audit.after(ctx)
        assert audit.size == 3

    @pytest.mark.asyncio
    async def test_recent(self):
        audit = AuditTrailMiddleware(max_entries=100)
        for i in range(10):
            ctx = MiddlewareContext(tool=f"tool_{i}", status="ok")
            await audit.after(ctx)
        recent = audit.recent(3)
        assert len(recent) == 3

    @pytest.mark.asyncio
    async def test_search_by_tool(self):
        audit = AuditTrailMiddleware(max_entries=100)
        ctx1 = MiddlewareContext(tool="web_search", status="ok")
        ctx2 = MiddlewareContext(tool="read_file", status="ok")
        await audit.after(ctx1)
        await audit.after(ctx2)
        results = audit.search(tool="web_search")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_by_status(self):
        audit = AuditTrailMiddleware(max_entries=100)
        ctx1 = MiddlewareContext(tool="a", status="ok")
        ctx2 = MiddlewareContext(tool="b", status="error")
        await audit.after(ctx1)
        await audit.after(ctx2)
        results = audit.search(status="error")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_order(self):
        audit = AuditTrailMiddleware()
        assert audit.order == 30


class TestPerToolRateLimitMiddleware:
    """Tests for per-tool rate limiting."""

    @pytest.mark.asyncio
    async def test_allows_under_limit(self):
        mw = PerToolRateLimitMiddleware(limits={"web_search": 5})
        ctx = MiddlewareContext(tool="web_search")
        await mw.before(ctx)
        assert not ctx.skip_execution

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        mw = PerToolRateLimitMiddleware(limits={"web_search": 2})
        for _ in range(2):
            ctx = MiddlewareContext(tool="web_search")
            await mw.before(ctx)
        ctx = MiddlewareContext(tool="web_search")
        await mw.before(ctx)
        assert ctx.skip_execution
        assert "rate limited" in ctx.skip_reason.lower()

    @pytest.mark.asyncio
    async def test_no_limit_allows_all(self):
        mw = PerToolRateLimitMiddleware(limits={})
        ctx = MiddlewareContext(tool="read_file")
        await mw.before(ctx)
        assert not ctx.skip_execution

    def test_set_and_remove_limit(self):
        mw = PerToolRateLimitMiddleware()
        mw.set_limit("web_search", 10)
        assert mw._limits["web_search"] == 10
        mw.remove_limit("web_search")
        assert "web_search" not in mw._limits


class TestCreateDefaultStack:
    """Tests for the default middleware stack factory."""

    def test_creates_five_middlewares(self):
        stack = create_default_stack()
        assert len(stack) == 5

    def test_order_is_sorted(self):
        stack = create_default_stack()
        orders = [mw.order for mw in stack.middlewares]
        assert orders == sorted(orders)

    def test_contains_expected_types(self):
        stack = create_default_stack()
        names = [mw.name for mw in stack.middlewares]
        assert "InputSanitizerMiddleware" in names
        assert "LoggingMiddleware" in names
        assert "MetricsMiddleware" in names
        assert "AuditTrailMiddleware" in names
        assert "OutputTruncationMiddleware" in names


# ═══════════════════════════════════════════════════════════════════════
# 9. Ethical Compliance + Dispatcher helpers
# ═══════════════════════════════════════════════════════════════════════


class TestEthicalCompliance:
    """Tests for _check_ethical_compliance keyword guard."""

    def test_yolo_always_allows(self):
        result = _check_ethical_compliance(
            "run_command", {"command": "drop_table users"}, "yolo"
        )
        assert result is None

    def test_paranoid_blocks_forbidden(self):
        result = _check_ethical_compliance(
            "run_command", {"command": "drop_table users"}, "paranoid"
        )
        assert result is not None
        assert "Blocked" in result
        assert "drop_table" in result

    def test_normal_warns_but_allows(self):
        result = _check_ethical_compliance(
            "run_command", {"command": "drop_table users"}, "normal"
        )
        # Normal mode only warns (returns None)
        assert result is None

    def test_no_match_allows(self):
        result = _check_ethical_compliance(
            "run_command", {"command": "ls -la"}, "paranoid"
        )
        assert result is None

    def test_forbidden_keywords_set(self):
        assert "drop_table" in _FORBIDDEN_KEYWORDS
        assert "bypass_auth" in _FORBIDDEN_KEYWORDS
        assert len(_FORBIDDEN_KEYWORDS) == 6


class TestToolAliasMap:
    """Tests for Gemini CLI alias normalization."""

    def test_aliases_exist(self):
        assert "run_in_terminal" in _TOOL_ALIAS_MAP
        assert _TOOL_ALIAS_MAP["run_in_terminal"] == "run_command"

    def test_edit_file_alias(self):
        assert _TOOL_ALIAS_MAP["edit_file"] == "write_file"

    def test_identity_mappings(self):
        # Some map to themselves
        assert _TOOL_ALIAS_MAP["read_file"] == "read_file"
        assert _TOOL_ALIAS_MAP["list_directory"] == "list_directory"


# ═══════════════════════════════════════════════════════════════════════
# 10. ToolContext + ToolError hierarchy
# ═══════════════════════════════════════════════════════════════════════


class TestToolError:
    """Tests for ToolError exception hierarchy."""

    def test_basic_construction(self):
        err = ToolError("something failed")
        assert str(err) == "something failed"
        assert err.kind == ToolErrorKind.EXECUTION
        assert err.recoverable is True
        assert err.suggestion == ""

    def test_full_construction(self):
        err = ToolError(
            "file not found",
            kind=ToolErrorKind.NOT_FOUND,
            tool_name="read_file",
            recoverable=False,
            suggestion="check path",
        )
        assert err.kind == ToolErrorKind.NOT_FOUND
        assert err.tool_name == "read_file"
        assert err.recoverable is False
        assert err.suggestion == "check path"

    def test_format_basic(self):
        err = ToolError("error msg")
        formatted = err.format()
        assert formatted == "[error msg]"

    def test_format_with_suggestion(self):
        err = ToolError("boom", suggestion="try again")
        formatted = err.format()
        assert "Suggestion: try again" in formatted

    def test_format_non_recoverable(self):
        err = ToolError("fatal", recoverable=False)
        formatted = err.format()
        assert "non-recoverable" in formatted

    def test_to_dict(self):
        err = ToolError("bad", kind=ToolErrorKind.BLOCKED, tool_name="run_command")
        d = err.to_dict()
        assert d["kind"] == "blocked"
        assert d["tool"] == "run_command"

    def test_is_exception(self):
        err = ToolError("test")
        assert isinstance(err, Exception)


class TestToolErrorKind:
    """Tests for ToolErrorKind enum."""

    def test_all_kinds_exist(self):
        kinds = {k.value for k in ToolErrorKind}
        expected = {
            "missing_param", "invalid_param", "not_found", "blocked",
            "unavailable", "timeout", "execution", "permission", "limit_exceeded",
        }
        assert kinds == expected


class TestConvenienceErrorConstructors:
    """Tests for missing_param, invalid_param, etc."""

    def test_missing_param(self):
        err = missing_param("path", tool="read_file")
        assert err.kind == ToolErrorKind.MISSING_PARAM
        assert "path" in str(err)

    def test_invalid_param(self):
        err = invalid_param("timeout", "must be positive", tool="run_command")
        assert err.kind == ToolErrorKind.INVALID_PARAM

    def test_subsystem_unavailable(self):
        err = subsystem_unavailable("desktop_operator")
        assert err.kind == ToolErrorKind.UNAVAILABLE
        assert err.suggestion != ""

    def test_resource_not_found(self):
        err = resource_not_found("File", path="/tmp/x.txt")
        assert err.kind == ToolErrorKind.NOT_FOUND
        assert "/tmp/x.txt" in str(err)

    def test_blocked(self):
        err = blocked("sensitive path")
        assert err.kind == ToolErrorKind.BLOCKED
        assert err.recoverable is False


class TestToolContext:
    """Tests for ToolContext dataclass."""

    def test_minimal_construction(self):
        ctx = ToolContext(config=None, memory={})
        assert ctx.desktop_operator is None
        assert ctx.unified_memory is None
        assert ctx.openclaw_enabled is False

    def test_full_construction(self):
        ctx = ToolContext(
            config={"test": True},
            memory={"key": {"value": 42}},
            sandbox="fake_sandbox",
            openclaw_enabled=True,
        )
        assert ctx.config["test"] is True
        assert ctx.sandbox == "fake_sandbox"
        assert ctx.openclaw_enabled is True


class TestWebCache:
    """Tests for shared web cache helpers."""

    def test_put_and_get(self):
        web_cache_put("test_key_1", "value")
        result = web_cache_get("test_key_1")
        assert result == "value"

    def test_get_missing_key(self):
        result = web_cache_get("nonexistent_key_xyz")
        assert result is None


class TestSensitivePaths:
    """Tests for sensitive path pattern lists."""

    def test_read_patterns_nonempty(self):
        assert len(SENSITIVE_READ_PATTERNS) > 10
        assert ".ssh/" in SENSITIVE_READ_PATTERNS
        assert ".env" in SENSITIVE_READ_PATTERNS

    def test_write_paths_nonempty(self):
        assert len(SENSITIVE_WRITE_PATHS) > 5
        assert "/etc/" in SENSITIVE_WRITE_PATHS

    def test_write_files_nonempty(self):
        assert ".bashrc" in SENSITIVE_WRITE_FILES


class TestDelegationDepth:
    """Tests for multi-agent recursion guard constants."""

    def test_max_depth(self):
        assert _MAX_DELEGATION_DEPTH == 3


# ═══════════════════════════════════════════════════════════════════════
# 11. Pipeline
# ═══════════════════════════════════════════════════════════════════════


class TestPipelineStep:
    """Tests for PipelineStep dataclass."""

    def test_from_dict_minimal(self):
        step = PipelineStep.from_dict({"tool": "read_file", "args": {"path": "/tmp"}})
        assert step.tool == "read_file"
        assert step.args["path"] == "/tmp"
        assert step.on_error == "stop"

    def test_from_dict_full(self):
        step = PipelineStep.from_dict({
            "tool": "web_search",
            "args": {"query": "test"},
            "name": "step1",
            "condition": "not_empty",
            "on_error": "continue",
            "timeout": 15.0,
            "approval": "REQUIRED",
        })
        assert step.name == "step1"
        assert step.condition == "not_empty"
        assert step.on_error == "continue"
        assert step.timeout == 15.0
        assert step.approval == "required"

    def test_to_dict(self):
        step = PipelineStep(tool="read_file", args={"path": "/tmp"}, name="s1")
        d = step.to_dict()
        assert d["tool"] == "read_file"
        assert d["name"] == "s1"

    def test_to_dict_omits_defaults(self):
        step = PipelineStep(tool="read_file", args={})
        d = step.to_dict()
        assert "name" not in d
        assert "condition" not in d


class TestPipelineResult:
    """Tests for PipelineResult."""

    def test_to_dict(self):
        r = PipelineResult(
            ok=True,
            steps_completed=2,
            total_steps=2,
            results=[
                {"step": 0, "tool": "a", "status": "ok", "elapsed_ms": 10.0, "output": "out"},
            ],
            final_output="final",
            elapsed_ms=100.0,
        )
        d = r.to_dict()
        assert d["ok"] is True
        assert d["steps_completed"] == 2
        assert len(d["step_results"]) == 1

    def test_to_dict_with_resume_token(self):
        r = PipelineResult(
            ok=True, steps_completed=1, total_steps=3,
            results=[], final_output="",
            elapsed_ms=50.0, status="needs_approval",
            resume_token="jp_abc123",
        )
        d = r.to_dict()
        assert d["resume_token"] == "jp_abc123"


class TestPipelineConditions:
    """Tests for ToolPipeline._check_condition."""

    @pytest.mark.parametrize(
        "condition,output,expected",
        [
            ("contains:hello", "hello world", True),
            ("contains:hello", "goodbye", False),
            ("not_contains:error", "success", True),
            ("not_contains:error", "has error", False),
            ("starts_with:ok", "ok done", True),
            ("starts_with:ok", "not ok", False),
            ("not_empty", "has content", True),
            ("not_empty", "", False),
            ("not_empty", "  ", False),
            ("is_error", "[Error: fail]", True),
            ("is_error", "all good", False),
            ("not_error", "all good", True),
            ("not_error", "[Error: fail]", False),
            ("lines>2", "a\nb\nc\nd", True),
            ("lines>2", "a\nb", False),
            ("lines<3", "a\nb", True),
            ("lines<3", "a\nb\nc\nd", False),
        ],
    )
    def test_condition_evaluation(self, condition, output, expected):
        result = ToolPipeline._check_condition(condition, output)
        assert result is expected

    def test_unknown_condition_defaults_true(self):
        assert ToolPipeline._check_condition("weird_cond", "anything") is True


class TestPipelineVariableSubstitution:
    """Tests for variable substitution in pipeline."""

    def _pipeline(self):
        # ToolPipeline needs a dispatcher, but we only test _substitute_vars
        mock_dispatcher = MagicMock()
        return ToolPipeline(mock_dispatcher)

    def test_prev_substitution(self):
        p = self._pipeline()
        result = p._substitute_string("Input: {{prev}}", "hello", {})
        assert result == "Input: hello"

    def test_step_n_substitution(self):
        p = self._pipeline()
        result = p._substitute_string(
            "Step 0 said: {{step.0}}",
            "",
            {"0": "step zero output"},
        )
        assert result == "Step 0 said: step zero output"

    def test_prev_lines(self):
        p = self._pipeline()
        result = p._substitute_string("{{prev_lines}}", "a\nb\nc", {})
        assert result == "3"

    def test_prev_len(self):
        p = self._pipeline()
        result = p._substitute_string("{{prev_len}}", "hello", {})
        assert result == "5"

    def test_unknown_variable_left_as_is(self):
        p = self._pipeline()
        result = p._substitute_string("{{unknown}}", "", {})
        assert result == "{{unknown}}"

    def test_substitute_vars_nested_dict(self):
        p = self._pipeline()
        args = {"outer": {"inner": "test {{prev}}"}}
        result = p._substitute_vars(args, "replaced", {})
        assert result["outer"]["inner"] == "test replaced"

    def test_substitute_vars_list(self):
        p = self._pipeline()
        args = {"items": ["{{prev}}", "static"]}
        result = p._substitute_vars(args, "dynamic", {})
        assert result["items"][0] == "dynamic"
        assert result["items"][1] == "static"

    def test_substitution_truncation(self):
        p = self._pipeline()
        huge = "x" * 100_000
        result = p._substitute_vars({"text": "{{prev}}"}, huge, {})
        # Should be capped at _MAX_SUBSTITUTION_LEN
        assert len(result["text"]) <= p._MAX_SUBSTITUTION_LEN


class TestPipelineConstants:
    """Tests for pipeline constants."""

    def test_max_steps(self):
        # Raised from 20 → 200 per "remove all limits"
        assert _MAX_PIPELINE_STEPS == 200

    def test_max_timeout(self):
        # Raised from 300 → 3600 (5min → 1h) per "remove all limits"
        assert _MAX_PIPELINE_TIMEOUT == 3600


# ═══════════════════════════════════════════════════════════════════════
# 12. Health Dashboard
# ═══════════════════════════════════════════════════════════════════════


class TestHealthDashboard:
    """Tests for HealthDashboard aggregation."""

    def _make_dashboard(self):
        """Create a HealthDashboard with mocked dispatcher."""
        dispatcher = MagicMock()
        dispatcher.execution_history = ExecutionHistory(max_entries=100)
        dispatcher.circuit_breaker = ToolCircuitBreaker()
        dispatcher.result_cache = ToolResultCache(max_entries=10)
        dispatcher.middleware = MiddlewareStack()
        dispatcher._adaptive = AdaptiveTimeoutTracker()
        dispatcher._tool_timeouts = {}
        dispatcher._timeout = 30
        return HealthDashboard(dispatcher)

    def test_report_structure(self):
        dashboard = self._make_dashboard()
        report = dashboard.report()
        assert "timestamp" in report
        assert "circuit_breakers" in report
        assert "cache" in report
        assert "execution_history" in report
        assert "adaptive_timeouts" in report
        assert "middleware" in report
        assert "overall_health" in report

    def test_overall_health_perfect_score(self):
        dashboard = self._make_dashboard()
        report = dashboard.report()
        health = report["overall_health"]
        assert health["score"] == 100.0
        assert health["status"] == "healthy"

    def test_overall_health_degrades_with_errors(self):
        dashboard = self._make_dashboard()
        dispatcher = dashboard._dispatcher
        for _ in range(5):
            dispatcher.execution_history.record("a", {}, "r", "error", 10.0)
        for _ in range(5):
            dispatcher.execution_history.record("b", {}, "r", "ok", 10.0)
        report = dashboard.report()
        health = report["overall_health"]
        assert health["score"] < 100.0

    def test_summary(self):
        dashboard = self._make_dashboard()
        summary = dashboard.summary()
        assert "total_calls" in summary
        assert "health" in summary


# ═══════════════════════════════════════════════════════════════════════
# 13. Helpers
# ═══════════════════════════════════════════════════════════════════════


class TestHelpers:
    """Tests for helper functions."""

    def test_truncate_short(self):
        assert _truncate("hi", 10) == "hi"

    def test_truncate_long(self):
        result = _truncate("a" * 100, 10)
        assert len(result) == 10
        assert result.endswith("...")

    def test_percentile_empty(self):
        assert _percentile([], 50) == 0.0

    def test_percentile_single(self):
        assert _percentile([5.0], 99) == 5.0

    def test_percentile_sorted(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        p50 = _percentile(data, 50)
        assert 5.0 <= p50 <= 6.0
        p95 = _percentile(data, 95)
        assert 9.0 <= p95 <= 10.0


# ═══════════════════════════════════════════════════════════════════════
# 14. Critical Patterns (trust policy)
# ═══════════════════════════════════════════════════════════════════════


class TestCriticalPatterns:
    """Tests for _CRITICAL_PATTERNS regex list in trust_policy."""

    @pytest.mark.parametrize(
        "input_str",
        [
            "rm -rf /",
            "sudo apt install",
            "chmod 777 /etc",
            "mkfs /dev/sda",
            "> /dev/null",
        ],
    )
    def test_critical_pattern_matches(self, input_str):
        matched = any(p.search(input_str) for p in _CRITICAL_PATTERNS)
        assert matched, f"Pattern should match: {input_str}"

    @pytest.mark.parametrize(
        "input_str",
        [
            "ls -la",
            "cat file.txt",
            "echo hello",
            "python script.py",
        ],
    )
    def test_safe_commands_not_matched(self, input_str):
        matched = any(p.search(input_str) for p in _CRITICAL_PATTERNS)
        assert not matched, f"Pattern should NOT match: {input_str}"


class TestRiskMap:
    """Tests for _RISK_MAP in trust_policy."""

    def test_read_file_is_low(self):
        assert _RISK_MAP.get("read_file") == "low"

    def test_run_command_is_high(self):
        assert _RISK_MAP.get("run_command") == "high"

    def test_write_file_is_medium(self):
        assert _RISK_MAP.get("write_file") == "medium"

    def test_unknown_tool_defaults_none(self):
        assert _RISK_MAP.get("nonexistent") is None
