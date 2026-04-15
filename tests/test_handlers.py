"""Tests for jarvis.tools.handlers — unit tests with mock ToolContext.

Tests file_ops, memory, identity, stats, and pipeline_handler modules,
plus ToolError formatting and convenience constructors.
"""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.tools.handlers._context import (
    ToolContext,
    ToolError,
    ToolErrorKind,
    blocked,
    invalid_param,
    missing_param,
    resource_not_found,
    subsystem_unavailable,
    web_cache_get,
    web_cache_put,
)


# ── Mock ToolContext Factory ─────────────────────────────────────────


def make_ctx(**overrides) -> ToolContext:
    """Create a minimal ToolContext for testing."""
    defaults = dict(
        config=MagicMock(),
        memory={},
        memory_service=None,
        mcts_planner=None,
        voice=None,
        desktop_operator=None,
        sandbox=None,
        docker_sandbox=None,
        sandbox_pool=None,
        skill_marketplace=None,
        openclaw_runtime=None,
        openclaw_enabled=False,
        llm_for_collab=None,
        unified_memory=None,
        trust_policy={},
        http_with_retry=None,
        format_sandbox_result=None,
        resolve_user_id=None,
    )
    defaults.update(overrides)
    return ToolContext(**defaults)


# ── ToolError Tests ──────────────────────────────────────────────────


class TestToolError:
    def test_basic_format(self):
        err = ToolError("something broke", kind=ToolErrorKind.EXECUTION)
        formatted = err.format()
        assert formatted == "[something broke]"
        assert err.kind == ToolErrorKind.EXECUTION
        assert err.recoverable is True

    def test_format_with_suggestion(self):
        err = ToolError(
            "file too large",
            kind=ToolErrorKind.LIMIT_EXCEEDED,
            suggestion="Use head/tail",
        )
        formatted = err.format()
        assert "file too large" in formatted
        assert "Suggestion: Use head/tail" in formatted

    def test_format_non_recoverable(self):
        err = ToolError(
            "blocked",
            kind=ToolErrorKind.BLOCKED,
            recoverable=False,
        )
        formatted = err.format()
        assert "(non-recoverable)" in formatted

    def test_to_dict(self):
        err = ToolError(
            "test error",
            kind=ToolErrorKind.TIMEOUT,
            tool_name="web_search",
            suggestion="try again",
        )
        d = err.to_dict()
        assert d["error"] == "test error"
        assert d["kind"] == "timeout"
        assert d["tool"] == "web_search"
        assert d["suggestion"] == "try again"
        assert d["recoverable"] is True

    def test_is_exception(self):
        err = ToolError("test")
        assert isinstance(err, Exception)
        assert str(err) == "test"


class TestConvenienceConstructors:
    def test_missing_param(self):
        err = missing_param("path", tool="read_file")
        assert isinstance(err, ToolError)
        assert err.kind == ToolErrorKind.MISSING_PARAM
        assert "path" in str(err)
        assert err.tool_name == "read_file"

    def test_invalid_param(self):
        err = invalid_param("mode", "must be 'fast' or 'slow'", tool="test")
        assert err.kind == ToolErrorKind.INVALID_PARAM
        assert "mode" in str(err)
        assert "must be" in str(err)

    def test_subsystem_unavailable(self):
        err = subsystem_unavailable("Desktop operator", tool="desktop_control")
        assert err.kind == ToolErrorKind.UNAVAILABLE
        assert "Desktop operator" in str(err)
        assert err.suggestion  # should have a suggestion

    def test_resource_not_found(self):
        err = resource_not_found("File", "/tmp/nope.txt", tool="read_file")
        assert err.kind == ToolErrorKind.NOT_FOUND
        assert "/tmp/nope.txt" in str(err)

    def test_blocked(self):
        err = blocked("sensitive path", tool="write_file")
        assert err.kind == ToolErrorKind.BLOCKED
        assert err.recoverable is False


# ── File Operations Handler Tests ────────────────────────────────────


class TestReadFileHandler:
    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path):
        from jarvis.tools.handlers.file_ops import handle_read_file

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        ctx = make_ctx()

        result = await handle_read_file({"path": str(test_file)}, ctx)
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_read_missing_file(self):
        from jarvis.tools.handlers.file_ops import handle_read_file

        ctx = make_ctx()
        with pytest.raises(ToolError) as exc_info:
            await handle_read_file({"path": "/tmp/nonexistent_jarvis_test_xyz.txt"}, ctx)
        assert exc_info.value.kind == ToolErrorKind.NOT_FOUND

    @pytest.mark.asyncio
    async def test_read_no_path(self):
        from jarvis.tools.handlers.file_ops import handle_read_file

        ctx = make_ctx()
        with pytest.raises(ToolError) as exc_info:
            await handle_read_file({}, ctx)
        assert exc_info.value.kind == ToolErrorKind.MISSING_PARAM

    @pytest.mark.asyncio
    async def test_read_directory_fails(self, tmp_path):
        from jarvis.tools.handlers.file_ops import handle_read_file

        ctx = make_ctx()
        with pytest.raises(ToolError) as exc_info:
            await handle_read_file({"path": str(tmp_path)}, ctx)
        assert exc_info.value.kind == ToolErrorKind.INVALID_PARAM

    @pytest.mark.asyncio
    async def test_read_sensitive_file_blocked(self, tmp_path):
        from jarvis.tools.handlers.file_ops import handle_read_file

        # Create a file that matches sensitive patterns
        ssh_dir = tmp_path / ".ssh"
        ssh_dir.mkdir()
        fake_key = ssh_dir / "id_rsa"
        fake_key.write_text("fake key data")
        ctx = make_ctx()
        with pytest.raises(ToolError) as exc_info:
            await handle_read_file({"path": str(fake_key)}, ctx)
        assert exc_info.value.kind == ToolErrorKind.BLOCKED


class TestWriteFileHandler:
    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_path):
        from jarvis.tools.handlers.file_ops import handle_write_file

        target = tmp_path / "output.txt"
        ctx = make_ctx()

        result = await handle_write_file(
            {"path": str(target), "content": "test content"}, ctx
        )
        assert "Successfully wrote" in result
        assert target.read_text() == "test content"

    @pytest.mark.asyncio
    async def test_write_creates_parent_dirs(self, tmp_path):
        from jarvis.tools.handlers.file_ops import handle_write_file

        target = tmp_path / "deep" / "nested" / "file.txt"
        ctx = make_ctx()

        await handle_write_file({"path": str(target), "content": "deep"}, ctx)
        assert target.exists()
        assert target.read_text() == "deep"

    @pytest.mark.asyncio
    async def test_write_no_path(self):
        from jarvis.tools.handlers.file_ops import handle_write_file

        ctx = make_ctx()
        with pytest.raises(ToolError) as exc_info:
            await handle_write_file({"content": "test"}, ctx)
        assert exc_info.value.kind == ToolErrorKind.MISSING_PARAM

    @pytest.mark.asyncio
    async def test_write_no_content(self):
        from jarvis.tools.handlers.file_ops import handle_write_file

        ctx = make_ctx()
        with pytest.raises(ToolError) as exc_info:
            await handle_write_file({"path": "/tmp/test.txt"}, ctx)
        assert exc_info.value.kind == ToolErrorKind.MISSING_PARAM

    @pytest.mark.asyncio
    async def test_write_sensitive_path_blocked(self):
        from jarvis.tools.handlers.file_ops import handle_write_file

        ctx = make_ctx()
        with pytest.raises(ToolError) as exc_info:
            await handle_write_file(
                {"path": "/etc/passwd", "content": "bad"}, ctx
            )
        assert exc_info.value.kind == ToolErrorKind.BLOCKED


class TestListDirectoryHandler:
    @pytest.mark.asyncio
    async def test_list_directory(self, tmp_path):
        from jarvis.tools.handlers.file_ops import handle_list_directory

        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "subdir").mkdir()
        ctx = make_ctx()

        result = await handle_list_directory({"path": str(tmp_path)}, ctx)
        assert "a.txt" in result
        assert "b.txt" in result
        assert "subdir" in result

    @pytest.mark.asyncio
    async def test_list_nonexistent(self):
        from jarvis.tools.handlers.file_ops import handle_list_directory

        ctx = make_ctx()
        with pytest.raises(ToolError) as exc_info:
            await handle_list_directory(
                {"path": "/tmp/jarvis_nonexistent_dir_test"}, ctx
            )
        assert exc_info.value.kind == ToolErrorKind.NOT_FOUND

    @pytest.mark.asyncio
    async def test_list_empty_dir(self, tmp_path):
        from jarvis.tools.handlers.file_ops import handle_list_directory

        ctx = make_ctx()
        result = await handle_list_directory({"path": str(tmp_path)}, ctx)
        assert "Empty directory" in result

    @pytest.mark.asyncio
    async def test_list_recursive(self, tmp_path):
        from jarvis.tools.handlers.file_ops import handle_list_directory

        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.txt").write_text("deep")
        ctx = make_ctx()

        result = await handle_list_directory(
            {"path": str(tmp_path), "recursive": True}, ctx
        )
        assert "deep.txt" in result


# ── Memory Handler Tests ─────────────────────────────────────────────


class TestMemoryStoreHandler:
    @pytest.mark.asyncio
    async def test_store_basic(self):
        from jarvis.tools.handlers.memory import handle_memory_store

        ctx = make_ctx()
        result = await handle_memory_store(
            {"key": "test_key", "content": "test data"}, ctx
        )
        assert "Stored memory" in result
        assert "test_key" in ctx.memory

    @pytest.mark.asyncio
    async def test_store_missing_key(self):
        from jarvis.tools.handlers.memory import handle_memory_store

        ctx = make_ctx()
        with pytest.raises(ToolError) as exc_info:
            await handle_memory_store({"content": "data"}, ctx)
        assert exc_info.value.kind == ToolErrorKind.MISSING_PARAM

    @pytest.mark.asyncio
    async def test_store_missing_content(self):
        from jarvis.tools.handlers.memory import handle_memory_store

        ctx = make_ctx()
        with pytest.raises(ToolError) as exc_info:
            await handle_memory_store({"key": "k"}, ctx)
        assert exc_info.value.kind == ToolErrorKind.MISSING_PARAM

    @pytest.mark.asyncio
    async def test_store_key_too_long(self):
        from jarvis.tools.handlers.memory import handle_memory_store

        ctx = make_ctx()
        with pytest.raises(ToolError) as exc_info:
            await handle_memory_store(
                {"key": "x" * 300, "content": "data"}, ctx
            )
        assert exc_info.value.kind == ToolErrorKind.LIMIT_EXCEEDED

    @pytest.mark.asyncio
    async def test_store_with_tags(self):
        from jarvis.tools.handlers.memory import handle_memory_store

        ctx = make_ctx()
        await handle_memory_store(
            {"key": "tagged", "content": "data", "tags": ["a", "b"]}, ctx
        )
        assert ctx.memory["tagged"]["tags"] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_store_with_unified_memory(self):
        from jarvis.tools.handlers.memory import handle_memory_store

        mock_unified = AsyncMock()
        mock_unified.store = AsyncMock(return_value="mem-123")
        ctx = make_ctx(unified_memory=mock_unified)

        result = await handle_memory_store(
            {"key": "k", "content": "data"}, ctx
        )
        assert "mem-123" in result
        mock_unified.store.assert_awaited_once()


class TestMemoryRecallHandler:
    @pytest.mark.asyncio
    async def test_recall_from_session(self):
        from jarvis.tools.handlers.memory import handle_memory_recall

        ctx = make_ctx()
        ctx.memory["architecture"] = {
            "content": "microservices pattern",
            "tags": ["arch"],
        }

        result = await handle_memory_recall({"query": "architecture"}, ctx)
        assert "microservices" in result

    @pytest.mark.asyncio
    async def test_recall_no_match(self):
        from jarvis.tools.handlers.memory import handle_memory_recall

        ctx = make_ctx()
        result = await handle_memory_recall({"query": "nonexistent"}, ctx)
        assert "No memories found" in result

    @pytest.mark.asyncio
    async def test_recall_missing_query(self):
        from jarvis.tools.handlers.memory import handle_memory_recall

        ctx = make_ctx()
        with pytest.raises(ToolError) as exc_info:
            await handle_memory_recall({}, ctx)
        assert exc_info.value.kind == ToolErrorKind.MISSING_PARAM

    @pytest.mark.asyncio
    async def test_recall_by_tag(self):
        from jarvis.tools.handlers.memory import handle_memory_recall

        ctx = make_ctx()
        ctx.memory["item1"] = {"content": "something", "tags": ["deploy"]}
        ctx.memory["item2"] = {"content": "other thing", "tags": ["code"]}

        result = await handle_memory_recall({"query": "deploy"}, ctx)
        assert "something" in result


# ── Stats Handler Tests ──────────────────────────────────────────────


class TestStatsHandler:
    @pytest.mark.asyncio
    async def test_stats_no_dispatcher(self):
        from jarvis.tools.handlers.stats import handle_tool_stats

        ctx = make_ctx()
        result = await handle_tool_stats({"section": "all"}, ctx)
        assert "No stats available" in result

    @pytest.mark.asyncio
    async def test_stats_invalid_section(self):
        from jarvis.tools.handlers.stats import handle_tool_stats

        ctx = make_ctx()
        with pytest.raises(ToolError) as exc_info:
            await handle_tool_stats({"section": "invalid_section"}, ctx)
        assert exc_info.value.kind == ToolErrorKind.INVALID_PARAM

    @pytest.mark.asyncio
    async def test_stats_with_mock_dispatcher(self):
        import json
        from jarvis.tools.handlers.stats import handle_tool_stats
        from jarvis.tools.resilience import (
            ToolCircuitBreaker,
            ToolResultCache,
            ExecutionHistory,
        )

        ctx = make_ctx()
        # Simulate dispatcher reference with real resilience objects
        from jarvis.tools.dispatcher import AdaptiveTimeoutTracker
        mock_dispatcher = MagicMock()
        mock_dispatcher.circuit_breaker = ToolCircuitBreaker()
        mock_dispatcher.result_cache = ToolResultCache()
        mock_dispatcher.execution_history = ExecutionHistory()
        mock_dispatcher._adaptive = AdaptiveTimeoutTracker()
        mock_dispatcher.execution_history.record("test", {}, "ok", "ok", 50.0)
        import weakref
        ctx._dispatcher_ref = weakref.ref(mock_dispatcher)

        result = await handle_tool_stats({"section": "all"}, ctx)
        data = json.loads(result)
        assert "circuit_breaker" in data
        assert "cache" in data
        assert "history" in data
        assert data["history"]["stats"]["total_calls"] == 1


# ── Web Cache Tests ──────────────────────────────────────────────────


class TestWebCache:
    def test_put_and_get(self):
        web_cache_put("test_key_jarvis", "cached_value")
        assert web_cache_get("test_key_jarvis") == "cached_value"

    def test_miss(self):
        assert web_cache_get("nonexistent_cache_key_xyz") is None

    def test_expiry(self):
        # We can't easily test TTL without mocking time, but verify the interface
        web_cache_put("expire_test", "value")
        # Should still be fresh
        assert web_cache_get("expire_test") == "value"


# ── Pipeline Handler Tests ───────────────────────────────────────────


class TestPipelineHandler:
    @pytest.mark.asyncio
    async def test_missing_steps(self):
        from jarvis.tools.handlers.pipeline_handler import handle_tool_pipeline

        ctx = make_ctx()
        with pytest.raises(ToolError) as exc_info:
            await handle_tool_pipeline({}, ctx)
        assert exc_info.value.kind == ToolErrorKind.MISSING_PARAM

    @pytest.mark.asyncio
    async def test_invalid_steps_type(self):
        from jarvis.tools.handlers.pipeline_handler import handle_tool_pipeline

        ctx = make_ctx()
        with pytest.raises(ToolError) as exc_info:
            await handle_tool_pipeline({"steps": "not a list"}, ctx)
        assert exc_info.value.kind == ToolErrorKind.INVALID_PARAM

    @pytest.mark.asyncio
    async def test_invalid_mode(self):
        from jarvis.tools.handlers.pipeline_handler import handle_tool_pipeline

        ctx = make_ctx()
        with pytest.raises(ToolError) as exc_info:
            await handle_tool_pipeline(
                {"steps": [{"tool": "a"}], "mode": "invalid"}, ctx
            )
        assert exc_info.value.kind == ToolErrorKind.INVALID_PARAM

    @pytest.mark.asyncio
    async def test_no_dispatcher_ref(self):
        from jarvis.tools.handlers.pipeline_handler import handle_tool_pipeline

        ctx = make_ctx()
        with pytest.raises(ToolError) as exc_info:
            await handle_tool_pipeline(
                {"steps": [{"tool": "a"}], "mode": "sequential"}, ctx
            )
        assert exc_info.value.kind == ToolErrorKind.UNAVAILABLE


# ── ToolErrorKind Coverage ───────────────────────────────────────────


class TestToolErrorKind:
    def test_all_kinds_are_strings(self):
        for kind in ToolErrorKind:
            assert isinstance(kind.value, str)

    def test_kind_count(self):
        # Ensure we haven't lost any kinds
        assert len(ToolErrorKind) >= 9

    def test_all_kinds_produce_valid_errors(self):
        for kind in ToolErrorKind:
            err = ToolError(f"test {kind.value}", kind=kind)
            formatted = err.format()
            assert formatted.startswith("[")
            assert formatted.endswith("]")
            d = err.to_dict()
            assert d["kind"] == kind.value
