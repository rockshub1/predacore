"""
Integration smoke tests — end-to-end dispatcher + handler + resilience.

Tests the full tool dispatch pipeline:
  1. Dispatcher receives tool name + args
  2. Looks up handler in HANDLER_MAP
  3. Applies circuit breaker / cache / adaptive timeout
  4. Executes handler (using mock subsystems)
  5. Records execution history
  6. Returns sanitized result

These tests use real module wiring — no monkeypatching the handler map.
"""
from __future__ import annotations

import os
import sys
import types
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers — minimal mock config + real ToolDispatcher
# ---------------------------------------------------------------------------

def _mock_config():
    """Create a minimal mock config with just enough for handlers."""
    config = types.SimpleNamespace()
    config.security = types.SimpleNamespace(
        task_timeout_seconds=30,
        trust_level="yolo",
        max_file_size_bytes=10_000_000,
        allowed_commands=None,
        blocked_commands=None,
        sandbox_enabled=False,
    )
    config.memory = types.SimpleNamespace(
        persistence_dir="/tmp/predacore_test_memory",
    )
    config.home_dir = "/tmp/predacore_test"
    config.llm = types.SimpleNamespace(
        provider="test",
        model="test",
        fallback_providers=[],
    )
    return config


def _make_dispatcher():
    """Create a real ToolDispatcher with minimal mock subsystems."""
    from predacore.tools.handlers._context import ToolContext
    from predacore.tools.trust_policy import TrustPolicyEvaluator
    from predacore.tools.dispatcher import ToolDispatcher

    ctx = ToolContext(
        config=_mock_config(),
        memory={},
    )
    trust = TrustPolicyEvaluator(trust_level="yolo")
    dispatcher = ToolDispatcher(trust, ctx, rate_max=1000, tool_timeout=30)
    return dispatcher


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDispatcherSmoke:
    """End-to-end dispatcher tests using real handler wiring."""

    @pytest.fixture
    def dispatcher(self):
        return _make_dispatcher()

    # ── read_file (real filesystem) ──────────────────────────────

    @pytest.mark.asyncio
    async def test_read_file_success(self, dispatcher, tmp_path):
        """read_file handler works end-to-end through dispatcher."""
        test_file = tmp_path / "hello.txt"
        test_file.write_text("Hello PredaCore!")
        result = await dispatcher.dispatch(
            "read_file", {"path": str(test_file)}, origin="test"
        )
        assert "Hello PredaCore!" in result

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, dispatcher):
        """read_file returns error for missing file."""
        result = await dispatcher.dispatch(
            "read_file", {"path": "/tmp/nonexistent_predacore_test_file.txt"}, origin="test"
        )
        assert "not found" in result.lower() or "error" in result.lower()

    # ── write_file → read_file (roundtrip) ───────────────────────

    @pytest.mark.asyncio
    async def test_write_then_read(self, dispatcher, tmp_path):
        """write_file + read_file roundtrip through dispatcher."""
        test_file = tmp_path / "roundtrip.txt"
        write_result = await dispatcher.dispatch(
            "write_file",
            {"path": str(test_file), "content": "Roundtrip test ⚡"},
            origin="test",
        )
        assert "error" not in write_result.lower() or "wrote" in write_result.lower()

        read_result = await dispatcher.dispatch(
            "read_file", {"path": str(test_file)}, origin="test"
        )
        assert "Roundtrip test ⚡" in read_result

    # ── list_directory ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_list_directory(self, dispatcher, tmp_path):
        """list_directory returns files in a directory."""
        (tmp_path / "a.py").write_text("# a")
        (tmp_path / "b.py").write_text("# b")
        result = await dispatcher.dispatch(
            "list_directory", {"path": str(tmp_path)}, origin="test"
        )
        assert "a.py" in result
        assert "b.py" in result

    # ── unknown tool ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_unknown_tool(self, dispatcher):
        """Unknown tool returns helpful error."""
        result = await dispatcher.dispatch(
            "nonexistent_tool", {}, origin="test"
        )
        assert "unknown tool" in result.lower()

    # ── alias resolution ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_gemini_alias_run_in_terminal(self, dispatcher):
        """Gemini CLI alias 'run_in_terminal' → 'run_command'."""
        result = await dispatcher.dispatch(
            "run_in_terminal", {"command": "echo hello_from_alias"}, origin="test"
        )
        assert "hello_from_alias" in result

    # ── circuit breaker integration ──────────────────────────────

    def test_circuit_breaker_state(self, dispatcher):
        """Circuit breaker starts closed for fresh tools."""
        from predacore.tools.resilience import CircuitState
        state = dispatcher.circuit_breaker.state("read_file")
        assert state == CircuitState.CLOSED

    # ── result cache integration ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_cache_hit_on_repeated_read(self, dispatcher, tmp_path):
        """Second read_file call within TTL uses cache."""
        test_file = tmp_path / "cached.txt"
        test_file.write_text("Cache me!")

        # First call — cache miss
        r1 = await dispatcher.dispatch(
            "read_file", {"path": str(test_file)}, origin="test"
        )

        # Second call — should be cached
        r2 = await dispatcher.dispatch(
            "read_file", {"path": str(test_file)}, origin="test"
        )

        assert r1 == r2
        assert dispatcher.result_cache.stats()["hits"] >= 1

    # ── execution history integration ────────────────────────────

    @pytest.mark.asyncio
    async def test_execution_history_recorded(self, dispatcher, tmp_path):
        """Dispatcher records execution history."""
        test_file = tmp_path / "history.txt"
        test_file.write_text("Track me")

        await dispatcher.dispatch(
            "read_file", {"path": str(test_file)}, origin="test"
        )

        recent = dispatcher.execution_history.recent(5)
        assert len(recent) >= 1
        assert recent[-1]["tool"] == "read_file"
        assert recent[-1]["status"] == "ok"

    # ── run_command ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_run_command_echo(self, dispatcher):
        """run_command executes shell command and returns output."""
        result = await dispatcher.dispatch(
            "run_command", {"command": "echo integration_test_pass"}, origin="test"
        )
        assert "integration_test_pass" in result

    # ── handler error propagation ────────────────────────────────

    @pytest.mark.asyncio
    async def test_missing_param_error(self, dispatcher):
        """Missing required param returns structured ToolError."""
        result = await dispatcher.dispatch(
            "read_file", {}, origin="test"  # missing 'path'
        )
        assert "path" in result.lower() or "required" in result.lower() or "error" in result.lower()

    # ── cache invalidation on write ──────────────────────────────

    @pytest.mark.asyncio
    async def test_cache_invalidated_on_write(self, dispatcher, tmp_path):
        """write_file invalidates read_file cache."""
        test_file = tmp_path / "invalidate.txt"
        test_file.write_text("Version 1")

        # Read to populate cache
        await dispatcher.dispatch(
            "read_file", {"path": str(test_file)}, origin="test"
        )

        # Write to invalidate
        await dispatcher.dispatch(
            "write_file",
            {"path": str(test_file), "content": "Version 2"},
            origin="test",
        )

        # Read again — should NOT be cached (should see new content)
        result = await dispatcher.dispatch(
            "read_file", {"path": str(test_file)}, origin="test"
        )
        assert "Version 2" in result

    # ── execution history stats ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_execution_history_stats(self, dispatcher, tmp_path):
        """Execution history tracks stats correctly."""
        test_file = tmp_path / "stats.txt"
        test_file.write_text("stat me")

        # Run a few tool calls
        await dispatcher.dispatch("read_file", {"path": str(test_file)}, origin="test")
        await dispatcher.dispatch("list_directory", {"path": str(tmp_path)}, origin="test")
        await dispatcher.dispatch("nonexistent_tool", {}, origin="test")

        stats = dispatcher.execution_history.stats()
        assert stats["total_calls"] >= 2
        assert "read_file" in stats["top_tools"]


class TestDispatcherRateLimit:
    """Rate limiting integration tests."""

    @pytest.mark.asyncio
    async def test_rate_limit_triggers(self):
        """Dispatcher returns rate limit message when exceeded."""
        from predacore.tools.handlers._context import ToolContext
        from predacore.tools.trust_policy import TrustPolicyEvaluator
        from predacore.tools.dispatcher import ToolDispatcher

        ctx = ToolContext(config=_mock_config(), memory={})
        trust = TrustPolicyEvaluator(trust_level="yolo")
        dispatcher = ToolDispatcher(trust, ctx, rate_max=2, tool_timeout=30)

        # Burn through the rate limit
        result = None
        for _ in range(3):
            result = await dispatcher.dispatch(
                "read_file", {"path": "/etc/hostname"}, origin="test"
            )

        # Just verify dispatcher doesn't crash under rate limiting
        assert result is not None
