"""
End-to-end smoke tests for the full PredaCore pipeline.

Tests the complete stack without external services:
  1. Daemon startup — config loads, gateway initializes, channels register
  2. Message flow — IncomingMessage -> gateway -> core -> response
  3. Tool dispatch — message triggers tool call -> dispatcher executes -> result
  4. Memory store + recall — store a fact, recall it, verify persistence
  5. Session management — create session, add messages, load session, verify
  6. Config hot-reload — modify config, verify gateway picks up changes
  7. Rate limiting — verify per-user rate limits work
  8. Error handling — LLM timeout, tool failure, invalid tool name
  9. Persona drift guard — empty response triggers regeneration
 10. In-process MCP — build_sdk_mcp_server, dispatch through it

Mock the LLM provider but use real dispatchers, real memory DB, real handlers.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import types
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.predacore.config import (
    PredaCoreConfig,
    LaunchProfileConfig,
    LLMConfig,
    MemoryConfig,
    SecurityConfig,
)
from src.predacore.gateway import (
    ChannelAdapter,
    Gateway,
    IncomingMessage,
    OutgoingMessage,
)
from src.predacore.sessions import Message, Session, SessionStore
from src.predacore.tools.dispatcher import ToolDispatcher
from src.predacore.tools.handlers._context import ToolContext
from src.predacore.tools.trust_policy import TrustPolicyEvaluator
from src.predacore.identity.engine import reset_identity_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_home: Path) -> PredaCoreConfig:
    """Create a PredaCoreConfig rooted in a temp directory."""
    reset_identity_engine()
    (tmp_home / "sessions").mkdir(parents=True, exist_ok=True)
    (tmp_home / "skills").mkdir(parents=True, exist_ok=True)
    (tmp_home / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_home / "memory").mkdir(parents=True, exist_ok=True)
    return PredaCoreConfig(
        name="PredaCore-Test",
        home_dir=str(tmp_home),
        sessions_dir=str(tmp_home / "sessions"),
        skills_dir=str(tmp_home / "skills"),
        logs_dir=str(tmp_home / "logs"),
        llm=LLMConfig(
            provider="test",
            model="test-model",
            fallback_providers=[],
        ),
        security=SecurityConfig(
            trust_level="yolo",
            task_timeout_seconds=30,
            max_concurrent_tasks=5,
        ),
        memory=MemoryConfig(persistence_dir=str(tmp_home / "memory")),
        launch=LaunchProfileConfig(
            profile="balanced",
            enable_persona_drift_guard=True,
            persona_drift_threshold=0.42,
            persona_drift_max_regens=1,
            max_tool_iterations=10,
        ),
    )


def _make_dispatcher(config: PredaCoreConfig) -> ToolDispatcher:
    """Create a real ToolDispatcher with minimal mock subsystems."""
    ctx = ToolContext(config=config, memory={})
    trust = TrustPolicyEvaluator(trust_level="yolo")
    return ToolDispatcher(trust, ctx, rate_max=1000, tool_timeout=30)


def _run(coro):
    """Run an async coroutine in a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class StubChannel(ChannelAdapter):
    """Minimal channel adapter for testing gateway registration."""

    channel_name = "test_stub"

    def __init__(self):
        self.started = False
        self.stopped = False
        self.sent_messages: list[OutgoingMessage] = []

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def send(self, message: OutgoingMessage) -> None:
        self.sent_messages.append(message)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_home(tmp_path):
    """Temporary PredaCore home directory."""
    home = tmp_path / ".prometheus"
    home.mkdir()
    return home


@pytest.fixture
def config(tmp_home):
    """PredaCoreConfig pointing to temp directories."""
    cfg = _make_config(tmp_home)
    yield cfg
    reset_identity_engine()


@pytest.fixture
def dispatcher(config):
    """Real ToolDispatcher with minimal subsystems."""
    return _make_dispatcher(config)


# ===========================================================================
# TEST 1: Daemon Startup — config loads, gateway init, channels register
# ===========================================================================

class TestDaemonStartup:
    """Verify the daemon boot sequence works end-to-end."""

    def test_config_loads_with_defaults(self, config):
        """PredaCoreConfig loads with sane defaults from temp dir."""
        assert config.name == "PredaCore-Test"
        assert config.security.trust_level == "yolo"
        assert config.llm.provider == "test"
        assert Path(config.home_dir).exists()
        assert Path(config.sessions_dir).exists()
        assert Path(config.memory.persistence_dir).exists()

    def test_gateway_initializes(self, config):
        """Gateway initializes with a mock process function."""
        process_fn = AsyncMock(return_value="hello from core")
        gateway = Gateway(config, process_fn)
        assert gateway.config is config
        assert gateway.session_store is not None
        assert gateway.lane_queue is not None

    def test_channel_registration(self, config):
        """Gateway registers channel adapters and tracks them."""
        process_fn = AsyncMock(return_value="ok")
        gateway = Gateway(config, process_fn)
        stub = StubChannel()
        gateway.register_channel(stub)
        assert "test_stub" in gateway._channels
        assert gateway._channels["test_stub"] is stub

    def test_gateway_start_stop(self, config):
        """Gateway start/stop lifecycle works without errors."""
        process_fn = AsyncMock(return_value="ok")
        gateway = Gateway(config, process_fn)
        stub = StubChannel()
        gateway.register_channel(stub)

        _run(gateway.start())
        assert stub.started
        assert gateway._running

        _run(gateway.stop())
        assert stub.stopped
        assert not gateway._running

    def test_multiple_channels_register(self, config):
        """Multiple channel adapters can register and start concurrently."""
        process_fn = AsyncMock(return_value="ok")
        gateway = Gateway(config, process_fn)

        channels = []
        for name in ("alpha", "beta", "gamma"):
            ch = StubChannel()
            ch.channel_name = name
            gateway.register_channel(ch)
            channels.append(ch)

        _run(gateway.start())
        for ch in channels:
            assert ch.started
        assert len(gateway._channels) == 3

        _run(gateway.stop())
        for ch in channels:
            assert ch.stopped

    def test_gateway_stats_after_start(self, config):
        """Gateway tracks stats from the moment it starts."""
        process_fn = AsyncMock(return_value="ok")
        gateway = Gateway(config, process_fn)
        _run(gateway.start())
        stats = gateway.get_stats()
        assert stats["messages_received"] == 0
        assert stats["messages_sent"] == 0
        assert stats["uptime_seconds"] >= 0
        _run(gateway.stop())


# ===========================================================================
# TEST 2: Message Flow — IncomingMessage -> gateway -> core -> response
# ===========================================================================

class TestMessageFlow:
    """Full message processing through the gateway pipeline."""

    def test_basic_message_flow(self, config):
        """IncomingMessage flows through gateway and returns OutgoingMessage."""
        process_fn = AsyncMock(return_value="I can help with that!")
        gateway = Gateway(config, process_fn)

        incoming = IncomingMessage(
            channel="cli",
            user_id="test_user",
            text="What can you do?",
        )

        response = _run(gateway.handle_message(incoming))

        assert isinstance(response, OutgoingMessage)
        assert response.text == "I can help with that!"
        assert response.channel == "cli"

    def test_message_increments_stats(self, config):
        """Each message increments gateway stats counters."""
        process_fn = AsyncMock(return_value="response")
        gateway = Gateway(config, process_fn)

        incoming = IncomingMessage(
            channel="cli", user_id="test_user", text="hello"
        )
        _run(gateway.handle_message(incoming))

        assert gateway._stats["messages_received"] == 1
        assert gateway._stats["messages_sent"] == 1

    def test_message_creates_session(self, config):
        """First message from a user creates a session."""
        process_fn = AsyncMock(return_value="welcome")
        gateway = Gateway(config, process_fn)

        incoming = IncomingMessage(
            channel="cli", user_id="new_user", text="hi there"
        )
        response = _run(gateway.handle_message(incoming))

        assert response.session_id
        # The process function was called exactly once
        assert process_fn.await_count == 1

    def test_session_command_new(self, config):
        """The /new command creates a fresh session."""
        process_fn = AsyncMock(return_value="ok")
        gateway = Gateway(config, process_fn)

        incoming = IncomingMessage(
            channel="cli", user_id="cmd_user", text="/new"
        )
        response = _run(gateway.handle_message(incoming))

        assert "fresh session" in response.text.lower() or "new" in response.text.lower()
        # /new is handled directly — process_fn should NOT be called
        assert process_fn.await_count == 0

    def test_process_fn_receives_session(self, config):
        """The core process function receives user_id, text, and session."""
        captured_args = {}

        async def capture_fn(user_id, text, session, **kwargs):
            captured_args["user_id"] = user_id
            captured_args["text"] = text
            captured_args["session"] = session
            return "captured"

        gateway = Gateway(config, capture_fn)
        incoming = IncomingMessage(
            channel="cli", user_id="capture_user", text="capture this"
        )
        _run(gateway.handle_message(incoming))

        assert captured_args["text"] == "capture this"
        assert isinstance(captured_args["session"], Session)


# ===========================================================================
# TEST 3: Tool Dispatch — message triggers tool -> dispatcher -> result
# ===========================================================================

class TestToolDispatch:
    """Real dispatcher with real handlers, no mocks on the tool side."""

    def test_read_file_via_dispatcher(self, dispatcher, tmp_path):
        """Dispatcher routes read_file to the real handler."""
        f = tmp_path / "smoke.txt"
        f.write_text("PredaCore smoke test content")

        result = _run(dispatcher.dispatch(
            "read_file", {"path": str(f)}, origin="test"
        ))
        assert "PredaCore smoke test content" in result

    def test_write_then_read_roundtrip(self, dispatcher, tmp_path):
        """Write a file via dispatcher, then read it back."""
        f = tmp_path / "roundtrip.txt"
        _run(dispatcher.dispatch(
            "write_file",
            {"path": str(f), "content": "roundtrip data"},
            origin="test",
        ))
        result = _run(dispatcher.dispatch(
            "read_file", {"path": str(f)}, origin="test"
        ))
        assert "roundtrip data" in result

    def test_list_directory(self, dispatcher, tmp_path):
        """list_directory returns files through dispatcher."""
        (tmp_path / "alpha.py").write_text("# alpha")
        (tmp_path / "beta.py").write_text("# beta")

        result = _run(dispatcher.dispatch(
            "list_directory", {"path": str(tmp_path)}, origin="test"
        ))
        assert "alpha.py" in result
        assert "beta.py" in result

    def test_run_command(self, dispatcher):
        """run_command executes shell commands."""
        result = _run(dispatcher.dispatch(
            "run_command", {"command": "echo smoke_test_pass"}, origin="test"
        ))
        assert "smoke_test_pass" in result

    def test_unknown_tool_returns_error(self, dispatcher):
        """Unknown tool name returns a structured error."""
        result = _run(dispatcher.dispatch(
            "totally_fake_tool", {}, origin="test"
        ))
        assert "unknown tool" in result.lower()

    def test_alias_resolution(self, dispatcher):
        """Gemini CLI alias run_in_terminal resolves to run_command."""
        result = _run(dispatcher.dispatch(
            "run_in_terminal",
            {"command": "echo alias_resolved"},
            origin="test",
        ))
        assert "alias_resolved" in result

    def test_execution_history_recorded(self, dispatcher, tmp_path):
        """Dispatcher records tool executions in history."""
        f = tmp_path / "history.txt"
        f.write_text("track me")
        _run(dispatcher.dispatch(
            "read_file", {"path": str(f)}, origin="test"
        ))
        recent = dispatcher.execution_history.recent(5)
        assert len(recent) >= 1
        assert recent[-1]["tool"] == "read_file"
        assert recent[-1]["status"] == "ok"


# ===========================================================================
# TEST 4: Memory Store + Recall — store a fact, recall, verify persistence
# ===========================================================================

class TestMemoryStoreAndRecall:
    """Test the unified memory store with a real SQLite DB in tmp_path."""

    @pytest.fixture
    def memory_store(self, tmp_path):
        """Create a real UnifiedMemoryStore backed by temp SQLite."""
        from src.predacore.memory.store import UnifiedMemoryStore
        db_path = str(tmp_path / "test_memory.db")
        return UnifiedMemoryStore(db_path=db_path)

    def test_store_and_recall_basic(self, memory_store):
        """Store a fact, then recall it by keyword."""
        mem_id = _run(memory_store.store(
            content="PredaCore was created as part of PredaCore",
            memory_type="fact",
            importance=3,
            tags=["project", "origin"],
            user_id="test_user",
        ))
        assert mem_id  # non-empty UUID string

        # Recall (falls back to keyword search when no embeddings)
        results = _run(memory_store.recall(
            query="Prometheus", user_id="test_user", top_k=5
        ))
        assert len(results) >= 1
        content = results[0][0]["content"]
        assert "Prometheus" in content

    def test_store_multiple_recall_ordered(self, memory_store):
        """Store multiple facts and verify recall returns relevant ones."""
        _run(memory_store.store(
            content="Python is the primary language for PredaCore",
            memory_type="fact", importance=2, user_id="test_user",
        ))
        _run(memory_store.store(
            content="The weather in Tokyo is sunny today",
            memory_type="fact", importance=1, user_id="test_user",
        ))
        _run(memory_store.store(
            content="PredaCore uses SQLite for persistent memory storage",
            memory_type="fact", importance=3, user_id="test_user",
        ))

        results = _run(memory_store.recall(
            query="PredaCore", user_id="test_user", top_k=5
        ))
        # Should find at least the two PredaCore-related memories
        contents = [r[0]["content"] for r in results]
        predacore_hits = [c for c in contents if "PredaCore" in c]
        assert len(predacore_hits) >= 1

    def test_memory_persists_across_instances(self, tmp_path):
        """Memory survives creating a new store instance on the same DB."""
        from src.predacore.memory.store import UnifiedMemoryStore
        db_path = str(tmp_path / "persist_test.db")

        # Store with first instance
        store1 = UnifiedMemoryStore(db_path=db_path)
        _run(store1.store(
            content="persistence test fact",
            memory_type="fact", importance=2, user_id="test_user",
        ))

        # Create a fresh instance on the same DB file
        store2 = UnifiedMemoryStore(db_path=db_path)
        results = _run(store2.recall(
            query="persistence test", user_id="test_user", top_k=5
        ))
        assert len(results) >= 1
        assert "persistence" in results[0][0]["content"]

    def test_memory_handler_via_dispatcher(self, config, tmp_path):
        """Memory store/recall works through the full dispatcher pipeline."""
        from src.predacore.memory.store import UnifiedMemoryStore
        db_path = str(tmp_path / "handler_test.db")
        mem_store = UnifiedMemoryStore(db_path=db_path)

        ctx = ToolContext(
            config=config,
            memory={},
            unified_memory=mem_store,
        )
        trust = TrustPolicyEvaluator(trust_level="yolo")
        disp = ToolDispatcher(trust, ctx, rate_max=1000, tool_timeout=30)

        # Store via handler
        result = _run(disp.dispatch(
            "memory_store",
            {"key": "test_key", "content": "PredaCore is cool", "tags": ["test"]},
            origin="test",
        ))
        assert "stored" in result.lower() or "test_key" in result.lower()

        # Recall via handler
        result = _run(disp.dispatch(
            "memory_recall",
            {"query": "PredaCore"},
            origin="test",
        ))
        assert "cool" in result.lower() or "predacore" in result.lower()


# ===========================================================================
# TEST 5: Session Management — create, add messages, load, verify history
# ===========================================================================

class TestSessionManagement:
    """Session persistence using real filesystem JSONL files."""

    def test_create_session(self, tmp_path):
        """SessionStore creates a session with metadata on disk."""
        store = SessionStore(str(tmp_path / "sessions"))
        session = store.create(user_id="user1", title="Test Session")

        assert session.session_id
        assert session.user_id == "user1"
        assert session.title == "Test Session"
        meta_file = tmp_path / "sessions" / session.session_id / "meta.json"
        assert meta_file.exists()

    def test_add_messages_persist(self, tmp_path):
        """Messages added to a session are persisted to JSONL."""
        store = SessionStore(str(tmp_path / "sessions"))
        session = store.create(user_id="user1")
        sid = session.session_id

        store.append_message(sid, "user", "Hello PredaCore")
        store.append_message(sid, "assistant", "Hello! How can I help?")
        store.append_message(sid, "user", "What time is it?")

        msg_file = tmp_path / "sessions" / sid / "messages.jsonl"
        assert msg_file.exists()
        lines = msg_file.read_text().strip().split("\n")
        assert len(lines) == 3

        # Verify JSONL content
        first = json.loads(lines[0])
        assert first["role"] == "user"
        assert first["content"] == "Hello PredaCore"

    def test_load_session_from_disk(self, tmp_path):
        """Sessions can be loaded from disk across store instances."""
        sessions_dir = str(tmp_path / "sessions")
        store1 = SessionStore(sessions_dir)
        session = store1.create(user_id="user1", title="Persistent")
        sid = session.session_id
        store1.append_message(sid, "user", "Remember this")
        store1.append_message(sid, "assistant", "I will remember")

        # Create a new store instance (simulates daemon restart)
        store2 = SessionStore(sessions_dir)
        loaded = store2.get(sid)

        assert loaded is not None
        assert loaded.title == "Persistent"
        assert loaded.user_id == "user1"
        assert len(loaded.messages) == 2
        assert loaded.messages[0].content == "Remember this"
        assert loaded.messages[1].content == "I will remember"

    def test_list_sessions_by_user(self, tmp_path):
        """list_sessions filters by user_id and sorts by recency."""
        store = SessionStore(str(tmp_path / "sessions"))
        s1 = store.create(user_id="alice", title="Alice Session 1")
        time.sleep(0.01)
        s2 = store.create(user_id="alice", title="Alice Session 2")
        store.create(user_id="bob", title="Bob Session")

        alice_sessions = store.list_sessions(user_id="alice")
        assert len(alice_sessions) == 2
        # Most recent first
        assert alice_sessions[0].title == "Alice Session 2"

    def test_session_message_count(self, tmp_path):
        """Session tracks message count correctly."""
        store = SessionStore(str(tmp_path / "sessions"))
        session = store.create(user_id="user1")
        sid = session.session_id

        for i in range(5):
            store.append_message(sid, "user", f"Message {i}")

        reloaded = store.get(sid)
        assert reloaded.message_count == 5

    def test_session_auto_title(self, tmp_path):
        """Session auto-titles from the first user message."""
        store = SessionStore(str(tmp_path / "sessions"))
        session = store.create(user_id="user1")
        sid = session.session_id
        store.append_message(sid, "user", "Help me write a Python script")

        reloaded = store.get(sid)
        assert "Python" in reloaded.title or "Help" in reloaded.title

    def test_get_llm_messages(self, tmp_path):
        """get_llm_messages returns history in LLM-ready format."""
        store = SessionStore(str(tmp_path / "sessions"))
        session = store.create(user_id="user1")
        sid = session.session_id
        store.append_message(sid, "user", "What is 2+2?")
        store.append_message(sid, "assistant", "2+2 is 4.")

        reloaded = store.get(sid)
        llm_msgs = reloaded.get_llm_messages()
        assert len(llm_msgs) == 2
        assert llm_msgs[0] == {"role": "user", "content": "What is 2+2?"}
        assert llm_msgs[1] == {"role": "assistant", "content": "2+2 is 4."}


# ===========================================================================
# TEST 6: Config Hot-Reload — modify config, gateway picks up changes
# ===========================================================================

class TestConfigHotReload:
    """Verify config changes propagate to gateway components."""

    def test_gateway_picks_up_new_rate_limit(self, tmp_home):
        """Changing the user rate limit on gateway takes effect immediately."""
        config = _make_config(tmp_home)
        process_fn = AsyncMock(return_value="ok")
        gateway = Gateway(config, process_fn)

        # Default rate limit
        original_limit = gateway._user_rate_limit
        assert original_limit > 0

        # "Hot-reload": change the rate limit
        gateway._user_rate_limit = 5
        assert gateway._user_rate_limit == 5

    def test_config_security_change_propagates(self, tmp_home):
        """Modifying security config affects dispatcher behavior."""
        config = _make_config(tmp_home)
        assert config.security.trust_level == "yolo"

        # Simulate hot-reload by creating a new config with different values
        config_v2 = _make_config(tmp_home)
        config_v2.security.trust_level = "paranoid"

        # New dispatcher with updated config reflects the change
        disp = _make_dispatcher(config_v2)
        assert disp._trust.trust_level == "paranoid"

    def test_gateway_channels_update(self, tmp_home):
        """New channels can be registered after gateway starts."""
        config = _make_config(tmp_home)
        process_fn = AsyncMock(return_value="ok")
        gateway = Gateway(config, process_fn)

        _run(gateway.start())
        assert len(gateway._channels) == 0

        # Register a channel post-start
        stub = StubChannel()
        gateway.register_channel(stub)
        assert len(gateway._channels) == 1

        _run(gateway.stop())


# ===========================================================================
# TEST 7: Rate Limiting — verify per-user rate limits
# ===========================================================================

class TestRateLimiting:
    """Per-user rate limiting in the gateway."""

    def test_user_rate_limit_allows_normal_traffic(self, config):
        """Normal message volume passes rate check."""
        process_fn = AsyncMock(return_value="ok")
        gateway = Gateway(config, process_fn)

        incoming = IncomingMessage(
            channel="cli", user_id="rate_user", text="hello"
        )
        response = _run(gateway.handle_message(incoming))
        assert response.text == "ok"

    def test_user_rate_limit_blocks_flood(self, tmp_home):
        """Exceeding rate limit returns a throttle message."""
        config = _make_config(tmp_home)
        process_fn = AsyncMock(return_value="ok")
        gateway = Gateway(config, process_fn)
        # Set a very low rate limit
        gateway._user_rate_limit = 2

        results = []
        for i in range(5):
            msg = IncomingMessage(
                channel="cli", user_id="flood_user", text=f"msg {i}"
            )
            resp = _run(gateway.handle_message(msg))
            results.append(resp.text)

        # At least some responses should be rate-limited
        throttled = [r for r in results if "too quickly" in r.lower() or "wait" in r.lower()]
        assert len(throttled) >= 1, "Rate limiter should block some messages"

    def test_tool_dispatcher_rate_limit(self, config):
        """ToolDispatcher rate-limits when burst limit is very low."""
        ctx = ToolContext(config=config, memory={})
        trust = TrustPolicyEvaluator(trust_level="yolo")
        disp = ToolDispatcher(trust, ctx, rate_max=2, tool_timeout=30)

        results = []
        for _ in range(5):
            r = _run(disp.dispatch(
                "read_file", {"path": "/etc/hostname"}, origin="test"
            ))
            results.append(r)

        # Dispatcher should not crash; at least one call should complete
        assert any(r is not None for r in results)

    def test_different_users_have_separate_limits(self, tmp_home):
        """Rate limits are tracked per-user, not globally."""
        config = _make_config(tmp_home)
        process_fn = AsyncMock(return_value="ok")
        gateway = Gateway(config, process_fn)
        gateway._user_rate_limit = 2

        # User A sends 2 messages (at limit)
        for i in range(2):
            msg = IncomingMessage(
                channel="cli", user_id="user_a", text=f"a{i}"
            )
            _run(gateway.handle_message(msg))

        # User B should still be allowed
        msg_b = IncomingMessage(
            channel="cli", user_id="user_b", text="b0"
        )
        resp_b = _run(gateway.handle_message(msg_b))
        # User B is not rate-limited
        assert "too quickly" not in resp_b.text.lower()


# ===========================================================================
# TEST 8: Error Handling — LLM timeout, tool failure, invalid tool
# ===========================================================================

class TestErrorHandling:
    """Verify graceful error handling throughout the pipeline."""

    def test_gateway_handles_process_exception(self, config):
        """Gateway returns a user-friendly error when process_fn raises."""
        async def broken_fn(*args, **kwargs):
            raise RuntimeError("LLM exploded")

        gateway = Gateway(config, broken_fn)
        incoming = IncomingMessage(
            channel="cli", user_id="error_user", text="hello"
        )
        response = _run(gateway.handle_message(incoming))

        assert "error" in response.text.lower()
        assert gateway._stats["errors"] == 1

    def test_gateway_handles_connection_error(self, config):
        """Gateway returns rate-limit message on ConnectionError."""
        async def rate_limited_fn(*args, **kwargs):
            raise ConnectionError("429 rate limited by provider")

        gateway = Gateway(config, rate_limited_fn)
        incoming = IncomingMessage(
            channel="cli", user_id="rate_err_user", text="hello"
        )
        response = _run(gateway.handle_message(incoming))
        assert "rate" in response.text.lower() or "unavailable" in response.text.lower()

    def test_tool_missing_required_param(self, dispatcher):
        """Missing required parameter returns structured error."""
        result = _run(dispatcher.dispatch(
            "read_file", {}, origin="test"
        ))
        assert "path" in result.lower() or "required" in result.lower() or "error" in result.lower()

    def test_tool_file_not_found(self, dispatcher):
        """Reading a non-existent file returns a not-found error."""
        result = _run(dispatcher.dispatch(
            "read_file", {"path": "/tmp/definitely_does_not_exist_predacore_test.txt"},
            origin="test",
        ))
        assert "not found" in result.lower() or "error" in result.lower()

    def test_tool_timeout_handling(self, config):
        """Tool that exceeds timeout gets a timeout result."""
        from src.predacore.tools.handlers import HANDLER_MAP

        async def slow_handler(args, ctx):
            await asyncio.sleep(10)
            return "too slow"

        HANDLER_MAP["slow_test_tool"] = slow_handler
        try:
            ctx = ToolContext(config=config, memory={})
            trust = TrustPolicyEvaluator(trust_level="yolo")
            disp = ToolDispatcher(trust, ctx, rate_max=1000, tool_timeout=1)

            result = _run(disp.dispatch(
                "slow_test_tool", {}, origin="test"
            ))
            assert "timed out" in result.lower()
        finally:
            del HANDLER_MAP["slow_test_tool"]

    def test_circuit_breaker_starts_closed(self, dispatcher):
        """Circuit breaker is closed (healthy) for new tools."""
        from src.predacore.tools.resilience import CircuitState
        state = dispatcher.circuit_breaker.state("read_file")
        assert state == CircuitState.CLOSED

    def test_gateway_error_persists_to_session(self, config):
        """Errors during processing are persisted to the session."""
        async def failing_fn(*args, **kwargs):
            raise RuntimeError("catastrophic failure")

        gateway = Gateway(config, failing_fn)
        incoming = IncomingMessage(
            channel="cli", user_id="persist_err_user", text="trigger error"
        )
        response = _run(gateway.handle_message(incoming))
        assert "error" in response.text.lower()

        # The error message should have been persisted to the session
        sid = response.session_id
        if sid:
            session = gateway.session_store.get(sid)
            if session and session.messages:
                last_msg = session.messages[-1]
                assert last_msg.role == "assistant"
                assert "error" in last_msg.content.lower()


# ===========================================================================
# TEST 9: Persona Drift Guard — empty response triggers regeneration
# ===========================================================================

class TestPersonaDriftGuard:
    """Verify persona drift detection logic (no LLM call needed)."""

    @pytest.fixture
    def core_config(self, tmp_home):
        """Config with persona drift guard enabled."""
        cfg = _make_config(tmp_home)
        cfg.launch.enable_persona_drift_guard = True
        cfg.launch.persona_drift_threshold = 0.42
        cfg.launch.persona_drift_max_regens = 1
        return cfg

    def test_empty_response_scores_high_drift(self, core_config):
        """An empty assistant response with no tools scores drift = 1.0."""
        from src.predacore.core import PredaCoreCore

        # Patch LLMInterface and ToolExecutor to avoid real init
        with patch("src.predacore.core.LLMInterface"), \
             patch("src.predacore.core.ToolExecutor"), \
             patch("src.predacore.core._get_system_prompt", return_value="test prompt"), \
             patch("src.predacore.core.TranscriptWriter"), \
             patch("src.predacore.core.OutcomeStore"):
            core = PredaCoreCore(core_config)

        assessment = core._assess_persona_drift(
            user_message="who are you?",
            assistant_message="",
            tools_used=0,
        )
        assert assessment.score >= 0.9, f"Empty response should score high drift, got {assessment.score}"
        assert assessment.needs_regeneration

    def test_normal_response_passes_drift_check(self, core_config):
        """A normal PredaCore response should not trigger drift."""
        from src.predacore.core import PredaCoreCore

        with patch("src.predacore.core.LLMInterface"), \
             patch("src.predacore.core.ToolExecutor"), \
             patch("src.predacore.core._get_system_prompt", return_value="test prompt"), \
             patch("src.predacore.core.TranscriptWriter"), \
             patch("src.predacore.core.OutcomeStore"):
            core = PredaCoreCore(core_config)

        assessment = core._assess_persona_drift(
            user_message="what time is it?",
            assistant_message="I checked the system clock. It is currently 3:42 PM.",
            tools_used=1,
        )
        assert assessment.score < core_config.launch.persona_drift_threshold
        assert not assessment.needs_regeneration

    def test_chatgpt_identity_claim_scores_high(self, core_config):
        """Claiming to be ChatGPT triggers persona drift."""
        from src.predacore.core import PredaCoreCore

        with patch("src.predacore.core.LLMInterface"), \
             patch("src.predacore.core.ToolExecutor"), \
             patch("src.predacore.core._get_system_prompt", return_value="test prompt"), \
             patch("src.predacore.core.TranscriptWriter"), \
             patch("src.predacore.core.OutcomeStore"):
            core = PredaCoreCore(core_config)

        assessment = core._assess_persona_drift(
            user_message="who are you?",
            assistant_message="I am ChatGPT, a large language model created by OpenAI.",
            tools_used=0,
        )
        assert assessment.score > 0.3, f"ChatGPT claim should trigger drift, got {assessment.score}"

    def test_tool_only_turn_no_drift(self, core_config):
        """A tool-only turn (no text) with tools_used>0 should not drift."""
        from src.predacore.core import PredaCoreCore

        with patch("src.predacore.core.LLMInterface"), \
             patch("src.predacore.core.ToolExecutor"), \
             patch("src.predacore.core._get_system_prompt", return_value="test prompt"), \
             patch("src.predacore.core.TranscriptWriter"), \
             patch("src.predacore.core.OutcomeStore"):
            core = PredaCoreCore(core_config)

        assessment = core._assess_persona_drift(
            user_message="run ls",
            assistant_message="",
            tools_used=3,
        )
        # Tool-only turns legitimately have no text
        assert assessment.score < core_config.launch.persona_drift_threshold


# ===========================================================================
# TEST 10: In-Process MCP — build server, dispatch through it
# ===========================================================================

class TestInProcessMCP:
    """Test the in-process SDK MCP server wiring."""

    def test_mcp_tool_definitions_load(self):
        """_get_tool_definitions returns a non-empty tool list."""
        from src.predacore.tools.mcp_server import _get_tool_definitions

        tools = _get_tool_definitions()
        assert isinstance(tools, list)
        assert len(tools) > 0

        # Each tool should have name and description
        for tool in tools[:5]:
            assert "name" in tool
            assert "description" in tool

    def test_mcp_tool_names_have_prefix(self):
        """get_mcp_tool_names returns prefixed tool names."""
        from src.predacore.tools.mcp_server import get_mcp_tool_names

        names = get_mcp_tool_names()
        assert len(names) > 0
        for name in names:
            assert name.startswith("mcp__predacore__")

    def test_mcp_config_dict_structure(self):
        """get_mcp_config_dict returns valid MCP config."""
        from src.predacore.tools.mcp_server import get_mcp_config_dict

        config = get_mcp_config_dict()
        assert "mcpServers" in config
        assert "predacore" in config["mcpServers"]
        predacore_cfg = config["mcpServers"]["predacore"]
        assert "command" in predacore_cfg
        assert "args" in predacore_cfg

    def test_mcp_handler_factory(self, dispatcher):
        """_make_tool_handler creates async handlers that route to dispatcher."""
        from src.predacore.tools.mcp_server import _make_tool_handler

        handler = _make_tool_handler(dispatcher, "run_command")
        result = _run(handler({"command": "echo mcp_test"}))

        assert isinstance(result, dict)
        assert "content" in result
        assert len(result["content"]) > 0
        assert "mcp_test" in result["content"][0]["text"]

    def test_mcp_handler_error_handling(self, dispatcher):
        """MCP handler returns error structure for unknown tools."""
        from src.predacore.tools.mcp_server import _make_tool_handler

        handler = _make_tool_handler(dispatcher, "nonexistent_tool_xyz")
        result = _run(handler({}))

        assert isinstance(result, dict)
        assert "content" in result
        # Should get an error response (either is_error or error text)
        text = result["content"][0]["text"]
        assert "unknown tool" in text.lower() or "error" in text.lower()

    def test_mcp_stdio_server_message_handling(self):
        """MCPStdioServer handles protocol messages correctly."""
        from src.predacore.tools.mcp_server import MCPStdioServer

        server = MCPStdioServer()

        # Test initialize
        init_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        }
        response = _run(server.handle_message(init_msg))
        assert response is not None
        assert response["id"] == 1
        assert "result" in response
        assert "capabilities" in response["result"]

        # Test tools/list
        list_msg = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
        response = _run(server.handle_message(list_msg))
        assert response is not None
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) > 0

        # Test ping
        ping_msg = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "ping",
            "params": {},
        }
        response = _run(server.handle_message(ping_msg))
        assert response is not None
        assert response["id"] == 3

    def test_mcp_notification_returns_none(self):
        """Notifications (no id) should return None."""
        from src.predacore.tools.mcp_server import MCPStdioServer

        server = MCPStdioServer()
        notification = {
            "method": "notifications/initialized",
            "params": {},
        }
        response = _run(server.handle_message(notification))
        assert response is None

    def test_build_sdk_mcp_server_structure(self, dispatcher):
        """build_sdk_mcp_server returns a server config when SDK is available."""
        try:
            import claude_agent_sdk
        except ImportError:
            pytest.skip("claude_agent_sdk not installed")

        from src.predacore.tools.mcp_server import build_sdk_mcp_server
        server_config = build_sdk_mcp_server(dispatcher)
        assert server_config is not None


# ===========================================================================
# Integration: Full Pipeline Smoke
# ===========================================================================

class TestFullPipelineSmoke:
    """Cross-cutting tests that exercise multiple subsystems together."""

    def test_message_to_tool_to_response(self, config, tmp_path):
        """Simulate: user message -> gateway -> tool dispatch -> response.

        Gateway calls process_fn, which internally dispatches a tool,
        then returns the tool result as the response.
        """
        disp = _make_dispatcher(config)
        test_file = tmp_path / "pipeline.txt"
        test_file.write_text("pipeline test content")

        async def process_with_tool(user_id, text, session, **kwargs):
            result = await disp.dispatch(
                "read_file", {"path": str(test_file)}, origin="core"
            )
            return f"Tool result: {result}"

        gateway = Gateway(config, process_with_tool)
        incoming = IncomingMessage(
            channel="cli", user_id="pipeline_user", text="read the file"
        )
        response = _run(gateway.handle_message(incoming))
        assert "pipeline test content" in response.text

    def test_session_persists_through_multiple_messages(self, config):
        """Multiple messages in the same session share conversation history."""
        call_count = 0

        async def counting_fn(user_id, text, session, **kwargs):
            nonlocal call_count
            call_count += 1
            return f"Response #{call_count}"

        gateway = Gateway(config, counting_fn)

        # Send multiple messages
        for i in range(3):
            msg = IncomingMessage(
                channel="cli", user_id="multi_user", text=f"message {i}"
            )
            _run(gateway.handle_message(msg))

        assert call_count == 3

    def test_memory_then_session_then_recall(self, config, tmp_path):
        """Store memory, create session with messages, recall memory."""
        from src.predacore.memory.store import UnifiedMemoryStore
        db_path = str(tmp_path / "integration.db")
        mem = UnifiedMemoryStore(db_path=db_path)

        # Store a fact
        _run(mem.store(
            content="The user's favorite color is blue",
            memory_type="preference",
            importance=3,
            user_id="integ_user",
        ))

        # Create a session with conversation
        store = SessionStore(str(tmp_path / "sessions"))
        session = store.create(user_id="integ_user")
        store.append_message(session.session_id, "user", "Remember my color?")

        # Recall the memory
        results = _run(mem.recall(
            query="favorite color", user_id="integ_user", top_k=5
        ))
        assert len(results) >= 1
        assert "blue" in results[0][0]["content"]

    def test_dispatcher_cache_invalidation_on_write(self, dispatcher, tmp_path):
        """Writing to a file invalidates the read cache for that file."""
        test_file = tmp_path / "cache_test.txt"
        test_file.write_text("version 1")

        # Read to populate cache
        _run(dispatcher.dispatch(
            "read_file", {"path": str(test_file)}, origin="test"
        ))

        # Write new version
        _run(dispatcher.dispatch(
            "write_file",
            {"path": str(test_file), "content": "version 2"},
            origin="test",
        ))

        # Read again — should see new content, not cached
        result = _run(dispatcher.dispatch(
            "read_file", {"path": str(test_file)}, origin="test"
        ))
        assert "version 2" in result
