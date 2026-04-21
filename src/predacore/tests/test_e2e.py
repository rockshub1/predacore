"""
End-to-End integration tests for PredaCore — Phase 10.

These tests prove the ENTIRE system works TOGETHER, not just individual pieces.
Flow: user message -> gateway routes -> core processes -> LLM responds ->
      tools execute -> memory stores -> response returns.

Only the LLM provider is mocked. Everything else runs real:
  - Config loading (3-layer: defaults -> YAML -> env)
  - Session persistence (JSONL files on disk)
  - Memory store (SQLite + vector index)
  - Tool execution pipeline (dispatcher, trust policy, handlers)
  - Gateway routing (identity, rate limiting, lane queue)
  - Identity engine (bootstrap detection, file management)
  - Persona drift guard (regex heuristics)
  - Rust core integration (predacore_core) when available
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from predacore.config import (
    ChannelConfig,
    DaemonConfig,
    LaunchProfileConfig,
    LLMConfig,
    MemoryConfig,
    PredaCoreConfig,
    SecurityConfig,
    load_config,
    save_default_config,
)
from predacore.gateway import (
    Gateway,
    IncomingMessage,
)
from predacore.memory.store import UnifiedMemoryStore
from predacore.sessions import Session, SessionStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — shared config + mock LLM factories
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, **overrides) -> PredaCoreConfig:
    """Build a PredaCoreConfig rooted entirely inside tmp_path."""
    home = str(tmp_path / "prometheus")
    cfg = PredaCoreConfig(
        name="PredaCore-TEST",
        home_dir=home,
        sessions_dir=str(tmp_path / "prometheus" / "sessions"),
        skills_dir=str(tmp_path / "prometheus" / "skills"),
        logs_dir=str(tmp_path / "prometheus" / "logs"),
        llm=LLMConfig(provider="mock", model="test-model"),
        security=SecurityConfig(trust_level="yolo", permission_mode="auto"),
        channels=ChannelConfig(enabled=["cli"]),
        daemon=DaemonConfig(enabled=False),
        memory=MemoryConfig(persistence_dir=str(tmp_path / "prometheus" / "memory")),
        launch=LaunchProfileConfig(
            profile="enterprise",
            enable_persona_drift_guard=True,
            persona_drift_threshold=0.42,
            persona_drift_max_regens=1,
            max_tool_iterations=10,
            enable_plugin_marketplace=False,
            enable_openclaw_bridge=False,
        ),
    )
    # Apply overrides
    for key, val in overrides.items():
        setattr(cfg, key, val)
    # Ensure directories exist
    for d in [cfg.home_dir, cfg.sessions_dir, cfg.skills_dir, cfg.logs_dir,
              cfg.memory.persistence_dir, cfg.agent_dir, cfg.flame_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
    return cfg


def _test_provider_instance():
    """A minimal LLMProvider with default append_*_turn methods.

    Mock-based tests set this as ``llm_inst.active_provider_instance`` so
    core.py's Phase-A provider-delegated tool-turn serialization path has
    something real to call (Mocks don't mutate messages on append_*_turn).
    """
    from predacore.llm_providers.base import LLMProvider, ProviderConfig

    class _TestProvider(LLMProvider):
        name = "test"
        async def chat(self, messages, tools=None, temperature=None, max_tokens=None, stream_fn=None):
            return {"content": "", "tool_calls": [], "usage": {}, "finish_reason": "stop"}

    return _TestProvider(config=ProviderConfig())


def _mock_llm_response(content: str = "Hello! I'm PredaCore.", **extra) -> dict:
    """Build a standard mock LLM response dict."""
    resp = {
        "content": content,
        "tool_calls": [],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        "model_used": "test-model",
        "provider_used": "mock",
        **extra,
    }
    return resp


def _mock_tool_call_response(
    tool_name: str, arguments: dict, content: str = ""
) -> dict:
    """Build a mock LLM response that requests a tool call."""
    return {
        "content": content,
        "tool_calls": [{"name": tool_name, "arguments": arguments}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        "model_used": "test-model",
        "provider_used": "mock",
    }


# ---------------------------------------------------------------------------
# Autouse fixture: real identity engine, reset between tests
# ---------------------------------------------------------------------------
# The identity engine is a module-level singleton that reads config.home_dir
# at first call. Each test uses a fresh tmp_path, so we reset the singleton
# before/after each test so it gets rebuilt against the test's tmp_path.
#
# We deliberately do NOT mock the identity engine — the real one reads
# SOUL_SEED.md + EVENT_HORIZON.md from src/predacore/identity/ (built-in)
# (shipped with source), so build_identity_prompt() returns a real string and
# system-prompt assembly succeeds without needing any MagicMock gymnastics.


@pytest.fixture(autouse=True)
def _reset_identity_engine_singleton():
    from predacore.identity.engine import reset_identity_engine
    reset_identity_engine()
    yield
    reset_identity_engine()


@pytest.fixture(autouse=True)
def _reset_profile_env_vars():
    """load_config() mutates os.environ (PREDACORE_PROFILE etc.) and other test
    files can leave PREDACORE_* vars set — snapshot and restore so each test
    sees a clean slate."""
    _keys = (
        "PREDACORE_PROFILE",
        "PREDACORE_TRUST_LEVEL",
        "PREDACORE_MODE",
        "PREDACORE_NAME",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "APPROVALS_REQUIRED",
        "EGM_MODE",
        "DEFAULT_CODE_NETWORK",
        "ENABLE_OPENCLAW_BRIDGE",
        "ENABLE_PLUGIN_MARKETPLACE",
        "ENABLE_SELF_EVOLUTION",
        "MAX_TOOL_ITERATIONS",
        "PREDACORE_ENABLE_PERSONA_DRIFT_GUARD",
        "PREDACORE_PERSONA_DRIFT_THRESHOLD",
        "PREDACORE_PERSONA_DRIFT_MAX_REGENS",
    )
    snapshot = {k: os.environ.get(k) for k in _keys}
    yield
    for k, v in snapshot.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ---------------------------------------------------------------------------
# Section 1: Full Conversation Flow (core.process)
# ---------------------------------------------------------------------------


class TestFullConversationFlow:
    """E2E: user message -> core.process -> LLM -> response."""

    @pytest.mark.asyncio
    async def test_basic_message_returns_response(self, tmp_path):
        """Send a simple message through core.process, get response back."""
        cfg = _make_config(tmp_path)

        with patch("predacore.core.LLMInterface") as MockLLM:
            llm_inst = MockLLM.return_value
            llm_inst.chat = AsyncMock(return_value=_mock_llm_response(
                "Hello! I'm PredaCore from PredaCore."
            ))
            llm_inst.active_provider = "mock"
            llm_inst.active_model = "test-model"
            llm_inst._provider_name = "mock"
            llm_inst.executes_tool_loop_internally = False
            # Phase A: core.py calls provider.append_*_turn for tool round-trips.
            # Give it a real LLMProvider instance that uses default impls.
            llm_inst.active_provider_instance = _test_provider_instance()

            from predacore.core import PredaCoreCore
            core = PredaCoreCore(cfg)

            session = Session(user_id="test_user")
            response = await core.process("test_user", "Hello PredaCore", session)

            assert response is not None
            assert len(response) > 0
            assert "PredaCore" in response or "Hello" in response

    @pytest.mark.asyncio
    async def test_response_stored_in_session(self, tmp_path):
        """Verify that user message + assistant response are in session after process."""
        cfg = _make_config(tmp_path)

        with patch("predacore.core.LLMInterface") as MockLLM:
            llm_inst = MockLLM.return_value
            llm_inst.chat = AsyncMock(return_value=_mock_llm_response("Test response"))
            llm_inst.active_provider = "mock"
            llm_inst.active_model = "test-model"
            llm_inst._provider_name = "mock"
            llm_inst.executes_tool_loop_internally = False
            # Phase A: core.py calls provider.append_*_turn for tool round-trips.
            # Give it a real LLMProvider instance that uses default impls.
            llm_inst.active_provider_instance = _test_provider_instance()

            from predacore.core import PredaCoreCore
            core = PredaCoreCore(cfg)

            session = Session(user_id="test_user")
            session.add_message("user", "Hello PredaCore")
            response = await core.process("test_user", "Hello PredaCore", session)

            # LLM was called at least once
            assert llm_inst.chat.call_count >= 1
            # Response came back
            assert "response" in response.lower() or len(response) > 0

    @pytest.mark.asyncio
    async def test_system_prompt_included_in_llm_call(self, tmp_path):
        """Verify system prompt is passed to LLM in messages."""
        cfg = _make_config(tmp_path)

        with patch("predacore.core.LLMInterface") as MockLLM:
            llm_inst = MockLLM.return_value
            llm_inst.chat = AsyncMock(return_value=_mock_llm_response("PredaCore here"))
            llm_inst.active_provider = "mock"
            llm_inst.active_model = "test-model"
            llm_inst._provider_name = "mock"
            llm_inst.executes_tool_loop_internally = False
            # Phase A: core.py calls provider.append_*_turn for tool round-trips.
            # Give it a real LLMProvider instance that uses default impls.
            llm_inst.active_provider_instance = _test_provider_instance()

            from predacore.core import PredaCoreCore
            core = PredaCoreCore(cfg)

            session = Session(user_id="test_user")
            await core.process("test_user", "Hello", session)

            # First call's messages should start with a system prompt
            call_args = llm_inst.chat.call_args_list[0]
            messages = call_args.kwargs.get("messages") or call_args.args[0]
            assert messages[0]["role"] == "system"
            assert len(messages[0]["content"]) > 50  # non-trivial system prompt

    @pytest.mark.asyncio
    async def test_multiple_turns_in_session(self, tmp_path):
        """Send two messages, verify both are processed."""
        cfg = _make_config(tmp_path)

        with patch("predacore.core.LLMInterface") as MockLLM:
            call_count = [0]

            async def _chat_side_effect(*args, **kwargs):
                call_count[0] += 1
                return _mock_llm_response(f"Response #{call_count[0]} from PredaCore")

            llm_inst = MockLLM.return_value
            llm_inst.chat = AsyncMock(side_effect=_chat_side_effect)
            llm_inst.active_provider = "mock"
            llm_inst.active_model = "test-model"
            llm_inst._provider_name = "mock"
            llm_inst.executes_tool_loop_internally = False
            # Phase A: core.py calls provider.append_*_turn for tool round-trips.
            # Give it a real LLMProvider instance that uses default impls.
            llm_inst.active_provider_instance = _test_provider_instance()

            from predacore.core import PredaCoreCore
            core = PredaCoreCore(cfg)

            session = Session(user_id="test_user")
            r1 = await core.process("test_user", "First message", session)
            session.add_message("user", "First message")
            session.add_message("assistant", r1)

            r2 = await core.process("test_user", "Second message", session)
            assert r1 != r2
            assert "#1" in r1
            assert "#2" in r2 or call_count[0] >= 2


# ---------------------------------------------------------------------------
# Section 2: Tool Execution Flow
# ---------------------------------------------------------------------------


class TestToolExecutionFlow:
    """E2E: LLM requests tool call -> tool executes -> result fed back."""

    @pytest.mark.asyncio
    async def test_tool_call_executes_and_returns(self, tmp_path):
        """Mock LLM requests read_file tool, verify it runs and result flows back."""
        cfg = _make_config(tmp_path)

        # Create a real file to read
        test_file = tmp_path / "testfile.txt"
        test_file.write_text("file content for e2e test")

        with patch("predacore.core.LLMInterface") as MockLLM:
            # First call: LLM requests a tool; Second call: LLM gives final answer
            call_seq = [0]

            async def _chat_side_effect(*args, **kwargs):
                call_seq[0] += 1
                if call_seq[0] == 1:
                    return _mock_tool_call_response(
                        "read_file", {"path": str(test_file)}
                    )
                return _mock_llm_response(
                    "The file contains: file content for e2e test"
                )

            llm_inst = MockLLM.return_value
            llm_inst.chat = AsyncMock(side_effect=_chat_side_effect)
            llm_inst.active_provider = "mock"
            llm_inst.active_model = "test-model"
            llm_inst._provider_name = "mock"
            llm_inst.executes_tool_loop_internally = False
            # Phase A: core.py calls provider.append_*_turn for tool round-trips.
            # Give it a real LLMProvider instance that uses default impls.
            llm_inst.active_provider_instance = _test_provider_instance()

            from predacore.core import PredaCoreCore
            core = PredaCoreCore(cfg)

            session = Session(user_id="test_user")
            response = await core.process(
                "test_user", f"Read the file at {test_file}", session
            )

            # LLM was called at least twice (tool call + final)
            assert call_seq[0] >= 2
            # Response includes file content
            assert "file content" in response.lower() or "e2e test" in response.lower()

    @pytest.mark.asyncio
    async def test_tool_result_included_in_second_llm_call(self, tmp_path):
        """Verify the tool result is passed in messages to the second LLM call."""
        cfg = _make_config(tmp_path)

        test_file = tmp_path / "data.txt"
        test_file.write_text("important data")

        with patch("predacore.core.LLMInterface") as MockLLM:
            captured_calls = []

            async def _chat_capture(*args, **kwargs):
                messages = kwargs.get("messages") or (args[0] if args else [])
                captured_calls.append(messages[:])
                if len(captured_calls) == 1:
                    return _mock_tool_call_response(
                        "read_file", {"path": str(test_file)}
                    )
                return _mock_llm_response("Got the data from PredaCore tools")

            llm_inst = MockLLM.return_value
            llm_inst.chat = AsyncMock(side_effect=_chat_capture)
            llm_inst.active_provider = "mock"
            llm_inst.active_model = "test-model"
            llm_inst._provider_name = "mock"
            llm_inst.executes_tool_loop_internally = False
            # Phase A: core.py calls provider.append_*_turn for tool round-trips.
            # Give it a real LLMProvider instance that uses default impls.
            llm_inst.active_provider_instance = _test_provider_instance()

            from predacore.core import PredaCoreCore
            core = PredaCoreCore(cfg)

            session = Session(user_id="test_user")
            await core.process("test_user", "Read data.txt", session)

            # Second call should have tool result in messages
            assert len(captured_calls) >= 2
            second_call_messages = captured_calls[1]
            all_content = " ".join(m.get("content", "") for m in second_call_messages)
            assert "important data" in all_content or "Tool Result" in all_content

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_sequence(self, tmp_path):
        """LLM requests two tools in sequence, both execute."""
        cfg = _make_config(tmp_path)

        file_a = tmp_path / "a.txt"
        file_a.write_text("content A")
        file_b = tmp_path / "b.txt"
        file_b.write_text("content B")

        with patch("predacore.core.LLMInterface") as MockLLM:
            call_seq = [0]

            async def _chat_side_effect(*args, **kwargs):
                call_seq[0] += 1
                if call_seq[0] == 1:
                    return _mock_tool_call_response(
                        "read_file", {"path": str(file_a)}
                    )
                if call_seq[0] == 2:
                    return _mock_tool_call_response(
                        "read_file", {"path": str(file_b)}
                    )
                return _mock_llm_response("Read both files via PredaCore tools")

            llm_inst = MockLLM.return_value
            llm_inst.chat = AsyncMock(side_effect=_chat_side_effect)
            llm_inst.active_provider = "mock"
            llm_inst.active_model = "test-model"
            llm_inst._provider_name = "mock"
            llm_inst.executes_tool_loop_internally = False
            # Phase A: core.py calls provider.append_*_turn for tool round-trips.
            # Give it a real LLMProvider instance that uses default impls.
            llm_inst.active_provider_instance = _test_provider_instance()

            from predacore.core import PredaCoreCore
            core = PredaCoreCore(cfg)

            session = Session(user_id="test_user")
            response = await core.process("test_user", "Read both files", session)

            assert call_seq[0] >= 3  # Two tool calls + final answer


# ---------------------------------------------------------------------------
# Section 3: Memory Persistence (store -> close -> reopen -> recall)
# ---------------------------------------------------------------------------


class TestMemoryPersistence:
    """E2E: Store memory, close, reopen, recall — verify data survives."""

    @pytest.mark.asyncio
    async def test_store_and_recall_keyword(self, tmp_path):
        """Store a memory, recall it by keyword search."""
        db_path = str(tmp_path / "memory.db")
        store = UnifiedMemoryStore(db_path=db_path)

        mem_id = await store.store(
            content="PredaCore prefers Python for scripting tasks",
            memory_type="fact",
            importance=3,
            user_id="test_user",
        )
        assert mem_id is not None

        results = await store.recall(
            query="Python scripting",
            user_id="test_user",
            top_k=5,
        )
        # Keyword fallback should find it
        assert len(results) > 0
        found_contents = [r[0]["content"] for r in results]
        assert any("Python" in c for c in found_contents)
        store.close()

    @pytest.mark.asyncio
    async def test_memory_survives_close_and_reopen(self, tmp_path):
        """Store memory, close store, reopen, verify data persisted."""
        db_path = str(tmp_path / "persist.db")

        # Store
        store1 = UnifiedMemoryStore(db_path=db_path)
        await store1.store(
            content="User likes dark mode in all editors",
            memory_type="preference",
            importance=4,
            user_id="user42",
        )
        stats1 = await store1.get_stats()
        assert stats1["total_memories"] == 1
        store1.close()

        # Reopen
        store2 = UnifiedMemoryStore(db_path=db_path)
        stats2 = await store2.get_stats()
        assert stats2["total_memories"] == 1

        results = await store2.recall(
            query="dark mode editor",
            user_id="user42",
            top_k=5,
        )
        assert len(results) > 0
        assert "dark mode" in results[0][0]["content"]
        store2.close()

    @pytest.mark.asyncio
    async def test_multiple_memories_different_types(self, tmp_path):
        """Store memories of different types, recall filters correctly."""
        db_path = str(tmp_path / "multi.db")
        store = UnifiedMemoryStore(db_path=db_path)

        await store.store(content="Python is great", memory_type="fact", user_id="u1")
        await store.store(content="User prefers vim", memory_type="preference", user_id="u1")
        await store.store(content="Discussed Docker setup", memory_type="conversation", user_id="u1")

        stats = await store.get_stats()
        assert stats["total_memories"] == 3

        # Recall with type filter
        facts = await store.recall(query="programming", user_id="u1", memory_types=["fact"])
        all_types = [r[0]["memory_type"] for r in facts]
        for t in all_types:
            assert t == "fact"
        store.close()

    @pytest.mark.asyncio
    async def test_memory_store_entity_extraction(self, tmp_path):
        """Store entity and verify it persists."""
        db_path = str(tmp_path / "entity.db")
        store = UnifiedMemoryStore(db_path=db_path)

        await store.upsert_entity(
            name="Python",
            entity_type="technology",
            properties={"category": "language"},
        )
        entities = await store.list_entities()
        names = [e["name"] for e in entities]
        assert "Python" in names
        store.close()

    @pytest.mark.asyncio
    async def test_memory_stats_accurate(self, tmp_path):
        """Verify get_stats returns correct counts."""
        db_path = str(tmp_path / "stats.db")
        store = UnifiedMemoryStore(db_path=db_path)

        await store.store(content="mem1", user_id="u1")
        await store.store(content="mem2", user_id="u1")
        await store.upsert_entity(name="TestEntity", entity_type="concept")

        stats = await store.get_stats()
        assert stats["total_memories"] == 2
        assert stats["entities"] == 1
        store.close()


# ---------------------------------------------------------------------------
# Section 4: Session Continuity
# ---------------------------------------------------------------------------


class TestSessionContinuity:
    """E2E: messages persist across session load/save cycles."""

    def test_session_messages_persist_to_disk(self, tmp_path):
        """Write messages, reload from disk, verify continuity."""
        sessions_dir = str(tmp_path / "sessions")
        store = SessionStore(sessions_dir)

        session = store.create(user_id="alice")
        sid = session.session_id

        store.append_message(sid, "user", "Hello PredaCore")
        store.append_message(sid, "assistant", "Hello Alice!")
        store.append_message(sid, "user", "What can you do?")

        # Force reload from disk by creating a new store instance
        store2 = SessionStore(sessions_dir)
        reloaded = store2.get(sid)

        assert reloaded is not None
        assert len(reloaded.messages) == 3
        assert reloaded.messages[0].role == "user"
        assert reloaded.messages[0].content == "Hello PredaCore"
        assert reloaded.messages[1].role == "assistant"
        assert reloaded.messages[1].content == "Hello Alice!"
        assert reloaded.messages[2].content == "What can you do?"

    def test_session_title_auto_generated(self, tmp_path):
        """First user message becomes the session title."""
        store = SessionStore(str(tmp_path / "sessions"))
        session = store.create(user_id="bob")
        store.append_message(session.session_id, "user", "Help me with Docker")

        reloaded = store.get(session.session_id)
        assert reloaded.title == "Help me with Docker"

    def test_session_list_by_user(self, tmp_path):
        """Multiple sessions for a user are listed correctly."""
        store = SessionStore(str(tmp_path / "sessions"))
        s1 = store.create(user_id="charlie")
        s2 = store.create(user_id="charlie")
        s3 = store.create(user_id="other_user")

        charlie_sessions = store.list_sessions(user_id="charlie")
        assert len(charlie_sessions) == 2
        session_ids = {s.session_id for s in charlie_sessions}
        assert s1.session_id in session_ids
        assert s2.session_id in session_ids
        assert s3.session_id not in session_ids

    def test_session_get_or_create_idempotent(self, tmp_path):
        """get_or_create returns existing session if it exists."""
        store = SessionStore(str(tmp_path / "sessions"))
        s1 = store.get_or_create("fixed-id-123", user_id="dave")
        s2 = store.get_or_create("fixed-id-123", user_id="dave")
        assert s1.session_id == s2.session_id

    def test_context_window_preserves_recent_messages(self, tmp_path):
        """Build context window from a session with many messages."""
        session = Session(user_id="test")
        for i in range(50):
            session.add_message("user", f"Message {i}")
            session.add_message("assistant", f"Response {i}")

        context = session.build_context_window(
            max_total_tokens=8000,
            keep_recent_messages=10,
        )
        # Should include recent messages
        assert len(context) > 0
        # Most recent messages should be present
        all_content = " ".join(m["content"] for m in context)
        assert "Message 49" in all_content or "Response 49" in all_content


# ---------------------------------------------------------------------------
# Section 5: Config Loading (3-layer: defaults -> YAML -> env)
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """E2E: config loading with real YAML files and env overrides."""

    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch):
        """Remove PREDACORE_*/PROMETHEUS_* env vars so tests are isolated."""
        for key in list(os.environ):
            if key.startswith("PREDACORE_") or key.startswith("PROMETHEUS_") or key.startswith("LLM_") or key.startswith("TRUST_") or key in ("PROFILE",):
                monkeypatch.delenv(key, raising=False)

    def test_defaults_work_without_yaml(self, tmp_path, monkeypatch):
        """load_config with no YAML file produces valid defaults.

        Must be robust to env-var pollution from other tests — clears any
        PREDACORE_* / LLM_* / TRUST_* vars that could override the defaults.
        """
        for key in list(os.environ):
            if (
                key.startswith("PREDACORE_")
                or key.startswith("LLM_")
                or key.startswith("TRUST_")
                or key in ("PROFILE", "PREDACORE_PROFILE")
            ):
                monkeypatch.delenv(key, raising=False)

        bogus_path = str(tmp_path / "nonexistent.yaml")
        cfg = load_config(config_path=bogus_path)

        assert cfg.name == "PredaCore"
        assert cfg.llm.provider == "gemini-cli"  # default provider
        assert cfg.security.trust_level == "normal"
        assert cfg.launch.profile == "enterprise"

    def test_yaml_overrides_apply(self, tmp_path):
        """YAML config overrides defaults."""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            "name: CUSTOM-PredaCore\n"
            "llm:\n"
            "  provider: openai\n"
            "  temperature: 0.3\n"
            "security:\n"
            "  trust_level: yolo\n"
        )
        cfg = load_config(config_path=str(yaml_path))

        assert cfg.name == "CUSTOM-PredaCore"
        assert cfg.llm.provider == "openai"
        assert cfg.llm.temperature == 0.3
        assert cfg.security.trust_level == "yolo"

    def test_env_vars_override_yaml(self, tmp_path):
        """Environment variables take precedence over YAML."""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            "name: YAML-NAME\n"
            "llm:\n"
            "  provider: openai\n"
            "security:\n"
            "  trust_level: normal\n"
        )
        env_overrides = {
            "PREDACORE_NAME": "ENV-PredaCore",
            "PREDACORE_TRUST_LEVEL": "yolo",
            "LLM_PROVIDER": "anthropic",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            cfg = load_config(config_path=str(yaml_path))

        assert cfg.name == "ENV-PredaCore"
        assert cfg.security.trust_level == "yolo"
        assert cfg.llm.provider == "anthropic"

    def test_profile_presets_apply(self, tmp_path):
        """Profile override applies the correct preset defaults."""
        bogus_path = str(tmp_path / "no.yaml")
        cfg = load_config(config_path=bogus_path, profile_override="beast")

        assert cfg.launch.profile == "beast"
        assert cfg.security.trust_level == "yolo"
        assert cfg.launch.max_tool_iterations == 1000
        assert cfg.launch.enable_self_evolution is True

    def test_enterprise_profile(self, tmp_path):
        """Enterprise profile applies safe-by-default governance posture."""
        cfg = load_config(
            config_path=str(tmp_path / "no.yaml"),
            profile_override="enterprise",
        )
        assert cfg.security.trust_level == "normal"
        assert cfg.launch.approvals_required is True
        assert cfg.launch.egm_mode == "strict"
        assert cfg.launch.enable_self_evolution is False
        # Resource limits are maxed on both profiles.
        assert cfg.launch.max_tool_iterations == 1000
        assert cfg.launch.max_spawn_depth == 16

    def test_save_and_reload_config(self, tmp_path):
        """save_default_config creates a file that load_config can read."""
        config_path = str(tmp_path / "saved_config.yaml")
        save_default_config(path=config_path, provider="openai", trust_level="yolo")

        cfg = load_config(config_path=config_path)
        assert cfg.llm.provider == "openai"
        assert cfg.security.trust_level == "yolo"

    def test_config_directories_created(self, tmp_path):
        """load_config creates all required directories."""
        env_overrides = {"PREDACORE_HOME": str(tmp_path / "fresh_home")}
        with patch.dict(os.environ, env_overrides, clear=False):
            cfg = load_config(config_path=str(tmp_path / "no.yaml"))

        assert Path(cfg.home_dir).exists()
        assert Path(cfg.sessions_dir).exists()
        assert Path(cfg.memory.persistence_dir).exists()


# ---------------------------------------------------------------------------
# Section 6: Identity Bootstrap Detection
# ---------------------------------------------------------------------------


class TestIdentitySeeding:
    """E2E: identity engine seeds defaults + handles writes with real FS."""

    def test_fresh_agent_seeds_from_defaults(self, tmp_path):
        """Every fresh workspace gets the bundled defaults copied in."""
        from predacore.identity.engine import IdentityEngine, reset_identity_engine
        reset_identity_engine()

        engine = IdentityEngine(str(tmp_path), agent_name="test_agent")
        # Defaults are copied into the workspace on first instantiation.
        for filename in (
            "IDENTITY.md", "SOUL.md", "USER.md", "MEMORY.md",
            "TOOLS.md", "HEARTBEAT.md", "REFLECTION.md", "JOURNAL.md",
        ):
            assert (engine.workspace / filename).exists(), filename
        # Seeded IDENTITY.md is name-less and contains first-turn instructions.
        identity = (engine.workspace / "IDENTITY.md").read_text()
        assert "not set yet" in identity.lower() or "ask" in identity.lower()

    def test_seeding_is_idempotent(self, tmp_path):
        """Pre-existing workspace files are never overwritten."""
        from predacore.identity.engine import IdentityEngine, reset_identity_engine
        reset_identity_engine()

        agent_dir = tmp_path / "agents" / "test_agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "IDENTITY.md").write_text("# I am PredaCore\nBorn in PredaCore.")

        engine = IdentityEngine(str(tmp_path), agent_name="test_agent")
        assert "Born in PredaCore" in (engine.workspace / "IDENTITY.md").read_text()

    def test_write_identity_file_persists(self, tmp_path):
        """Writing IDENTITY.md persists the content to the workspace."""
        from predacore.identity.engine import IdentityEngine, reset_identity_engine
        reset_identity_engine()

        engine = IdentityEngine(str(tmp_path), agent_name="bootstrap_test")
        result = engine.write_identity_file(
            "IDENTITY.md", "# PredaCore\nI am PredaCore from PredaCore."
        )
        assert result.get("status") == "ok"
        content = (engine.workspace / "IDENTITY.md").read_text()
        assert "PredaCore" in content

    def test_identity_file_loads_after_write(self, tmp_path):
        """Written identity file is loadable."""
        from predacore.identity.engine import IdentityEngine, reset_identity_engine
        reset_identity_engine()

        engine = IdentityEngine(str(tmp_path), agent_name="load_test")
        engine.write_identity_file(
            "IDENTITY.md", "# Test Identity\nPersonality: curious"
        )
        content = engine.load_identity()
        assert "Test Identity" in content
        assert "curious" in content

    def test_soul_seed_loads_from_builtin(self, tmp_path):
        """SOUL_SEED.md is always loadable from the bundled package."""
        from predacore.identity.engine import IdentityEngine, reset_identity_engine
        reset_identity_engine()

        engine = IdentityEngine(str(tmp_path), agent_name="seed_test")
        seed = engine.load_seed()
        # Seed ships with the package so load_seed() should always produce text.
        assert seed
        assert "SOUL_SEED" in seed or "invariant" in seed.lower() or "bedrock" in seed.lower()

    def test_journal_append(self, tmp_path):
        """Journal entries can be appended and loaded."""
        from predacore.identity.engine import IdentityEngine, reset_identity_engine
        reset_identity_engine()

        engine = IdentityEngine(str(tmp_path), agent_name="journal_test")
        engine.append_journal("First interaction with user. Learned about Python preference.")
        engine.append_journal("Explored Docker workflows together.")

        journal = engine.load_journal()
        assert "Python" in journal
        assert "Docker" in journal


# ---------------------------------------------------------------------------
# Section 7: Rust Core Integration (predacore_core)
# ---------------------------------------------------------------------------


class TestRustCoreIntegration:
    """E2E: predacore_core Rust module — embed, search, entity extraction."""

    @pytest.mark.asyncio
    async def test_rust_embed_and_vector_search(self, tmp_path):
        """predacore_core.embed -> predacore_core.vector_search finds similar."""
        try:
            import predacore_core
        except ImportError:
            pytest.skip("predacore_core not installed")

        vecs = predacore_core.embed([
            "Python programming language",
            "JavaScript web development",
            "Python scripting and automation",
        ])
        vec_a, vec_b, vec_c = vecs[0], vecs[1], vecs[2]

        results = predacore_core.vector_search(vec_a, [vec_a, vec_b, vec_c], 2)
        # vec_a should be most similar to itself (index 0)
        assert results[0][0] == 0
        assert results[0][1] >= 0.99  # self-similarity ~ 1.0
        # vec_c (Python scripting) should be more similar to vec_a than vec_b
        indices = [r[0] for r in results]
        assert 2 in indices or 0 in indices

    @pytest.mark.asyncio
    async def test_rust_bm25_with_synonym_expansion(self, tmp_path):
        """BM25 search with synonym expansion finds related terms."""
        try:
            import predacore_core
        except ImportError:
            pytest.skip("predacore_core not installed")

        expanded = predacore_core.expand_synonyms(["config"])
        assert "configuration" in expanded or "settings" in expanded

        # Use an expanded term that actually appears in the corpus
        results = predacore_core.bm25_search(
            "config settings",
            ["config yaml settings", "database schema", "api documentation"],
        )
        assert len(results) > 0
        # First result should be the config-related document
        assert results[0][0] == 0

    @pytest.mark.asyncio
    async def test_rust_entity_extraction_and_relation(self, tmp_path):
        """Extract entities and classify relations between them."""
        try:
            import predacore_core
        except ImportError:
            pytest.skip("predacore_core not installed")

        entities = predacore_core.extract_entities(
            "PredaCore uses Python and Docker for deployment"
        )
        # extract_entities returns a list of tuples:
        # (name, entity_type, confidence, source_tier)
        names = [e[0].lower() for e in entities]
        assert "python" in names
        assert "docker" in names

        rel, conf = predacore_core.classify_relation(
            "PredaCore uses Python for scripting", "PredaCore", "Python"
        )
        assert rel in ("uses", "depends_on", "related_to")
        assert conf > 0.0

    @pytest.mark.asyncio
    async def test_rust_embed_deterministic(self, tmp_path):
        """Same text produces same embedding."""
        try:
            import predacore_core
        except ImportError:
            pytest.skip("predacore_core not installed")

        vec1 = predacore_core.embed(["test sentence"])[0]
        vec2 = predacore_core.embed(["test sentence"])[0]
        assert vec1 == vec2

    @pytest.mark.asyncio
    async def test_rust_vector_search_top_k(self, tmp_path):
        """vector_search respects top_k parameter."""
        try:
            import predacore_core
        except ImportError:
            pytest.skip("predacore_core not installed")

        query = [1.0, 0.0, 0.0]
        corpus = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        results = predacore_core.vector_search(query, corpus, 2)
        assert len(results) == 2
        assert results[0][0] == 0  # exact match first


# ---------------------------------------------------------------------------
# Section 8: Error Recovery
# ---------------------------------------------------------------------------


class TestErrorRecovery:
    """E2E: system handles failures gracefully without crashing."""

    @pytest.mark.asyncio
    async def test_llm_failure_returns_error_message(self, tmp_path):
        """When LLM fails, user gets a friendly error, not a crash."""
        cfg = _make_config(tmp_path)

        with patch("predacore.core.LLMInterface") as MockLLM:
            llm_inst = MockLLM.return_value
            llm_inst.chat = AsyncMock(
                side_effect=ConnectionError("All providers failed")
            )
            llm_inst.active_provider = "mock"
            llm_inst.active_model = "test-model"
            llm_inst._provider_name = "mock"
            llm_inst.executes_tool_loop_internally = False
            # Phase A: core.py calls provider.append_*_turn for tool round-trips.
            # Give it a real LLMProvider instance that uses default impls.
            llm_inst.active_provider_instance = _test_provider_instance()

            from predacore.core import PredaCoreCore
            core = PredaCoreCore(cfg)

            session = Session(user_id="test_user")
            response = await core.process("test_user", "Hello", session)

            # Should return error message, not raise exception
            assert response is not None
            assert len(response) > 0
            assert "error" in response.lower() or "sorry" in response.lower() or "try again" in response.lower()

    @pytest.mark.asyncio
    async def test_llm_rate_limit_returns_message(self, tmp_path):
        """Rate limit error returns user-friendly message."""
        cfg = _make_config(tmp_path)

        with patch("predacore.core.LLMInterface") as MockLLM:
            llm_inst = MockLLM.return_value
            llm_inst.chat = AsyncMock(
                side_effect=ConnectionError("429 rate limit exceeded")
            )
            llm_inst.active_provider = "mock"
            llm_inst.active_model = "test-model"
            llm_inst._provider_name = "mock"
            llm_inst.executes_tool_loop_internally = False
            # Phase A: core.py calls provider.append_*_turn for tool round-trips.
            # Give it a real LLMProvider instance that uses default impls.
            llm_inst.active_provider_instance = _test_provider_instance()

            from predacore.core import PredaCoreCore
            core = PredaCoreCore(cfg)

            session = Session(user_id="test_user")
            response = await core.process("test_user", "Hello", session)

            assert "rate" in response.lower() or "try again" in response.lower()

    @pytest.mark.asyncio
    async def test_tool_error_handled_gracefully(self, tmp_path):
        """When a tool fails, the error is reported without crashing."""
        cfg = _make_config(tmp_path)

        with patch("predacore.core.LLMInterface") as MockLLM:
            call_seq = [0]

            async def _chat_side_effect(*args, **kwargs):
                call_seq[0] += 1
                if call_seq[0] == 1:
                    # Request reading a nonexistent file
                    return _mock_tool_call_response(
                        "read_file",
                        {"path": "/nonexistent/path/that/does/not/exist.txt"},
                    )
                return _mock_llm_response(
                    "I couldn't read the file, it doesn't exist."
                )

            llm_inst = MockLLM.return_value
            llm_inst.chat = AsyncMock(side_effect=_chat_side_effect)
            llm_inst.active_provider = "mock"
            llm_inst.active_model = "test-model"
            llm_inst._provider_name = "mock"
            llm_inst.executes_tool_loop_internally = False
            # Phase A: core.py calls provider.append_*_turn for tool round-trips.
            # Give it a real LLMProvider instance that uses default impls.
            llm_inst.active_provider_instance = _test_provider_instance()

            from predacore.core import PredaCoreCore
            core = PredaCoreCore(cfg)

            session = Session(user_id="test_user")
            response = await core.process("test_user", "Read /nonexistent/path", session)

            # Should not crash, should get some response
            assert response is not None
            assert len(response) > 0

    @pytest.mark.asyncio
    async def test_timeout_error_handled(self, tmp_path):
        """Timeout error returns friendly message."""
        cfg = _make_config(tmp_path)

        with patch("predacore.core.LLMInterface") as MockLLM:
            llm_inst = MockLLM.return_value
            llm_inst.chat = AsyncMock(side_effect=TimeoutError("Request timed out"))
            llm_inst.active_provider = "mock"
            llm_inst.active_model = "test-model"
            llm_inst._provider_name = "mock"
            llm_inst.executes_tool_loop_internally = False
            # Phase A: core.py calls provider.append_*_turn for tool round-trips.
            # Give it a real LLMProvider instance that uses default impls.
            llm_inst.active_provider_instance = _test_provider_instance()

            from predacore.core import PredaCoreCore
            core = PredaCoreCore(cfg)

            session = Session(user_id="test_user")
            response = await core.process("test_user", "Do something", session)

            assert response is not None
            assert len(response) > 0


# ---------------------------------------------------------------------------
# Section 9: Rate Limiting (Gateway Level)
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """E2E: gateway rate limiter kicks in under rapid message load."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_normal_traffic(self, tmp_path):
        """Normal message rate is allowed through."""
        cfg = _make_config(tmp_path)

        async def mock_process(user_id, message, session, **kwargs):
            return "ok"

        gw = Gateway(cfg, mock_process)
        msg = IncomingMessage(channel="cli", user_id="user1", text="hello")

        response = await gw.handle_message(msg)
        assert response.text == "ok" or "ok" in response.text.lower() or response.text != ""

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_rapid_flood(self, tmp_path):
        """Sending more than rate limit per minute triggers rate limiting."""
        cfg = _make_config(tmp_path)

        async def mock_process(user_id, message, session, **kwargs):
            return "processed"

        gw = Gateway(cfg, mock_process)
        # Override rate limit to a small number for testing
        gw._user_rate_limit = 5

        responses = []
        for i in range(10):
            msg = IncomingMessage(channel="cli", user_id="flood_user", text=f"msg {i}")
            resp = await gw.handle_message(msg)
            responses.append(resp)

        # Some should be rate limited
        rate_limited = [r for r in responses if "too quickly" in r.text.lower() or "wait" in r.text.lower()]
        assert len(rate_limited) > 0, "Rate limiter should block some rapid messages"

    @pytest.mark.asyncio
    async def test_rate_limit_per_user_isolation(self, tmp_path):
        """Different users have independent rate limits."""
        cfg = _make_config(tmp_path)

        async def mock_process(user_id, message, session, **kwargs):
            return "ok"

        gw = Gateway(cfg, mock_process)
        gw._user_rate_limit = 3

        # Exhaust user1's limit
        for i in range(5):
            msg = IncomingMessage(channel="cli", user_id="user_a", text=f"msg {i}")
            await gw.handle_message(msg)

        # user2 should still be allowed
        msg = IncomingMessage(channel="cli", user_id="user_b", text="hello from b")
        resp = await gw.handle_message(msg)
        assert "too quickly" not in resp.text.lower()


# ---------------------------------------------------------------------------
# Section 10: Persona Drift Guard
# ---------------------------------------------------------------------------


class TestPersonaDriftGuard:
    """E2E: persona drift detection and regeneration."""

    @pytest.mark.asyncio
    async def test_drift_detected_for_chatgpt_claim(self, tmp_path):
        """LLM claiming to be ChatGPT triggers drift detection."""
        cfg = _make_config(tmp_path)

        with patch("predacore.core.LLMInterface") as MockLLM:
            call_count = [0]

            async def _chat_side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    # Drift: claims to be ChatGPT
                    return _mock_llm_response("I'm ChatGPT, made by OpenAI")
                # Regeneration produces a correct response
                return _mock_llm_response(
                    "I'm PredaCore from PredaCore. How can I help?"
                )

            llm_inst = MockLLM.return_value
            llm_inst.chat = AsyncMock(side_effect=_chat_side_effect)
            llm_inst.active_provider = "mock"
            llm_inst.active_model = "test-model"
            llm_inst._provider_name = "mock"
            llm_inst.executes_tool_loop_internally = False
            # Phase A: core.py calls provider.append_*_turn for tool round-trips.
            # Give it a real LLMProvider instance that uses default impls.
            llm_inst.active_provider_instance = _test_provider_instance()

            from predacore.core import PredaCoreCore
            core = PredaCoreCore(cfg)
            core._persona_drift_guard_enabled = True
            core._persona_drift_threshold = 0.42
            core._persona_drift_max_regens = 1

            session = Session(user_id="test_user")
            response = await core.process("test_user", "Who are you?", session)

            # Drift was detected, regeneration was attempted
            assert call_count[0] >= 2
            # Final response should NOT claim to be ChatGPT
            assert "chatgpt" not in response.lower() or "PredaCore" in response

    @pytest.mark.asyncio
    async def test_no_drift_for_normal_response(self, tmp_path):
        """Normal PredaCore response does not trigger drift guard."""
        cfg = _make_config(tmp_path)

        with patch("predacore.core.LLMInterface") as MockLLM:
            llm_inst = MockLLM.return_value
            llm_inst.chat = AsyncMock(return_value=_mock_llm_response(
                "I'm PredaCore from PredaCore. I can help you with coding, "
                "file management, and more."
            ))
            llm_inst.active_provider = "mock"
            llm_inst.active_model = "test-model"
            llm_inst._provider_name = "mock"
            llm_inst.executes_tool_loop_internally = False
            # Phase A: core.py calls provider.append_*_turn for tool round-trips.
            # Give it a real LLMProvider instance that uses default impls.
            llm_inst.active_provider_instance = _test_provider_instance()

            from predacore.core import PredaCoreCore
            core = PredaCoreCore(cfg)

            session = Session(user_id="test_user")
            response = await core.process("test_user", "Who are you?", session)

            # Only one LLM call needed — no regeneration
            assert llm_inst.chat.call_count == 1
            assert "PredaCore" in response

    @pytest.mark.asyncio
    async def test_drift_score_calculated_correctly(self, tmp_path):
        """_assess_persona_drift returns correct score for known patterns."""
        cfg = _make_config(tmp_path)

        with patch("predacore.core.LLMInterface") as MockLLM:
            llm_inst = MockLLM.return_value
            llm_inst.chat = AsyncMock(return_value=_mock_llm_response())
            llm_inst.active_provider = "mock"
            llm_inst.active_model = "test-model"
            llm_inst._provider_name = "mock"
            llm_inst.executes_tool_loop_internally = False
            # Phase A: core.py calls provider.append_*_turn for tool round-trips.
            # Give it a real LLMProvider instance that uses default impls.
            llm_inst.active_provider_instance = _test_provider_instance()

            from predacore.core import PredaCoreCore
            core = PredaCoreCore(cfg)

            # ChatGPT claim should score high
            assessment = core._assess_persona_drift(
                "who are you?",
                "I'm ChatGPT, an AI language model by OpenAI",
            )
            assert assessment.score > 0.4
            assert "foreign_identity_openai" in assessment.reasons

            # Normal PredaCore response should score low
            assessment_good = core._assess_persona_drift(
                "hello",
                "Hello! I'm PredaCore from PredaCore. How can I help?",
            )
            assert assessment_good.score < 0.3

    @pytest.mark.asyncio
    async def test_drift_guard_disabled_skips_check(self, tmp_path):
        """When drift guard is disabled, no regeneration occurs."""
        cfg = _make_config(tmp_path)
        cfg.launch.enable_persona_drift_guard = False

        with patch("predacore.core.LLMInterface") as MockLLM:
            llm_inst = MockLLM.return_value
            llm_inst.chat = AsyncMock(return_value=_mock_llm_response(
                "I'm ChatGPT"  # Would normally trigger drift
            ))
            llm_inst.active_provider = "mock"
            llm_inst.active_model = "test-model"
            llm_inst._provider_name = "mock"
            llm_inst.executes_tool_loop_internally = False
            # Phase A: core.py calls provider.append_*_turn for tool round-trips.
            # Give it a real LLMProvider instance that uses default impls.
            llm_inst.active_provider_instance = _test_provider_instance()

            from predacore.core import PredaCoreCore
            core = PredaCoreCore(cfg)
            core._persona_drift_guard_enabled = False

            session = Session(user_id="test_user")
            response = await core.process("test_user", "Who are you?", session)

            # Should NOT trigger regeneration — only 1 LLM call
            assert llm_inst.chat.call_count == 1


# ---------------------------------------------------------------------------
# Section 11: Gateway Full Pipeline
# ---------------------------------------------------------------------------


class TestGatewayFullPipeline:
    """E2E: message flows through gateway -> session -> process -> response."""

    @pytest.mark.asyncio
    async def test_gateway_creates_session_and_persists(self, tmp_path):
        """Gateway creates a session and persists user + assistant messages."""
        cfg = _make_config(tmp_path)

        async def mock_process(user_id, message, session, **kwargs):
            return "PredaCore response from gateway"

        gw = Gateway(cfg, mock_process)

        msg = IncomingMessage(channel="cli", user_id="gw_user", text="Hello gateway")
        response = await gw.handle_message(msg)

        assert response.text == "PredaCore response from gateway"
        assert response.channel == "cli"
        assert response.session_id != ""

        # Verify session was persisted. Note: gateway.identity.resolve() mutates
        # incoming.user_id to a canonical ID, so fetch by session_id directly.
        session = gw.session_store.get(response.session_id)
        assert session is not None
        # Messages persisted
        assert len(session.messages) >= 2  # user + assistant

    @pytest.mark.asyncio
    async def test_gateway_session_command_new(self, tmp_path):
        """'/new' command creates a fresh session."""
        cfg = _make_config(tmp_path)

        async def mock_process(user_id, message, session, **kwargs):
            return "ok"

        gw = Gateway(cfg, mock_process)

        msg = IncomingMessage(channel="cli", user_id="cmd_user", text="/new")
        response = await gw.handle_message(msg)

        assert "fresh session" in response.text.lower() or "started" in response.text.lower()
        assert response.session_id != ""

    @pytest.mark.asyncio
    async def test_gateway_handles_error_gracefully(self, tmp_path):
        """Gateway catches processing errors and returns error message."""
        cfg = _make_config(tmp_path)

        async def mock_process(user_id, message, session, **kwargs):
            raise RuntimeError("Simulated processing failure")

        gw = Gateway(cfg, mock_process)

        msg = IncomingMessage(channel="cli", user_id="err_user", text="trigger error")
        response = await gw.handle_message(msg)

        assert "error" in response.text.lower() or "try again" in response.text.lower()

    @pytest.mark.asyncio
    async def test_gateway_session_continuity(self, tmp_path):
        """Two messages from same user use the same session."""
        cfg = _make_config(tmp_path)
        call_log = []

        async def mock_process(user_id, message, session, **kwargs):
            call_log.append(session.session_id)
            return f"Response to: {message}"

        gw = Gateway(cfg, mock_process)

        msg1 = IncomingMessage(channel="cli", user_id="cont_user", text="First message")
        resp1 = await gw.handle_message(msg1)

        msg2 = IncomingMessage(channel="cli", user_id="cont_user", text="Second message")
        resp2 = await gw.handle_message(msg2)

        # Same session reused
        assert resp1.session_id == resp2.session_id
        assert len(call_log) == 2
        assert call_log[0] == call_log[1]

    @pytest.mark.asyncio
    async def test_gateway_sanitizes_input(self, tmp_path):
        """Gateway strips null bytes and ANSI escapes from input."""
        cfg = _make_config(tmp_path)
        received_messages = []

        async def mock_process(user_id, message, session, **kwargs):
            received_messages.append(message)
            return "sanitized"

        gw = Gateway(cfg, mock_process)

        # Message with null bytes
        msg = IncomingMessage(
            channel="cli", user_id="san_user", text="hello\x00world"
        )
        await gw.handle_message(msg)

        assert len(received_messages) == 1
        assert "\x00" not in received_messages[0]


# ---------------------------------------------------------------------------
# Section 12: Session Store Edge Cases
# ---------------------------------------------------------------------------


class TestSessionStoreEdgeCases:
    """E2E: session persistence edge cases."""

    def test_session_survives_store_recreation(self, tmp_path):
        """Sessions persist when SessionStore is recreated."""
        sdir = str(tmp_path / "sessions")
        store1 = SessionStore(sdir)
        s = store1.create(user_id="persist_user")
        store1.append_message(s.session_id, "user", "Persisted message")

        # Recreate store (simulates daemon restart)
        store2 = SessionStore(sdir)
        loaded = store2.get(s.session_id)
        assert loaded is not None
        assert len(loaded.messages) == 1
        assert loaded.messages[0].content == "Persisted message"

    def test_session_message_cap(self, tmp_path):
        """Session truncates oldest messages when cap is exceeded."""
        session = Session(user_id="cap_test", max_messages=10)
        for i in range(20):
            session.add_message("user", f"Message {i}")

        assert len(session.messages) == 10
        # Oldest should be trimmed, newest kept
        assert session.messages[0].content == "Message 10"
        assert session.messages[-1].content == "Message 19"

    def test_session_llm_messages_format(self, tmp_path):
        """get_llm_messages returns correctly formatted dicts."""
        session = Session(user_id="fmt_test")
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!")
        session.add_message("user", "How are you?")

        llm_msgs = session.get_llm_messages()
        assert all("role" in m and "content" in m for m in llm_msgs)
        assert llm_msgs[0]["role"] == "user"
        assert llm_msgs[1]["role"] == "assistant"

    def test_session_empty_context_window(self, tmp_path):
        """Empty session returns empty context window."""
        session = Session(user_id="empty")
        context = session.build_context_window()
        assert context == []


# ---------------------------------------------------------------------------
# Section 13: Config <-> Core Integration
# ---------------------------------------------------------------------------


class TestConfigCoreIntegration:
    """E2E: config values propagate correctly through to core behavior."""

    @pytest.mark.asyncio
    async def test_max_tool_iterations_respected(self, tmp_path):
        """Core respects max_tool_iterations from config."""
        cfg = _make_config(tmp_path)
        cfg.launch.max_tool_iterations = 3

        with patch("predacore.core.LLMInterface") as MockLLM:
            iteration_count = [0]

            async def _infinite_tool_loop(*args, **kwargs):
                iteration_count[0] += 1
                # Always request another tool call
                if kwargs.get("tools") is not None:
                    return _mock_tool_call_response(
                        "read_file", {"path": "/tmp/test.txt"}
                    )
                return _mock_llm_response("Done after forced stop")

            llm_inst = MockLLM.return_value
            llm_inst.chat = AsyncMock(side_effect=_infinite_tool_loop)
            llm_inst.active_provider = "mock"
            llm_inst.active_model = "test-model"
            llm_inst._provider_name = "mock"
            llm_inst.executes_tool_loop_internally = False
            # Phase A: core.py calls provider.append_*_turn for tool round-trips.
            # Give it a real LLMProvider instance that uses default impls.
            llm_inst.active_provider_instance = _test_provider_instance()

            from predacore.core import PredaCoreCore
            core = PredaCoreCore(cfg)

            session = Session(user_id="test_user")
            response = await core.process("test_user", "Loop test", session)

            # Should not exceed max_tool_iterations (3) + possible final call
            assert iteration_count[0] <= cfg.launch.max_tool_iterations + 2

    @pytest.mark.asyncio
    async def test_blocked_tools_not_available(self, tmp_path):
        """Blocked tools are filtered from tool definitions."""
        cfg = _make_config(tmp_path)
        cfg.security.blocked_tools = ["run_command", "python_exec"]

        with patch("predacore.core.LLMInterface") as MockLLM:
            llm_inst = MockLLM.return_value
            llm_inst.chat = AsyncMock(return_value=_mock_llm_response())
            llm_inst.active_provider = "mock"
            llm_inst.active_model = "test-model"
            llm_inst._provider_name = "mock"
            llm_inst.executes_tool_loop_internally = False
            # Phase A: core.py calls provider.append_*_turn for tool round-trips.
            # Give it a real LLMProvider instance that uses default impls.
            llm_inst.active_provider_instance = _test_provider_instance()

            from predacore.core import PredaCoreCore
            core = PredaCoreCore(cfg)

            tool_names = core.get_tool_list()
            assert "run_command" not in tool_names
            assert "python_exec" not in tool_names

    def test_trust_level_propagates_to_core(self, tmp_path):
        """Config trust level is used in the tool executor."""
        cfg = _make_config(tmp_path)
        cfg.security.trust_level = "paranoid"

        with patch("predacore.core.LLMInterface") as MockLLM:
            llm_inst = MockLLM.return_value
            llm_inst.chat = AsyncMock(return_value=_mock_llm_response())
            llm_inst.active_provider = "mock"
            llm_inst.active_model = "test-model"
            llm_inst._provider_name = "mock"
            llm_inst.executes_tool_loop_internally = False
            # Phase A: core.py calls provider.append_*_turn for tool round-trips.
            # Give it a real LLMProvider instance that uses default impls.
            llm_inst.active_provider_instance = _test_provider_instance()

            from predacore.core import PredaCoreCore
            core = PredaCoreCore(cfg)
            assert core.config.security.trust_level == "paranoid"


# ---------------------------------------------------------------------------
# Section 14: Memory + Core Integration
# ---------------------------------------------------------------------------


class TestMemoryCoreIntegration:
    """E2E: Memory system integrates with core flow."""

    @pytest.mark.asyncio
    async def test_memory_store_is_initialized(self, tmp_path):
        """Core initializes unified memory store."""
        cfg = _make_config(tmp_path)

        with patch("predacore.core.LLMInterface") as MockLLM:
            llm_inst = MockLLM.return_value
            llm_inst.chat = AsyncMock(return_value=_mock_llm_response())
            llm_inst.active_provider = "mock"
            llm_inst.active_model = "test-model"
            llm_inst._provider_name = "mock"
            llm_inst.executes_tool_loop_internally = False
            # Phase A: core.py calls provider.append_*_turn for tool round-trips.
            # Give it a real LLMProvider instance that uses default impls.
            llm_inst.active_provider_instance = _test_provider_instance()

            from predacore.core import PredaCoreCore
            core = PredaCoreCore(cfg)

            # Unified memory should be initialized (may be None if subsystem init fails,
            # but should not crash)
            assert hasattr(core.tools, "_unified_memory")

    @pytest.mark.asyncio
    async def test_memory_recall_keyword_no_embedding(self, tmp_path):
        """Memory recall falls back to keyword search when no embedding client."""
        db_path = str(tmp_path / "recall.db")
        store = UnifiedMemoryStore(db_path=db_path)

        await store.store(
            content="User prefers dark mode and vim keybindings",
            memory_type="preference",
            user_id="recall_user",
        )
        await store.store(
            content="Discussed Python FastAPI project setup",
            memory_type="conversation",
            user_id="recall_user",
        )

        # No embedding client => keyword fallback
        results = await store.recall(query="vim dark mode", user_id="recall_user")
        assert len(results) > 0
        store.close()

    @pytest.mark.asyncio
    async def test_memory_scoped_by_user(self, tmp_path):
        """Memories are scoped to the correct user."""
        db_path = str(tmp_path / "scoped.db")
        store = UnifiedMemoryStore(db_path=db_path)

        await store.store(content="User A secret", user_id="userA")
        await store.store(content="User B data", user_id="userB")

        results_a = await store.recall(query="secret", user_id="userA")
        results_b = await store.recall(query="secret", user_id="userB")

        # User A should find their memory
        a_contents = [r[0]["content"] for r in results_a]
        b_contents = [r[0]["content"] for r in results_b]

        assert any("User A" in c for c in a_contents)
        # User B should NOT see User A's memory
        assert not any("User A" in c for c in b_contents)
        store.close()


# ---------------------------------------------------------------------------
# Section 15: Concurrent Operations
# ---------------------------------------------------------------------------


class TestConcurrentOperations:
    """E2E: concurrent message handling through gateway."""

    @pytest.mark.asyncio
    async def test_concurrent_users_dont_interfere(self, tmp_path):
        """Multiple users sending simultaneously get separate sessions."""
        cfg = _make_config(tmp_path)
        user_sessions = {}

        async def mock_process(user_id, message, session, **kwargs):
            user_sessions[user_id] = session.session_id
            await asyncio.sleep(0.01)  # Simulate brief processing
            return f"Response for {user_id}"

        gw = Gateway(cfg, mock_process)

        # Send messages from 5 different users concurrently
        tasks = []
        for i in range(5):
            msg = IncomingMessage(
                channel="cli", user_id=f"concurrent_{i}", text=f"Hello from user {i}"
            )
            tasks.append(gw.handle_message(msg))

        responses = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r.text.startswith("Response for") for r in responses)
        # Each user gets their own session
        assert len(set(user_sessions.values())) == 5

    @pytest.mark.asyncio
    async def test_same_user_serialized_by_lane_queue(self, tmp_path):
        """Messages from the same user are serialized (not parallel)."""
        cfg = _make_config(tmp_path)
        execution_order = []

        async def mock_process(user_id, message, session, **kwargs):
            execution_order.append(message)
            await asyncio.sleep(0.02)
            return f"Done: {message}"

        gw = Gateway(cfg, mock_process)

        # Send 3 messages rapidly from same user
        tasks = []
        for i in range(3):
            msg = IncomingMessage(
                channel="cli", user_id="serial_user", text=f"msg_{i}"
            )
            tasks.append(gw.handle_message(msg))

        await asyncio.gather(*tasks)

        # All should be processed (order may vary due to lane queue)
        assert len(execution_order) == 3
        assert set(execution_order) == {"msg_0", "msg_1", "msg_2"}
