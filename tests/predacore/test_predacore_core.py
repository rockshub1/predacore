"""
🧪 PredaCore Test Suite — Identity System, Config, Sessions, Core

Tests the entire PredaCore stack from soul to tools:
  1. Identity file loading (SOUL.md, USER.md, MEMORY.md)
  2. System prompt assembly (4-layer architecture)
  3. Config system (defaults, YAML override, ENV override)
  4. Session persistence (JSONL, metadata, archival)
  5. Lane Queue (serial execution, timeouts)
  6. Trust policies (yolo, normal, paranoid)
  7. Tool definitions and executor
"""
from __future__ import annotations

import os
import textwrap
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# ── Imports from PredaCore ──────────────────────────────────────────────
from src.predacore.config import (
    ChannelConfig,
    PredaCoreConfig,
    LaunchProfileConfig,
    LLMConfig,
    MemoryConfig,
    OpenClawBridgeConfig,
    SecurityConfig,
    load_config,
)
from src.predacore.core import (
    BUILTIN_TOOLS,
    OPENCLAW_BRIDGE_TOOLS,
    PredaCoreCore,
    ToolExecutor,
    _get_system_prompt,
)
from src.predacore.identity.engine import get_identity_engine
from src.predacore.prompts import _load_identity_file
from src.predacore.identity.engine import reset_identity_engine
from src.predacore.tools.handlers._context import ToolContext
from src.predacore.tools.handlers.identity import handle_identity_read, handle_identity_update
from src.predacore.tools.registry import TRUST_POLICIES, build_builtin_registry
from src.predacore.services.lane_queue import LaneQueue
from src.predacore.sessions import Session, SessionStore

# ── Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IDENTITY_DIR = PROJECT_ROOT / "src" / "predacore" / "identity"


# ══════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def tmp_home(tmp_path):
    """Create a temporary ~/.prometheus-like directory."""
    home = tmp_path / ".prometheus"
    home.mkdir()
    (home / "sessions").mkdir()
    (home / "skills").mkdir()
    (home / "logs").mkdir()
    (home / "memory").mkdir()
    return home


@pytest.fixture
def config(tmp_home):
    """Create a PredaCoreConfig pointing to temp directories."""
    # Reset singleton so each test gets a fresh IdentityEngine
    reset_identity_engine()
    cfg = PredaCoreConfig(
        name="PredaCore-Test",
        home_dir=str(tmp_home),
        sessions_dir=str(tmp_home / "sessions"),
        skills_dir=str(tmp_home / "skills"),
        logs_dir=str(tmp_home / "logs"),
        llm=LLMConfig(provider="gemini-cli"),
        security=SecurityConfig(trust_level="normal"),
        memory=MemoryConfig(persistence_dir=str(tmp_home / "memory")),
    )
    yield cfg
    reset_identity_engine()


@pytest.fixture
def identity_dir(tmp_home):
    """Create an agent identity directory with test files."""
    idir = tmp_home / "agents" / "default"
    idir.mkdir(parents=True)
    return idir


# ══════════════════════════════════════════════════════════════════════
# TEST GROUP 1: Identity File Loading
# ══════════════════════════════════════════════════════════════════════


class TestIdentityFileLoading:
    """Tests for _load_identity_file() — the soul reader."""

    def test_loads_builtin_soul_seed(self, config):
        """SOUL_SEED.md should load from the built-in identity directory."""
        soul = _load_identity_file("SOUL_SEED.md", config)
        assert len(soul) > 0, "SOUL_SEED.md should not be empty"

    def test_loads_user_workspace_files(self, config, identity_dir):
        """USER.md and MEMORY.md load from user workspace (~/.predacore/agents/{agent}/)."""
        (identity_dir / "USER.md").write_text("# User\nName: TestUser")
        user = _load_identity_file("USER.md", config)
        assert "TestUser" in user

        (identity_dir / "MEMORY.md").write_text("# Memory\nSome memories")
        memory = _load_identity_file("MEMORY.md", config)
        assert "memories" in memory

    def test_user_override_takes_priority(self, config, identity_dir):
        """User's ~/.predacore/agents/{agent}/ files should override built-in."""
        custom_seed = "# Custom Seed\nI am a test agent."
        (identity_dir / "SOUL_SEED.md").write_text(custom_seed)

        seed = _load_identity_file("SOUL_SEED.md", config)
        assert seed == custom_seed, "User override should take priority"

    def test_missing_file_returns_empty(self, config):
        """Non-existent identity file should return empty string."""
        result = _load_identity_file("NONEXISTENT.md", config)
        assert result == "", "Missing file should return empty string"

    def test_empty_file_returns_empty(self, config, identity_dir):
        """Empty identity file should return empty string."""
        (identity_dir / "EMPTY.md").write_text("")
        result = _load_identity_file("EMPTY.md", config)
        assert result == "", "Empty file should return empty string"

    def test_unicode_identity_file(self, config, identity_dir):
        """Identity files with unicode should load correctly."""
        unicode_content = "# 🔥 Soul\nI am PredaCore — éàü 日本語 中文"
        (identity_dir / "SOUL.md").write_text(unicode_content, encoding="utf-8")

        soul = _load_identity_file("SOUL.md", config)
        assert "🔥" in soul
        assert "日本語" in soul

    def test_whitespace_trimmed(self, config, identity_dir):
        """Identity file content should be stripped of leading/trailing whitespace."""
        padded = "\n\n  # Soul  \n\n"
        (identity_dir / "TRIM.md").write_text(padded)
        result = _load_identity_file("TRIM.md", config)
        assert result == "# Soul"


# ══════════════════════════════════════════════════════════════════════
# TEST GROUP 2: System Prompt Assembly
# ══════════════════════════════════════════════════════════════════════


class TestSystemPromptAssembly:
    """Tests for _get_system_prompt() — the 4-layer prompt builder."""

    def test_prompt_contains_soul_seed(self, config):
        """System prompt should contain SOUL_SEED.md content."""
        prompt = _get_system_prompt(config)
        assert "SOUL_SEED" in prompt or "Cannot Change" in prompt
        assert len(prompt) > 500, "Full prompt should be substantial"

    def test_prompt_contains_runtime_context(self, config):
        """System prompt should include dynamic runtime context."""
        prompt = _get_system_prompt(config)
        assert "Runtime Context" in prompt
        assert config.launch.profile in prompt
        assert "NORMAL" in prompt

    def test_prompt_contains_trust_policy(self, config):
        """System prompt should include trust policy description."""
        prompt = _get_system_prompt(config)
        policy = TRUST_POLICIES["normal"]
        assert policy["description"] in prompt

    def test_prompt_with_yolo_trust(self, config):
        """YOLO trust level should appear in the prompt."""
        config.security.trust_level = "yolo"
        prompt = _get_system_prompt(config)
        assert "YOLO" in prompt

    def test_prompt_with_paranoid_trust(self, config):
        """Paranoid trust level should appear in the prompt."""
        config.security.trust_level = "paranoid"
        prompt = _get_system_prompt(config)
        assert "PARANOID" in prompt

    def test_custom_user_profile_in_prompt(self, config, identity_dir):
        """Custom USER.md should appear in the system prompt when bootstrapped."""
        # Must be bootstrapped (IDENTITY.md exists) for USER.md to be loaded
        (identity_dir / "IDENTITY.md").write_text("# Identity\nName: TestAgent")
        custom_user = "# Custom User\n- Name: TestBot\n- Role: Tester"
        (identity_dir / "USER.md").write_text(custom_user)

        prompt = _get_system_prompt(config)
        assert "TestBot" in prompt
        assert "Tester" in prompt

    def test_prompt_includes_date(self, config):
        """Runtime context should include the current date."""
        prompt = _get_system_prompt(config)
        current_year = time.strftime("%Y")
        assert current_year in prompt

    def test_prompt_with_custom_soul_override(self, config, identity_dir):
        """Custom SOUL.md should appear in prompt when bootstrapped."""
        (identity_dir / "IDENTITY.md").write_text("# Identity\nName: TestAgent")
        (identity_dir / "SOUL.md").write_text("# I am FRIDAY, a test AI")
        prompt = _get_system_prompt(config)
        assert "FRIDAY" in prompt

    def test_prompt_with_custom_memory(self, config, identity_dir):
        """Custom MEMORY.md should appear in the prompt."""
        # MEMORY.md is loaded via _load_identity_file in prompts.py (both paths)
        (identity_dir / "MEMORY.md").write_text("# Memory\n- User likes Python 3.12")
        prompt = _get_system_prompt(config)
        assert "Python 3.12" in prompt

    def test_self_evolving_prompt_does_not_duplicate_memory(self, config, identity_dir):
        """Self-evolving path should inject MEMORY.md exactly once."""
        (identity_dir / "IDENTITY.md").write_text("# Identity\nName: TestAgent")
        unique_line = "UNIQUE_MEMORY_LINE_20260403"
        (identity_dir / "MEMORY.md").write_text(f"# Memory\n- {unique_line}")
        prompt = _get_system_prompt(config)
        assert prompt.count(unique_line) == 1

    def test_identity_engine_prefers_workspace_seed_and_bootstrap(self, config, identity_dir):
        """Per-agent SOUL_SEED.md and BOOTSTRAP.md overrides should be real, not decorative."""
        (identity_dir / "SOUL_SEED.md").write_text("# Workspace Seed\nOverride seed")
        (identity_dir / "BOOTSTRAP.md").write_text("# Workspace Bootstrap\nOverride bootstrap")
        engine = get_identity_engine(config)
        assert engine.load_seed() == "# Workspace Seed\nOverride seed"
        assert engine.load_bootstrap_prompt() == "# Workspace Bootstrap\nOverride bootstrap"

    def test_bootstrap_complete_requires_full_identity_surface(self, config, identity_dir):
        """Bootstrap should fail until the full required identity surface exists."""
        (identity_dir / "IDENTITY.md").write_text("# Identity\nName: TestAgent")
        (identity_dir / "SOUL.md").write_text("# Soul\nVoice")
        (identity_dir / "USER.md").write_text("# User\nShubh")

        engine = get_identity_engine(config)
        result = engine.mark_bootstrap_complete()

        assert result["status"] == "error"
        assert result["missing_files"] == [
            "TOOLS.md",
            "MEMORY.md",
            "HEARTBEAT.md",
            "REFLECTION.md",
        ]

    def test_bootstrap_complete_archives_bootstrap_prompt(self, config, identity_dir):
        """bootstrap_complete should archive BOOTSTRAP.md after full bootstrap succeeds."""
        for filename, content in {
            "IDENTITY.md": "# Identity\nName: TestAgent",
            "SOUL.md": "# Soul\nVoice",
            "USER.md": "# User\nShubh",
            "TOOLS.md": "# Tools\nVerified",
            "MEMORY.md": "# Memory\nDurable",
            "HEARTBEAT.md": "# Heartbeat\nRules",
            "REFLECTION.md": "# Reflection\nRules",
            "BOOTSTRAP.md": "# Bootstrap\nHello",
        }.items():
            (identity_dir / filename).write_text(content)

        engine = get_identity_engine(config)
        result = engine.mark_bootstrap_complete()

        assert result["status"] == "ok"
        assert result["archived_bootstrap"]
        assert not (identity_dir / "BOOTSTRAP.md").exists()
        archived_files = list(
            (identity_dir / "_archive").glob("BOOTSTRAP_bootstrap_complete_*.md")
        )
        assert len(archived_files) == 1

    def test_prompt_includes_self_meta_prompt_when_enabled(self, config, identity_dir):
        """Self-evolution mode should include self meta prompt text.

        prompts._load_self_meta_prompt() searches (in order):
          1. $PREDACORE_META_PROMPT_FILE
          2. ~/.predacore/agents/{agent}/META_PROMPT.md
          3. docs/predacore_self_meta_prompt.md (bundled default)

        Drop a sentinel META_PROMPT.md in the temp agent dir and verify
        the prompt builder injects it when self-evolution is enabled.
        """
        sentinel = "## PredaCore Self Meta Prompt\nSentinel meta guidance."
        (identity_dir / "META_PROMPT.md").write_text(sentinel)

        config.launch.enable_self_evolution = True
        prompt = _get_system_prompt(config)
        assert "PredaCore Self Meta Prompt" in prompt


# ══════════════════════════════════════════════════════════════════════
# TEST GROUP 3: Config System
# ══════════════════════════════════════════════════════════════════════


class TestConfigSystem:
    """Tests for PredaCoreConfig — defaults, YAML, ENV overrides."""

    def test_default_config(self):
        """Default config should have sensible values."""
        cfg = PredaCoreConfig()
        assert cfg.name == "PredaCore"
        assert cfg.llm.provider == "gemini-cli"
        assert cfg.security.trust_level == "normal"
        assert cfg.launch.profile == "enterprise"

    def test_home_dir_defaults(self):
        """Home dir should default to ~/.prometheus."""
        cfg = PredaCoreConfig()
        assert ".prometheus" in cfg.home_dir

    def test_sub_dirs_computed(self):
        """Session, skills, logs dirs should be computed from home_dir."""
        cfg = PredaCoreConfig(home_dir="/tmp/test_prom")
        assert cfg.sessions_dir == "/tmp/test_prom/sessions"
        assert cfg.skills_dir == "/tmp/test_prom/skills"
        assert cfg.logs_dir == "/tmp/test_prom/logs"

    def test_custom_name(self):
        """Config should accept custom agent name."""
        cfg = PredaCoreConfig(name="FRIDAY")
        assert cfg.name == "FRIDAY"

    def test_llm_config_defaults(self):
        """LLM config should have sensible defaults."""
        llm = LLMConfig()
        assert llm.provider == "gemini-cli"
        assert llm.temperature >= 0
        assert llm.max_tokens > 0

    def test_security_config(self):
        """Security config trust levels should work."""
        for level in ["yolo", "normal", "paranoid"]:
            sec = SecurityConfig(trust_level=level)
            assert sec.trust_level == level

    def test_memory_config_defaults(self):
        """Memory config should have persistence_dir."""
        mem = MemoryConfig()
        assert hasattr(mem, "persistence_dir")

    def test_channel_config(self):
        """Channel config should have CLI enabled by default."""
        ch = ChannelConfig()
        assert "cli" in ch.enabled

    def test_launch_profile_config_defaults(self):
        """Launch profile defaults should be deterministic."""
        lp = LaunchProfileConfig()
        assert lp.profile == "enterprise"
        assert lp.approvals_required is True
        assert lp.default_code_network is False
        # 2-mode simplification: resource limits maxed on both profiles.
        assert lp.max_tool_iterations == 1000
        assert lp.enable_persona_drift_guard is True
        assert lp.persona_drift_threshold == 0.32
        assert lp.persona_drift_max_regens == 5

    def test_load_config_beast_profile_from_env(self, tmp_path, monkeypatch):
        """Beast profile should apply preset defaults and sync runtime env."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("name: PredaCore\n")

        monkeypatch.setenv("PREDACORE_PROFILE", "beast")
        cfg = load_config(str(cfg_file))

        assert cfg.launch.profile == "beast"
        assert cfg.security.trust_level == "yolo"
        assert cfg.launch.approvals_required is False
        assert cfg.launch.egm_mode == "off"
        assert cfg.launch.max_tool_iterations == 1000
        assert cfg.launch.persona_drift_threshold == 0.60
        assert os.environ.get("APPROVALS_REQUIRED") == "0"
        assert os.environ.get("DEFAULT_CODE_NETWORK") == "1"

    def test_load_config_profile_override_takes_precedence(self, tmp_path, monkeypatch):
        """Explicit profile override should beat env-selected profile."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("name: PredaCore\n")

        monkeypatch.setenv("PREDACORE_PROFILE", "beast")
        cfg = load_config(str(cfg_file), profile_override="enterprise")

        assert cfg.launch.profile == "enterprise"
        assert cfg.security.trust_level == "normal"
        assert cfg.launch.approvals_required is True
        assert os.environ.get("APPROVALS_REQUIRED") == "1"

    def test_load_config_profile_presets_allow_yaml_overrides(self, tmp_path):
        """YAML values should still override profile defaults."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            textwrap.dedent(
                """
                launch:
                  profile: beast
                security:
                  max_concurrent_tasks: 42
                channels:
                  enabled: [cli, webchat, telegram]
                """
            ).strip()
            + "\n"
        )

        cfg = load_config(str(cfg_file))
        assert cfg.launch.profile == "beast"
        assert cfg.security.max_concurrent_tasks == 42
        assert "telegram" in cfg.channels.enabled

    def test_version_string(self):
        """Config should have a version."""
        cfg = PredaCoreConfig()
        assert cfg.version == "0.1.0"

    def test_load_config_openclaw_env_overrides(self, tmp_path, monkeypatch):
        """OpenClaw bridge settings should be configurable from env vars."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("name: PredaCore\n")

        monkeypatch.setenv("PREDACORE_PROFILE", "beast")
        monkeypatch.setenv("OPENCLAW_BRIDGE_URL", "https://bridge.example.com")
        monkeypatch.setenv("OPENCLAW_BRIDGE_API_KEY", "test-token")
        monkeypatch.setenv("OPENCLAW_BRIDGE_MODEL", "openclaw:main")
        monkeypatch.setenv("OPENCLAW_BRIDGE_AGENT_ID", "main")
        monkeypatch.setenv("OPENCLAW_BRIDGE_TIMEOUT", "222")
        monkeypatch.setenv("OPENCLAW_BRIDGE_VERIFY_TLS", "0")
        monkeypatch.setenv("OPENCLAW_BRIDGE_STATUS_PATH", "/v2/jobs/{task_id}")
        monkeypatch.setenv("OPENCLAW_BRIDGE_MAX_RETRIES", "4")
        monkeypatch.setenv("OPENCLAW_BRIDGE_RETRY_BACKOFF", "1.25")
        monkeypatch.setenv("OPENCLAW_BRIDGE_POLL_INTERVAL", "0.5")
        monkeypatch.setenv("OPENCLAW_BRIDGE_MAX_POLL_SECONDS", "90")
        monkeypatch.setenv("OPENCLAW_SKILLS_DIR", "/tmp/openclaw/skills")
        monkeypatch.setenv("OPENCLAW_AUTO_IMPORT_SKILLS", "0")

        cfg = load_config(str(cfg_file))
        assert cfg.openclaw.base_url == "https://bridge.example.com"
        assert cfg.openclaw.api_key == "test-token"
        assert cfg.openclaw.model == "openclaw:main"
        assert cfg.openclaw.agent_id == "main"
        assert cfg.openclaw.timeout_seconds == 222
        assert cfg.openclaw.verify_tls is False
        assert cfg.openclaw.status_path == "/v2/jobs/{task_id}"
        assert cfg.openclaw.max_retries == 4
        assert cfg.openclaw.retry_backoff_seconds == 1.25
        assert cfg.openclaw.poll_interval_seconds == 0.5
        assert cfg.openclaw.max_poll_seconds == 90
        assert cfg.openclaw.skills_dir == "/tmp/openclaw/skills"
        assert cfg.openclaw.auto_import_skills is False

    def test_load_config_persona_drift_env_overrides(self, tmp_path, monkeypatch):
        """Persona drift guard should be configurable from env vars."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("name: PredaCore\n")

        monkeypatch.setenv("PREDACORE_ENABLE_PERSONA_DRIFT_GUARD", "1")
        monkeypatch.setenv("PREDACORE_PERSONA_DRIFT_THRESHOLD", "0.35")
        monkeypatch.setenv("PREDACORE_PERSONA_DRIFT_MAX_REGENS", "3")

        cfg = load_config(str(cfg_file))
        assert cfg.launch.enable_persona_drift_guard is True
        assert cfg.launch.persona_drift_threshold == 0.35
        assert cfg.launch.persona_drift_max_regens == 3


# ══════════════════════════════════════════════════════════════════════
# TEST GROUP 4: Session Persistence
# ══════════════════════════════════════════════════════════════════════


class TestSessionPersistence:
    """Tests for Session and SessionStore — JSONL persistence."""

    def test_session_creation(self):
        """Session should initialize with sane defaults."""
        s = Session(session_id="test-123", user_id="user-1")
        assert s.session_id == "test-123"
        assert s.user_id == "user-1"
        assert s.message_count == 0

    def test_session_add_message(self):
        """Adding messages should increase the count."""
        s = Session()
        s.add_message("user", "Hello PredaCore")
        s.add_message("assistant", "Hello! How can I help?")
        assert s.message_count == 2

    def test_session_store_create(self, tmp_home):
        """SessionStore.create should return a Session."""
        store = SessionStore(str(tmp_home / "sessions"))
        session = store.create("user-1")
        assert session is not None
        assert session.user_id == "user-1"
        assert len(session.session_id) > 0

    def test_session_store_get(self, tmp_home):
        """SessionStore.get should retrieve a saved session."""
        store = SessionStore(str(tmp_home / "sessions"))
        session = store.create("user-1")
        sid = session.session_id

        # Append a message
        store.append_message(sid, "user", "Test message")

        # Retrieve
        loaded = store.get(sid)
        assert loaded is not None
        assert loaded.session_id == sid

    def test_session_store_list(self, tmp_home):
        """SessionStore.list_sessions should list all sessions."""
        store = SessionStore(str(tmp_home / "sessions"))
        store.create("user-1")
        store.create("user-1")

        sessions = store.list_sessions()
        assert len(sessions) >= 2

    def test_session_llm_format(self):
        """Messages should convert to LLM format correctly."""
        s = Session()
        s.add_message("user", "Hello")
        s.add_message("assistant", "Hi there!")

        llm_msgs = s.get_llm_messages()
        assert len(llm_msgs) == 2
        assert llm_msgs[0]["role"] == "user"
        assert llm_msgs[0]["content"] == "Hello"

    def test_session_context_summary(self):
        """Session context summary should return a string."""
        s = Session()
        s.add_message("user", "What is Python?")
        summary = s.get_context_summary()
        assert isinstance(summary, str)

    def test_session_build_context_window_compacts_older_history(self):
        """Long sessions should preserve recent turns and summarize older ones."""
        s = Session()
        for idx in range(20):
            s.add_message(
                "user",
                f"User topic {idx}: " + ("details " * 80),
            )
            s.add_message(
                "assistant",
                f"Assistant reply {idx}: " + ("analysis " * 80),
            )

        packed = s.build_context_window(
            max_total_tokens=1_000,
            keep_recent_messages=6,
            summary_max_tokens=180,
        )

        assert packed[0]["role"] == "system"
        assert "Session summary" in packed[0]["content"]
        assert any("User topic 19" in msg["content"] for msg in packed)
        assert len(packed) < len(s.get_llm_messages())


# ══════════════════════════════════════════════════════════════════════
# TEST GROUP 5: Lane Queue
# ══════════════════════════════════════════════════════════════════════


class TestLaneQueue:
    """Tests for LaneQueue — serial task execution."""

    def test_queue_creation(self):
        """LaneQueue should initialize."""
        q = LaneQueue()
        assert q is not None

    @pytest.mark.asyncio
    async def test_serial_execution(self):
        """Tasks in the same lane should execute serially."""
        q = LaneQueue()
        results = []

        async def task(value):
            results.append(value)
            return value

        await q.submit("lane-1", task, 1)
        await q.submit("lane-1", task, 2)
        await q.submit("lane-1", task, 3)

        assert results == [1, 2, 3], "Tasks should execute in order"
        await q.shutdown()

    @pytest.mark.asyncio
    async def test_different_lanes_independent(self):
        """Tasks in different lanes should be independent."""
        q = LaneQueue()
        results = []

        async def task(value):
            results.append(value)
            return value

        await q.submit("lane-A", task, "A")
        await q.submit("lane-B", task, "B")

        assert "A" in results
        assert "B" in results
        await q.shutdown()

    def test_lane_stats(self):
        """get_lane_stats should return a dict."""
        q = LaneQueue()
        stats = q.get_lane_stats()
        assert isinstance(stats, (dict, list))


# ══════════════════════════════════════════════════════════════════════
# TEST GROUP 6: Trust Policies
# ══════════════════════════════════════════════════════════════════════


class TestTrustPolicies:
    """Tests for trust level definitions."""

    def test_all_trust_levels_defined(self):
        """All three trust levels should be defined."""
        assert "yolo" in TRUST_POLICIES
        assert "normal" in TRUST_POLICIES
        assert "paranoid" in TRUST_POLICIES

    def test_trust_policies_have_descriptions(self):
        """Each trust policy should have a description."""
        for level, policy in TRUST_POLICIES.items():
            assert "description" in policy, f"{level} missing description"
            assert len(policy["description"]) > 0

    def test_yolo_is_most_permissive(self):
        """YOLO should auto-approve most/all tools."""
        yolo = TRUST_POLICIES["yolo"]
        assert (
            "*" in yolo.get("auto_approve_tools", [])
            or len(yolo.get("auto_approve_tools", [])) > 3
        )


# ══════════════════════════════════════════════════════════════════════
# TEST GROUP 7: Tool Definitions
# ══════════════════════════════════════════════════════════════════════


class TestToolDefinitions:
    """Tests for built-in tool schemas."""

    def test_builtin_tools_not_empty(self):
        """Should have at least 5 built-in tools."""
        assert len(BUILTIN_TOOLS) >= 5

    def test_all_tools_have_name(self):
        """Every tool must have a name."""
        for tool in BUILTIN_TOOLS:
            assert "name" in tool, f"Tool missing name: {tool}"
            assert len(tool["name"]) > 0

    def test_all_tools_have_description(self):
        """Every tool must have a description."""
        for tool in BUILTIN_TOOLS:
            assert "description" in tool, f"Tool {tool.get('name')} missing description"

    def test_all_tools_have_parameters(self):
        """Every tool must have a parameters schema."""
        for tool in BUILTIN_TOOLS:
            assert "parameters" in tool, f"Tool {tool.get('name')} missing parameters"
            assert "type" in tool["parameters"]

    def test_core_tools_present(self):
        """Core tools must be registered."""
        tool_names = {t["name"] for t in BUILTIN_TOOLS}
        expected = {
            "read_file",
            "write_file",
            "run_command",
            "web_search",
            "python_exec",
            "desktop_control",
        }
        for name in expected:
            assert name in tool_names, f"Missing core tool: {name}"

    def test_memory_tools_present(self):
        """Memory tools must be registered."""
        tool_names = {t["name"] for t in BUILTIN_TOOLS}
        assert "memory_store" in tool_names
        assert "memory_recall" in tool_names

    def test_native_tools_present(self):
        """Native in-process tools must be registered (no legacy gRPC tools)."""
        tool_names = {t["name"] for t in BUILTIN_TOOLS}
        expected = {
            "memory_store",
            "memory_recall",
            "strategic_plan",
            "multi_agent",
            "speak",
        }
        for name in expected:
            assert name in tool_names, f"Missing native tool: {name}"
        # Legacy gRPC tools must NOT be present
        legacy = {
            "legacy_rpc_health",
            "legacy_csc_process_goal",
            "legacy_daf_list_agent_types",
            "legacy_wil_list_tools",
            "legacy_wil_execute_tool",
            "legacy_kn_semantic_search",
            "legacy_kn_ingest_text",
        }
        for name in legacy:
            assert name not in tool_names, f"Legacy tool still present: {name}"

    def test_memory_store_accepts_tags(self):
        """memory_store schema should support tags array for categorization."""
        tool = next(t for t in BUILTIN_TOOLS if t["name"] == "memory_store")
        props = tool["parameters"]["properties"]
        assert "tags" in props, "memory_store missing 'tags' property"
        assert props["tags"]["type"] == "array"

    def test_memory_store_accepts_key(self):
        """memory_store schema should accept a key parameter."""
        tool = next(t for t in BUILTIN_TOOLS if t["name"] == "memory_store")
        props = tool["parameters"]["properties"]
        assert "key" in props, "memory_store missing 'key' property"
        assert props["key"]["type"] == "string"

    def test_memory_recall_accepts_query(self):
        """memory_recall schema should support query-based retrieval."""
        tool = next(t for t in BUILTIN_TOOLS if t["name"] == "memory_recall")
        props = tool["parameters"]["properties"]
        assert "query" in props, "memory_recall missing 'query' property"
        assert props["query"]["type"] == "string"

    def test_desktop_control_schema_has_health_check(self):
        """Desktop tool should expose health_check for readiness diagnostics."""
        desktop_tool = next(t for t in BUILTIN_TOOLS if t["name"] == "desktop_control")
        actions = desktop_tool["parameters"]["properties"]["action"]["enum"]
        assert "health_check" in actions
        assert "ax_query" in actions
        assert "ax_click" in actions
        assert "ax_set_value" in actions

    def test_openclaw_bridge_tool_schema(self):
        """OpenClaw bridge tool definition should expose task parameter."""
        assert len(OPENCLAW_BRIDGE_TOOLS) == 1
        tool = OPENCLAW_BRIDGE_TOOLS[0]
        assert tool["name"] == "openclaw_delegate"
        assert "task" in tool["parameters"]["required"]

    def test_tool_names_are_snake_case(self):
        """All tool names should be snake_case."""
        import re

        for tool in BUILTIN_TOOLS:
            assert re.match(
                r"^[a-z][a-z0-9_]*$", tool["name"]
            ), f"Tool name not snake_case: {tool['name']}"


# ══════════════════════════════════════════════════════════════════════
# TEST GROUP 8: Tool Executor
# ══════════════════════════════════════════════════════════════════════


class TestToolExecutor:
    """Tests for the ToolExecutor — PredaCore's hands."""

    def test_executor_init(self, config):
        """ToolExecutor should initialize with config."""
        executor = ToolExecutor(config)
        assert executor is not None

    def test_requires_confirmation_normal(self, config):
        """Normal trust should require confirmation for dangerous tools."""
        config.security.trust_level = "normal"
        executor = ToolExecutor(config)
        assert not executor._requires_confirmation("read_file")

    def test_requires_confirmation_paranoid(self, config):
        """Paranoid trust should require confirmation for everything."""
        config.security.trust_level = "paranoid"
        executor = ToolExecutor(config)
        assert executor._requires_confirmation("run_command")

    @pytest.mark.asyncio
    async def test_read_file_tool(self, config, tmp_home):
        """read_file tool should read actual files."""
        test_file = tmp_home / "test.txt"
        test_file.write_text("Hello from PredaCore!")

        executor = ToolExecutor(config)
        result = await executor.execute("read_file", {"path": str(test_file)})
        assert "Hello from PredaCore!" in result

    @pytest.mark.asyncio
    async def test_memory_store_and_recall(self, config):
        """memory_store and memory_recall should work together."""
        executor = ToolExecutor(config)

        # Store a fact (API uses key + content)
        await executor.execute(
            "memory_store",
            {"key": "test_fact", "content": "PredaCore is awesome"},
        )

        # Recall it (API uses query)
        result = await executor.execute(
            "memory_recall",
            {"query": "test_fact"},
        )
        assert "awesome" in result

    @pytest.mark.asyncio
    async def test_unknown_tool(self, config):
        """Unknown tool should return error message."""
        executor = ToolExecutor(config)
        result = await executor.execute("nonexistent_tool", {})
        assert "Unknown tool" in result or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_python_exec_tool(self, config):
        """python_exec tool should execute Python code."""
        executor = ToolExecutor(config)
        result = await executor.execute("python_exec", {"code": "print(2 + 2)"})
        assert "4" in result

    @pytest.mark.asyncio
    async def test_openclaw_delegate_requires_endpoint(self, config):
        """Bridge tool should require endpoint configuration when enabled."""
        config.launch.enable_openclaw_bridge = True
        executor = ToolExecutor(config)
        result = await executor.execute("openclaw_delegate", {"task": "hello"})
        assert "OPENCLAW_BRIDGE_URL" in result


class TestCoreLaunchBehavior:
    """Core behavior tied to launch profile controls."""

    def test_core_exposes_openclaw_tool_when_enabled(self, tmp_home, monkeypatch):
        cfg = PredaCoreConfig(
            home_dir=str(tmp_home),
            sessions_dir=str(tmp_home / "sessions"),
            skills_dir=str(tmp_home / "skills"),
            logs_dir=str(tmp_home / "logs"),
            llm=LLMConfig(provider="gemini-cli"),
            launch=LaunchProfileConfig(
                profile="beast",
                approvals_required=False,
                egm_mode="off",
                default_code_network=True,
                enable_openclaw_bridge=True,
                max_tool_iterations=25,
            ),
            openclaw=OpenClawBridgeConfig(base_url="https://bridge.example.com"),
        )

        core = PredaCoreCore(cfg)
        tools = core.get_tool_list()
        assert "openclaw_delegate" in tools
        status = core.get_status()
        assert status["max_tool_iterations"] == 25
        assert status["openclaw"]["enabled"] is True
        assert "ledger_path" in status["openclaw"]
        assert status["persona_drift_guard"]["enabled"] is True

        # Clean up env vars set by PredaCoreCore.__init__ to prevent test pollution
        monkeypatch.delenv("PREDACORE_ENABLE_OPENCLAW_BRIDGE", raising=False)
        monkeypatch.delenv("PREDACORE_ENABLE_PLUGIN_MARKETPLACE", raising=False)


class TestPersonaDriftGuard:
    """Response regeneration behavior for persona drift protection."""

    @pytest.mark.asyncio
    async def test_regenerates_when_drift_exceeds_threshold(self, config):
        config.launch.enable_persona_drift_guard = True
        config.launch.persona_drift_threshold = 0.2
        config.launch.persona_drift_max_regens = 1
        core = PredaCoreCore(config)

        core.llm.chat = AsyncMock(
            side_effect=[
                {
                    "content": "As an AI language model, I cannot run tools.",
                    "tool_calls": [],
                    "usage": {"completion_tokens": 10},
                },
                {
                    "content": "I am PredaCore from PredaCore. I can run tools.",
                    "tool_calls": [],
                    "usage": {"completion_tokens": 11},
                },
            ]
        )

        session = Session(session_id="s1", user_id="u1")
        out = await core.process(user_id="u1", message="who are you", session=session)

        assert "PredaCore" in out
        assert core.llm.chat.await_count == 2

    @pytest.mark.asyncio
    async def test_no_regen_when_guard_disabled(self, config):
        config.launch.enable_persona_drift_guard = False
        config.launch.persona_drift_threshold = 0.2
        config.launch.persona_drift_max_regens = 2
        core = PredaCoreCore(config)

        core.llm.chat = AsyncMock(
            return_value={
                "content": "As an AI language model, I cannot run tools.",
                "tool_calls": [],
                "usage": {"completion_tokens": 10},
            }
        )

        session = Session(session_id="s2", user_id="u2")
        await core.process(user_id="u2", message="who are you", session=session)

        assert core.llm.chat.await_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_tool_refusal_when_tools_available(self, config):
        config.launch.max_tool_iterations = 3
        core = PredaCoreCore(config)

        core.llm.chat = AsyncMock(
            side_effect=[
                {
                    "content": (
                        "I can’t create an Apple Notes note from here because I don’t "
                        "have desktop automation access in this session."
                    ),
                    "tool_calls": [],
                    "usage": {"completion_tokens": 12},
                },
                {
                    "content": "ok",
                    "tool_calls": [],
                    "usage": {"completion_tokens": 4},
                },
            ]
        )

        session = Session(session_id="s2b", user_id="u2b")
        out = await core.process(
            user_id="u2b",
            message="create a note called Prometheus Tools",
            session=session,
        )

        assert out == "ok"
        assert core.llm.chat.await_count == 2

    @pytest.mark.asyncio
    async def test_regenerates_unverified_terminal_claim_without_tools(self, config):
        config.launch.enable_persona_drift_guard = True
        config.launch.persona_drift_threshold = 0.6
        config.launch.persona_drift_max_regens = 1
        core = PredaCoreCore(config)

        core.llm.chat = AsyncMock(
            side_effect=[
                {
                    "content": "I checked with terminal and switched the model for you.",
                    "tool_calls": [],
                    "usage": {"completion_tokens": 10},
                },
                {
                    "content": (
                        "I have not executed terminal commands in this turn. "
                        "I can run a command now to verify the active model."
                    ),
                    "tool_calls": [],
                    "usage": {"completion_tokens": 12},
                },
            ]
        )

        session = Session(session_id="s3", user_id="u3")
        out = await core.process(
            user_id="u3",
            message="find out with terminal what model is running",
            session=session,
        )

        assert "not executed terminal commands" in out
        assert core.llm.chat.await_count == 2

    @pytest.mark.asyncio
    async def test_allows_execution_summary_after_tool_use(self, config):
        config.launch.enable_persona_drift_guard = True
        config.launch.persona_drift_threshold = 0.6
        config.launch.persona_drift_max_regens = 1
        core = PredaCoreCore(config)

        core.tools.execute = AsyncMock(return_value="model=gemini-3-pro-preview")
        core.llm.chat = AsyncMock(
            side_effect=[
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "run_command",
                            "arguments": {"command": "echo model=gemini-3-pro-preview"},
                        }
                    ],
                    "usage": {"completion_tokens": 4},
                },
                {
                    "content": "I checked with terminal and found model=gemini-3-pro-preview.",
                    "tool_calls": [],
                    "usage": {"completion_tokens": 8},
                },
            ]
        )

        session = Session(session_id="s4", user_id="u4")
        out = await core.process(
            user_id="u4",
            message="find out with terminal what model is running",
            session=session,
        )

        assert "model=gemini-3-pro-preview" in out
        assert core.llm.chat.await_count == 2
        assert core.tools.execute.await_count == 1

    @pytest.mark.asyncio
    async def test_direct_native_tool_shortcut_bypasses_llm(self, config):
        core = PredaCoreCore(config)
        core.tools.execute = AsyncMock(return_value='{"results": []}')
        core.llm.chat = AsyncMock(
            return_value={"content": "fallback", "tool_calls": [], "usage": {}}
        )

        session = Session(session_id="s5", user_id="u5")
        out = await core.process(
            user_id="u5",
            message="Run memory_recall and return raw JSON",
            session=session,
        )

        assert out == '{"results": []}'
        core.tools.execute.assert_awaited_once()
        args, kwargs = core.tools.execute.await_args
        assert args[0] == "memory_recall"
        assert isinstance(args[1], dict)
        assert kwargs.get("confirm_fn") is None
        core.llm.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_direct_tool_shortcut_parses_json_args(self, config):
        core = PredaCoreCore(config)
        core.tools.execute = AsyncMock(return_value='{"results": []}')
        core.llm.chat = AsyncMock(
            return_value={"content": "fallback", "tool_calls": [], "usage": {}}
        )

        session = Session(session_id="s6", user_id="u6")
        out = await core.process(
            user_id="u6",
            message='execute memory_recall {"query": "test", "top_k": 3}',
            session=session,
        )

        assert out == '{"results": []}'
        core.tools.execute.assert_awaited_once()
        args, _ = core.tools.execute.await_args
        assert args[0] == "memory_recall"
        assert args[1].get("query") == "test"
        assert args[1].get("top_k") == 3
        core.llm.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_compacts_long_history_before_llm_call(self, config):
        core = PredaCoreCore(config)
        core.llm.chat = AsyncMock(
            return_value={"content": "ok", "tool_calls": [], "usage": {}}
        )

        session = Session(session_id="s7", user_id="u7")
        for idx in range(30):
            session.add_message(
                "user",
                f"Long user message {idx}: " + ("context " * 500),
            )
            session.add_message(
                "assistant",
                f"Long assistant reply {idx}: " + ("response " * 500),
            )

        out = await core.process(
            user_id="u7",
            message="What matters now?",
            session=session,
        )

        assert out == "ok"
        sent_messages = core.llm.chat.await_args.kwargs["messages"]
        history_messages = sent_messages[1:-1]
        assert history_messages
        assert history_messages[0]["role"] == "system"
        assert "Session summary" in history_messages[0]["content"]

        total_tokens = sum(
            Session.estimate_tokens(msg["content"]) for msg in sent_messages
        )
        assert total_tokens < 40_000


# ══════════════════════════════════════════════════════════════════════
# TEST GROUP 9: SOUL.md Content Validation
# ══════════════════════════════════════════════════════════════════════


class TestSoulSeedContent:
    """Validate that SOUL_SEED.md has all required sections."""

    @pytest.fixture
    def seed_content(self):
        """Load the built-in SOUL_SEED.md using absolute path."""
        seed_path = IDENTITY_DIR / "SOUL_SEED.md"
        assert seed_path.exists(), f"SOUL_SEED.md not found at {seed_path}"
        return seed_path.read_text(encoding="utf-8")

    def test_has_security_invariants(self, seed_content):
        """SOUL_SEED.md must define security invariants."""
        # Current soul seed uses "The Invariants" as the section header and
        # enumerates credential, destructive-op, and sandbox rules under it.
        assert "Invariants" in seed_content
        assert "Credentials" in seed_content or "credentials" in seed_content

    def test_has_trust_levels(self, seed_content):
        """SOUL_SEED.md must describe trust levels."""
        assert "yolo" in seed_content
        assert "normal" in seed_content
        assert "paranoid" in seed_content

    def test_has_ethical_boundaries(self, seed_content):
        """SOUL_SEED.md must define ethical boundaries."""
        assert "Ethical" in seed_content or "credential" in seed_content.lower()

    def test_has_memory_rules(self, seed_content):
        """SOUL_SEED.md should reference the memory system / memory rules."""
        # Current soul seed references "memory" in lowercase across multiple
        # invariants (credentials never stored in memory, session isolation
        # "through the memory system the human controls", etc.).
        assert "memory" in seed_content.lower()

    def test_no_secrets(self, seed_content):
        """SOUL_SEED.md must not contain any actual secrets."""
        forbidden = ["sk-", "AIza", "Bearer ", "password:", "secret:"]
        for pattern in forbidden:
            assert (
                pattern not in seed_content
            ), f"SOUL_SEED.md contains potential secret: {pattern}"

    def test_no_personal_info(self, seed_content):
        """SOUL_SEED.md must not contain personal user data (launch-safe)."""
        # The seed is immutable and ships with source — no personal data allowed
        content_lower = seed_content.lower()
        assert "shubham" not in content_lower, "SOUL_SEED.md contains personal name"

    def test_references_self_discovery(self, seed_content):
        """SOUL_SEED.md should reference identity self-discovery."""
        assert "discover" in seed_content.lower() or "evolve" in seed_content.lower()


# ══════════════════════════════════════════════════════════════════════
# TEST GROUP 12: Integration — Full Prompt Assembly
# ══════════════════════════════════════════════════════════════════════


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_prompt_assembly(self, config):
        """Full prompt should assemble all layers."""
        prompt = _get_system_prompt(config)

        # Should be substantial (seed + runtime context)
        assert len(prompt) > 500, f"Full prompt too short: {len(prompt)} chars"

        # Should contain content from identity seed. The seed header
        # evolved over time — accept any of the historic or current
        # headings ("Cannot Change" / "Security Invariants" / "The
        # Invariants" / "The Promises You Keep").
        assert (
            "Invariants" in prompt
            or "Cannot Change" in prompt
            or "Promises You Keep" in prompt
        )
        assert "Runtime Context" in prompt  # Runtime

    def test_prompt_with_all_configs(self, config):
        """Prompt should work with various config combinations."""
        for trust in ["yolo", "normal"]:
            for profile in ["enterprise", "beast"]:
                config.security.trust_level = trust
                config.launch.profile = profile
                prompt = _get_system_prompt(config)
                assert len(prompt) > 0
                assert trust.upper() in prompt
                assert profile in prompt


# ══════════════════════════════════════════════════════════════════════
# TEST GROUP 14: Edge Cases & Security
# ══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and security tests."""

    def test_prompt_never_empty(self, config):
        """System prompt should never be empty, regardless of config."""
        prompt = _get_system_prompt(config)
        assert len(prompt) > 100

    def test_identity_files_exist(self):
        """Core identity files must exist in the repo."""
        assert (IDENTITY_DIR / "SOUL_SEED.md").exists(), "SOUL_SEED.md missing!"
        assert (IDENTITY_DIR / "EVENT_HORIZON.md").exists(), "EVENT_HORIZON.md missing!"

    def test_soul_seed_not_trivially_small(self):
        """SOUL_SEED.md should be substantial (>2KB)."""
        size = (IDENTITY_DIR / "SOUL_SEED.md").stat().st_size
        assert size > 2000, f"SOUL_SEED.md too small: {size} bytes"

    def test_identity_files_are_utf8(self):
        """Identity files should be valid UTF-8."""
        for name in ["SOUL_SEED.md", "EVENT_HORIZON.md"]:
            content = (IDENTITY_DIR / name).read_text(encoding="utf-8")
            assert len(content) > 0


class TestIdentityToolSurface:
    """Identity tool schemas/handlers should match the engine's writable files."""

    def test_registry_exposes_extended_identity_files(self):
        reg = build_builtin_registry()
        read_enum = reg.get("identity_read").parameters["properties"]["file"]["enum"]
        write_enum = reg.get("identity_update").parameters["properties"]["file"]["enum"]
        for name in ["TOOLS", "MEMORY", "HEARTBEAT", "REFLECTION"]:
            assert name in read_enum
            assert name in write_enum

    @pytest.mark.asyncio
    async def test_identity_handlers_support_extended_files(self, config, identity_dir):
        ctx = ToolContext(config=config, memory={})
        memory_text = "# Memory\n- shared blackboard promotes into global memory"
        tools_text = "# Tools\n- dynamic registry is source of truth"

        result = await handle_identity_update({"file": "MEMORY", "content": memory_text}, ctx)
        assert '"status": "ok"' in result
        assert "blackboard" in await handle_identity_read({"file": "MEMORY"}, ctx)

        result = await handle_identity_update({"file": "TOOLS", "content": tools_text}, ctx)
        assert '"status": "ok"' in result
        assert "dynamic registry" in await handle_identity_read({"file": "TOOLS"}, ctx)
