"""
Comprehensive tests for predacore.config — 3-layer configuration system.

Tests: defaults, YAML loading, env overrides, profile presets,
dataclass construction, helper functions, and edge cases.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from predacore.config import (
    DEFAULT_AGENT,
    DEFAULT_HOME,
    DEFAULT_PROFILE,
    PROFILE_PRESETS,
    AgentLLMConfig,
    ChannelConfig,
    DaemonConfig,
    LaunchProfileConfig,
    LLMConfig,
    MemoryConfig,
    OpenClawBridgeConfig,
    OperatorsConfig,
    PredaCoreConfig,
    SecurityConfig,
    _deep_merge,
    _dict_to_config,
    _env_overrides,
    _get_profile_defaults,
    _load_yaml_config,
    _parse_bool,
    _parse_csv,
    _resolve_profile_name,
    _safe_float,
    _safe_int,
    _validate_port,
    load_config,
    save_default_config,
)

# ── Helper Functions ───────────────────────────────────────────────


class TestParseBool:
    """Tests for _parse_bool helper."""

    @pytest.mark.parametrize("val", ["1", "true", "True", "TRUE", "yes", "Yes", "on", "ON", "y", "Y"])
    def test_truthy_values(self, val):
        assert _parse_bool(val) is True

    @pytest.mark.parametrize("val", ["0", "false", "False", "no", "off", "n", "", "random", "maybe"])
    def test_falsy_values(self, val):
        assert _parse_bool(val) is False

    def test_whitespace_handling(self):
        assert _parse_bool("  true  ") is True
        assert _parse_bool("  false  ") is False


class TestParseCSV:
    """Tests for _parse_csv helper."""

    def test_basic_csv(self):
        assert _parse_csv("cli,telegram,discord") == ["cli", "telegram", "discord"]

    def test_whitespace_stripping(self):
        assert _parse_csv("cli , telegram , discord") == ["cli", "telegram", "discord"]

    def test_empty_string(self):
        assert _parse_csv("") == []

    def test_single_value(self):
        assert _parse_csv("cli") == ["cli"]

    def test_trailing_comma(self):
        assert _parse_csv("a,b,") == ["a", "b"]

    def test_empty_entries_filtered(self):
        assert _parse_csv("a,,b,,,c") == ["a", "b", "c"]


class TestSafeInt:
    """Tests for _safe_int helper."""

    def test_valid_int(self):
        assert _safe_int("42") == 42

    def test_negative(self):
        assert _safe_int("-5") == -5

    def test_invalid_string(self):
        assert _safe_int("abc", default=10) == 10

    def test_empty_string(self):
        assert _safe_int("", default=0) == 0

    def test_float_string(self):
        assert _safe_int("3.14", default=3) == 3

    def test_none(self):
        assert _safe_int(None, default=7) == 7


class TestSafeFloat:
    """Tests for _safe_float helper."""

    def test_valid_float(self):
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_integer_string(self):
        assert _safe_float("42") == 42.0

    def test_invalid_string(self):
        assert _safe_float("xyz", default=1.5) == 1.5

    def test_empty_string(self):
        assert _safe_float("", default=0.0) == 0.0

    def test_none(self):
        assert _safe_float(None, default=2.5) == 2.5


class TestValidatePort:
    """Tests for _validate_port helper."""

    def test_valid_port(self):
        assert _validate_port(8080) == 8080

    def test_min_port(self):
        assert _validate_port(1) == 1

    def test_max_port(self):
        assert _validate_port(65535) == 65535

    def test_below_min_clamped(self):
        assert _validate_port(0) == 1

    def test_above_max_clamped(self):
        assert _validate_port(70000) == 65535

    def test_negative_clamped(self):
        assert _validate_port(-1) == 1


class TestDeepMerge:
    """Tests for _deep_merge helper."""

    def test_flat_merge(self):
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_override(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3, "z": 4}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_deep_nested(self):
        base = {"a": {"b": {"c": 1}}}
        override = {"a": {"b": {"d": 2}}}
        result = _deep_merge(base, override)
        assert result == {"a": {"b": {"c": 1, "d": 2}}}

    def test_override_dict_with_scalar(self):
        result = _deep_merge({"a": {"x": 1}}, {"a": "flat"})
        assert result == {"a": "flat"}

    def test_base_not_mutated(self):
        base = {"a": 1}
        _deep_merge(base, {"b": 2})
        assert base == {"a": 1}

    def test_empty_base(self):
        assert _deep_merge({}, {"a": 1}) == {"a": 1}

    def test_empty_override(self):
        assert _deep_merge({"a": 1}, {}) == {"a": 1}

    def test_both_empty(self):
        assert _deep_merge({}, {}) == {}


# ── Dataclass Defaults ─────────────────────────────────────────────


class TestDataclassDefaults:
    """Test that all config dataclasses have sensible defaults."""

    def test_llm_config_defaults(self):
        cfg = LLMConfig()
        assert cfg.provider == "gemini-cli"
        assert cfg.model == ""
        assert cfg.temperature == 0.7
        # Raised 4096 → 32000 per "remove all limits" — Claude Opus extended output
        assert cfg.max_tokens == 32000
        assert cfg.reasoning_effort == "medium"
        assert cfg.auto_fallback is False
        assert cfg.max_retries == 3
        assert cfg.retry_jitter is True
        # Default is now empty — fallback providers are explicit opt-in. A
        # stale default like ["gemini", "openrouter"] leaked into error
        # messages ("/model openrouter") for users who never configured those.
        assert cfg.fallback_providers == []

    def test_agent_llm_config_defaults(self):
        cfg = AgentLLMConfig()
        assert cfg.provider == ""
        assert cfg.model == ""

    def test_channel_config_defaults(self):
        cfg = ChannelConfig()
        # Default channel is now "webchat" — the daemon hosts a WebSocket
        # server at port 3000 that both the browser UI and `predacore chat`
        # connect to. "cli" was the pre-gateway legacy channel.
        assert cfg.enabled == ["webchat"]
        assert cfg.telegram_token == ""
        assert cfg.discord_token == ""
        assert cfg.whatsapp_token == ""
        assert cfg.webchat_port == 3000

    def test_security_config_defaults(self):
        cfg = SecurityConfig()
        assert cfg.trust_level == "ask_everytime"
        assert cfg.permission_mode == "auto"
        assert cfg.docker_sandbox is False
        # Raised 5 → 50 per "remove all limits"
        assert cfg.max_concurrent_tasks == 50
        # Raised 300 → 3600 per "remove all limits"
        assert cfg.task_timeout_seconds == 3600
        assert cfg.allowed_tools == []
        assert cfg.blocked_tools == []

    def test_daemon_config_defaults(self):
        cfg = DaemonConfig()
        assert cfg.enabled is False
        assert cfg.webhook_port == 8765

    def test_launch_profile_config_defaults(self):
        cfg = LaunchProfileConfig()
        assert cfg.profile == "enterprise"
        # 2-mode simplification: resource limits maxed on both profiles
        assert cfg.max_tool_iterations == 1000
        assert cfg.enable_persona_drift_guard is True
        assert cfg.persona_drift_threshold == pytest.approx(0.32)
        assert cfg.persona_drift_max_regens == 5
        assert cfg.max_spawn_depth == 16
        assert cfg.max_spawn_fanout == 64

    def test_memory_config_defaults(self):
        cfg = MemoryConfig()
        assert cfg.enable_knowledge_graph is True
        assert cfg.enable_vector_store is True
        assert cfg.working_memory_capacity == 7
        assert cfg.decay_rate == pytest.approx(0.01)

    def test_openclaw_config_defaults(self):
        cfg = OpenClawBridgeConfig()
        assert cfg.base_url == ""
        assert cfg.timeout_seconds == 180
        assert cfg.max_retries == 2

    def test_operators_config_defaults(self):
        cfg = OperatorsConfig()
        assert cfg.macro_max_steps == 50
        assert cfg.macro_max_depth == 3
        assert cfg.ax_default_depth == 4
        assert cfg.screenshot_max_b64_bytes == 10_000_000


class TestPredaCoreConfigMaster:
    """Tests for the master PredaCoreConfig dataclass."""

    def test_defaults(self):
        cfg = PredaCoreConfig()
        assert cfg.name == "PredaCore"
        assert cfg.version == "0.1.0"
        assert cfg.agent == "default"

    def test_post_init_sets_paths(self):
        cfg = PredaCoreConfig()
        assert cfg.home_dir == str(DEFAULT_HOME)
        assert "sessions" in cfg.sessions_dir
        assert "skills" in cfg.skills_dir
        assert "logs" in cfg.logs_dir
        assert "memory" in cfg.memory.persistence_dir
        assert "db.sock" in cfg.daemon.db_socket_path

    def test_custom_home_dir(self):
        cfg = PredaCoreConfig(home_dir="/tmp/custom_prom")
        assert cfg.home_dir == "/tmp/custom_prom"
        assert cfg.sessions_dir == "/tmp/custom_prom/sessions"
        assert cfg.skills_dir == "/tmp/custom_prom/skills"
        assert cfg.logs_dir == "/tmp/custom_prom/logs"

    def test_agents_dir_property(self):
        cfg = PredaCoreConfig(home_dir="/tmp/prom")
        assert cfg.agents_dir == "/tmp/prom/agents"

    def test_agent_dir_property(self):
        cfg = PredaCoreConfig(home_dir="/tmp/prom", agent="default")
        assert cfg.agent_dir == "/tmp/prom/agents/default"

    def test_flame_dir_property(self):
        cfg = PredaCoreConfig(home_dir="/tmp/prom")
        assert cfg.flame_dir == "/tmp/prom/flame"

    def test_sub_configs_are_instances(self):
        cfg = PredaCoreConfig()
        assert isinstance(cfg.llm, LLMConfig)
        assert isinstance(cfg.agent_llm, AgentLLMConfig)
        assert isinstance(cfg.channels, ChannelConfig)
        assert isinstance(cfg.security, SecurityConfig)
        assert isinstance(cfg.daemon, DaemonConfig)
        assert isinstance(cfg.memory, MemoryConfig)
        assert isinstance(cfg.launch, LaunchProfileConfig)
        assert isinstance(cfg.openclaw, OpenClawBridgeConfig)
        assert isinstance(cfg.operators, OperatorsConfig)


# ── Profile Presets ────────────────────────────────────────────────


class TestProfilePresets:
    """Tests for the 2 launch profile presets (enterprise + beast)."""

    def test_enterprise_exists(self):
        assert "enterprise" in PROFILE_PRESETS

    def test_beast_exists(self):
        assert "beast" in PROFILE_PRESETS

    def test_exactly_two_profiles(self):
        assert set(PROFILE_PRESETS.keys()) == {"enterprise", "beast"}

    def test_enterprise_posture(self):
        p = PROFILE_PRESETS["enterprise"]
        assert p["security"]["trust_level"] == "ask_everytime"
        assert p["security"]["docker_sandbox"] is True
        assert p["launch"]["approvals_required"] is True
        assert p["launch"]["egm_mode"] == "strict"
        assert p["launch"]["enable_self_evolution"] is False
        assert p["launch"]["enable_plugin_marketplace"] is False
        assert p["launch"]["default_code_network"] is False

    def test_beast_posture(self):
        p = PROFILE_PRESETS["beast"]
        assert p["security"]["trust_level"] == "yolo"
        assert p["security"]["docker_sandbox"] is True
        assert p["launch"]["approvals_required"] is False
        assert p["launch"]["egm_mode"] == "off"
        assert p["launch"]["enable_self_evolution"] is True
        assert p["launch"]["enable_plugin_marketplace"] is True
        assert p["launch"]["default_code_network"] is True

    def test_resource_limits_maxed_on_both(self):
        # Both modes share maxed-out resource limits; the split is governance-only.
        for name in ("enterprise", "beast"):
            p = PROFILE_PRESETS[name]
            assert p["security"]["max_concurrent_tasks"] == 100
            assert p["security"]["task_timeout_seconds"] == 3600
            assert p["launch"]["max_spawn_depth"] == 16
            assert p["launch"]["max_spawn_fanout"] == 64
            assert p["launch"]["max_tool_iterations"] == 1000
            assert p["launch"]["persona_drift_max_regens"] == 5

    def test_get_profile_defaults_valid(self):
        result = _get_profile_defaults("enterprise")
        assert result == PROFILE_PRESETS["enterprise"]

    def test_get_profile_defaults_unknown_falls_back(self):
        result = _get_profile_defaults("nonexistent_profile")
        assert result == PROFILE_PRESETS[DEFAULT_PROFILE]

    def test_resolve_profile_from_override(self):
        assert _resolve_profile_name({}, "beast") == "beast"

    def test_resolve_profile_from_merged(self):
        merged = {"launch": {"profile": "enterprise"}}
        assert _resolve_profile_name(merged, None) == "enterprise"

    def test_resolve_profile_default(self):
        assert _resolve_profile_name({}, None) == DEFAULT_PROFILE


# ── YAML Loading ───────────────────────────────────────────────────


class TestYAMLLoading:
    """Tests for YAML config file loading."""

    def test_missing_file_returns_empty(self):
        result = _load_yaml_config(Path("/nonexistent/config.yaml"))
        assert result == {}

    def test_valid_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("name: TestBot\nmode: enterprise\n")
        result = _load_yaml_config(config_file)
        assert result["name"] == "TestBot"
        assert result["mode"] == "enterprise"

    def test_empty_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        result = _load_yaml_config(config_file)
        assert result == {}

    def test_nested_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "llm:\n  provider: openai\n  model: gpt-4\nsecurity:\n  trust_level: ask_everytime\n"
        )
        result = _load_yaml_config(config_file)
        assert result["llm"]["provider"] == "openai"
        assert result["security"]["trust_level"] == "ask_everytime"


# ── Dict to Config ─────────────────────────────────────────────────


class TestDictToConfig:
    """Tests for _dict_to_config conversion."""

    def test_empty_dict(self):
        cfg = _dict_to_config({})
        assert isinstance(cfg, PredaCoreConfig)
        assert cfg.name == "PredaCore"

    def test_top_level_fields(self):
        cfg = _dict_to_config({"name": "MyBot"})
        assert cfg.name == "MyBot"

    def test_nested_sub_config(self):
        cfg = _dict_to_config({
            "llm": {"provider": "openai", "model": "gpt-4", "temperature": 0.3},
            "security": {"trust_level": "ask_everytime"},
        })
        assert cfg.llm.provider == "openai"
        assert cfg.llm.model == "gpt-4"
        assert cfg.llm.temperature == pytest.approx(0.3)
        assert cfg.security.trust_level == "ask_everytime"

    def test_unknown_keys_ignored(self):
        cfg = _dict_to_config({"llm": {"provider": "openai", "bogus_key": True}})
        assert cfg.llm.provider == "openai"
        assert not hasattr(cfg.llm, "bogus_key")

    def test_all_sub_configs(self):
        data = {
            "llm": {"provider": "anthropic"},
            "agent_llm": {"provider": "gemini"},
            "channels": {"enabled": ["cli", "telegram"]},
            "security": {"trust_level": "yolo"},
            "daemon": {"enabled": True},
            "memory": {"decay_rate": 0.05},
            "launch": {"profile": "beast"},
            "openclaw": {"base_url": "http://bridge"},
            "operators": {"macro_max_steps": 100},
        }
        cfg = _dict_to_config(data)
        assert cfg.llm.provider == "anthropic"
        assert cfg.agent_llm.provider == "gemini"
        assert cfg.channels.enabled == ["cli", "telegram"]
        assert cfg.security.trust_level == "yolo"
        assert cfg.daemon.enabled is True
        assert cfg.memory.decay_rate == pytest.approx(0.05)
        assert cfg.launch.profile == "beast"
        assert cfg.openclaw.base_url == "http://bridge"
        assert cfg.operators.macro_max_steps == 100


# ── Environment Overrides ──────────────────────────────────────────


class TestEnvOverrides:
    """Tests for environment variable override mapping."""

    def test_no_env_vars_empty_result(self):
        with patch.dict(os.environ, {}, clear=True):
            result = _env_overrides()
            assert result == {} or all(v == {} for v in result.values() if isinstance(v, dict))

    def test_llm_provider_override(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True):
            result = _env_overrides()
            assert result.get("llm", {}).get("provider") == "openai"

    def test_trust_level_override(self):
        with patch.dict(os.environ, {"PREDACORE_TRUST_LEVEL": "ask_everytime"}, clear=True):
            result = _env_overrides()
            assert result.get("security", {}).get("trust_level") == "ask_everytime"

    def test_channels_csv(self):
        with patch.dict(os.environ, {"PREDACORE_CHANNELS": "cli,telegram,discord"}, clear=True):
            result = _env_overrides()
            assert result.get("channels", {}).get("enabled") == ["cli", "telegram", "discord"]

    def test_bool_env_vars(self):
        with patch.dict(os.environ, {"PREDACORE_ENABLE_SELF_EVOLUTION": "true"}, clear=True):
            result = _env_overrides()
            assert result.get("launch", {}).get("enable_self_evolution") is True

    def test_int_env_vars(self):
        with patch.dict(os.environ, {"PREDACORE_MAX_TOOL_ITERATIONS": "25"}, clear=True):
            result = _env_overrides()
            assert result.get("launch", {}).get("max_tool_iterations") == 25

    def test_float_env_vars(self):
        with patch.dict(os.environ, {"PREDACORE_PERSONA_DRIFT_THRESHOLD": "0.55"}, clear=True):
            result = _env_overrides()
            assert result.get("launch", {}).get("persona_drift_threshold") == pytest.approx(0.55)

    def test_provider_api_key_resolution(self):
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test123"},
            clear=True,
        ):
            result = _env_overrides()
            assert result.get("llm", {}).get("api_key") == "sk-test123"

    def test_wrong_provider_key_not_applied(self):
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "gemini", "OPENAI_API_KEY": "sk-test123"},
            clear=True,
        ):
            result = _env_overrides()
            # OpenAI key should NOT be applied when provider is gemini
            assert result.get("llm", {}).get("api_key") is None


# ── Full load_config ───────────────────────────────────────────────


class TestLoadConfig:
    """Tests for the full load_config pipeline."""

    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch):
        """Remove PREDACORE_*/PROMETHEUS_* env vars so tests are isolated."""
        for key in list(os.environ):
            if key.startswith("PREDACORE_") or key.startswith("PROMETHEUS_") or key.startswith("LLM_") or key.startswith("TRUST_") or key in ("PROFILE",):
                monkeypatch.delenv(key, raising=False)

    def test_defaults_without_yaml(self, tmp_path):
        """Config loads with pure defaults when no YAML exists."""
        cfg = load_config(config_path=str(tmp_path / "nonexistent.yaml"))
        assert isinstance(cfg, PredaCoreConfig)
        assert cfg.name == "PredaCore"
        assert cfg.launch.profile == "enterprise"

    def test_yaml_override(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("name: CustomBot\nllm:\n  provider: openai\n")
        cfg = load_config(config_path=str(config_file))
        assert cfg.name == "CustomBot"
        assert cfg.llm.provider == "openai"

    def test_env_overrides_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("llm:\n  provider: gemini\n")
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic"}, clear=False):
            cfg = load_config(config_path=str(config_file))
            assert cfg.llm.provider == "anthropic"  # env wins

    def test_profile_override(self, tmp_path):
        cfg = load_config(
            config_path=str(tmp_path / "none.yaml"),
            profile_override="beast",
        )
        assert cfg.launch.profile == "beast"
        assert cfg.security.trust_level == "yolo"

    def test_legacy_normal_normalized_to_ask_everytime(self, tmp_path):
        """Old configs with trust_level: normal still load — silently mapped."""
        config_file = tmp_path / "legacy.yaml"
        config_file.write_text("security:\n  trust_level: normal\n")
        cfg = load_config(config_path=str(config_file))
        assert cfg.security.trust_level == "ask_everytime"

    def test_legacy_paranoid_normalized_to_ask_everytime(self, tmp_path):
        """Old configs with trust_level: paranoid still load — silently mapped."""
        config_file = tmp_path / "legacy.yaml"
        config_file.write_text("security:\n  trust_level: paranoid\n")
        cfg = load_config(config_path=str(config_file))
        assert cfg.security.trust_level == "ask_everytime"

    def test_directories_created(self, tmp_path):
        home = tmp_path / "prom_home"
        with patch.dict(os.environ, {"PREDACORE_HOME": str(home)}, clear=False):
            cfg = load_config(config_path=str(tmp_path / "none.yaml"))
            assert Path(cfg.home_dir).exists()
            assert Path(cfg.sessions_dir).exists()
            assert Path(cfg.logs_dir).exists()
            assert Path(cfg.memory.persistence_dir).exists()


# ── save_default_config ────────────────────────────────────────────


class TestSaveDefaultConfig:
    """Tests for save_default_config."""

    def test_creates_file(self, tmp_path):
        target = tmp_path / "config.yaml"
        result = save_default_config(str(target))
        assert result == target
        assert target.exists()

    def test_content_has_provider(self, tmp_path):
        target = tmp_path / "config.yaml"
        save_default_config(str(target), provider="openai")
        content = target.read_text()
        assert "provider: openai" in content

    def test_content_has_trust_level(self, tmp_path):
        target = tmp_path / "config.yaml"
        save_default_config(str(target), trust_level="ask_everytime")
        content = target.read_text()
        assert "trust_level: ask_everytime" in content

    def test_content_has_model(self, tmp_path):
        target = tmp_path / "config.yaml"
        save_default_config(str(target), model="gpt-4o")
        content = target.read_text()
        assert 'model: "gpt-4o"' in content

    def test_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "config.yaml"
        save_default_config(str(target))
        assert target.exists()

    def test_restrictive_permissions(self, tmp_path):
        target = tmp_path / "config.yaml"
        save_default_config(str(target))
        mode = target.stat().st_mode & 0o777
        assert mode == 0o600

    def test_default_provider_is_gemini_cli(self, tmp_path):
        target = tmp_path / "config.yaml"
        save_default_config(str(target))
        content = target.read_text()
        assert "provider: gemini-cli" in content


# ── Module Constants ───────────────────────────────────────────────


class TestModuleConstants:
    """Verify module-level constants are correct."""

    def test_default_home(self):
        assert DEFAULT_HOME == Path.home() / ".predacore"

    def test_default_profile(self):
        assert DEFAULT_PROFILE == "enterprise"

    def test_default_agent(self):
        assert DEFAULT_AGENT == "default"

    def test_profile_presets_count(self):
        assert len(PROFILE_PRESETS) == 2
