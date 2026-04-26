"""
PredaCore Configuration System.

Loads config from (in priority order):
  1. Environment variables (PREDACORE_*)
  2. ~/.predacore/config.yaml
  3. Hardcoded defaults

Usage:
    from predacore.config import load_config
    cfg = load_config()
    print(cfg.trust_level)  # "normal"
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_HOME = Path.home() / ".predacore"
DEFAULT_CONFIG_FILE = DEFAULT_HOME / "config.yaml"
DEFAULT_SESSIONS_DIR = DEFAULT_HOME / "sessions"
DEFAULT_MEMORY_DIR = DEFAULT_HOME / "memory"
DEFAULT_SKILLS_DIR = DEFAULT_HOME / "skills"
DEFAULT_LOGS_DIR = DEFAULT_HOME / "logs"
DEFAULT_AGENTS_DIR = DEFAULT_HOME / "agents"
DEFAULT_FLAME_DIR = DEFAULT_HOME / "flame"
DEFAULT_PROFILE = "enterprise"
DEFAULT_AGENT = "default"


@dataclass
class LLMConfig:
    """LLM provider settings."""

    provider: str = (
        "gemini-cli"  # Free default — no API key needed if `gemini` CLI is installed
        # Other providers: openai | gemini | anthropic | groq | xai
        # | deepseek | cerebras | together | openrouter | sambanova
        # | mistral | fireworks | nvidia | zhipu | ollama
    )
    model: str = ""  # Model name (provider-specific)
    fallback_models: list[str] = field(
        default_factory=list
    )  # Ordered model fallback chain (provider-specific)
    api_key: str = ""  # API key (if needed)
    base_url: str = ""  # Custom endpoint (Ollama, etc.)
    temperature: float = 0.7
    max_tokens: int = 32000  # raised 4096 → 32000 per "remove all limits" — Claude Opus extended output
    reasoning_effort: str = "medium"  # minimal | low | medium | high
    # Explicit opt-in only. Leaving this empty means "primary only — if it
    # fails, tell the user and stop." Silently rerouting to a provider the
    # user didn't configure (and might not have credentials for) produces
    # confusing errors like "No API key for provider X" when X isn't even
    # the provider the user picked in setup.
    fallback_providers: list[str] = field(default_factory=list)
    auto_fallback: bool = False  # OFF = ask user to switch on rate limit, no silent fallback
    # Adaptive throttle settings (prevents 429s during rapid tool loops)
    throttle_ramp_after: int = 3  # start throttling after N rapid calls
    throttle_min_delay: float = 1.0  # initial delay (seconds)
    throttle_max_delay: float = 10.0  # maximum delay (seconds)
    throttle_rapid_window: float = 5.0  # seconds — calls within this = "rapid"
    # Retry/backoff settings (handles 429s when they occur)
    max_retries: int = 3  # max retry attempts on 429/529
    retry_backoff_base: float = 2.0  # exponential backoff base (2^attempt)
    retry_jitter: bool = True  # randomize delay to prevent thundering herd
    retry_max_wait: float = 60.0  # max wait between retries (seconds)


@dataclass
class AgentLLMConfig(LLMConfig):
    """Dedicated cognition provider settings for agent orchestration."""

    provider: str = ""  # Blank = inherit from main llm config
    model: str = ""
    fallback_models: list[str] = field(default_factory=list)
    fallback_providers: list[str] = field(default_factory=list)


@dataclass
class ChannelConfig:
    """Messaging channel settings."""

    enabled: list[str] = field(default_factory=lambda: ["webchat"])
    telegram_token: str = ""
    discord_token: str = ""
    whatsapp_token: str = ""
    webchat_port: int = 3000


@dataclass
class SecurityConfig:
    """Security and ethical governance settings."""

    trust_level: str = "normal"  # yolo | normal | paranoid
    permission_mode: str = "auto"  # auto | ask | deny — tool approval mode
    approval_timeout: int = 30  # seconds to wait for user approval
    remember_approvals: bool = True  # persist approval decisions
    docker_sandbox: bool = False  # Use Docker for code execution
    allowed_tools: list[str] = field(default_factory=list)  # Empty = all allowed
    blocked_tools: list[str] = field(default_factory=list)
    max_concurrent_tasks: int = 50       # raised 5 → 50 per "remove all limits"
    task_timeout_seconds: int = 3600     # raised 300 → 3600 per "remove all limits"


@dataclass
class DaemonConfig:
    """24/7 daemon mode settings."""

    enabled: bool = False
    cron_file: str = ""  # Path to cron definitions
    webhook_port: int = 8765
    webhook_secret: str = ""
    heartbeat_interval: int = 30
    db_socket_path: str = ""  # Unix socket for DB service (default: ~/.predacore/db.sock)


@dataclass
class LaunchProfileConfig:
    """Runtime profile toggles. Two modes: `enterprise` (safe-by-default, strict
    governance) and `beast` (yolo, autonomous, plugin-open). Resource limits are
    maxed on both modes — the split is about governance posture, not capacity."""

    profile: str = DEFAULT_PROFILE  # enterprise | beast
    approvals_required: bool = True  # User-toggleable on both modes (--approvals/--no-approvals, or PREDACORE_APPROVALS_REQUIRED)
    egm_mode: str = "strict"  # off | log_only | strict
    default_code_network: bool = False
    enable_openclaw_bridge: bool = False
    enable_plugin_marketplace: bool = False
    enable_self_evolution: bool = False
    max_spawn_depth: int = 16  # Agent recursion cap — Python stack allows up to ~20 safely
    max_spawn_fanout: int = 64  # Max concurrent child agents
    max_tool_iterations: int = 1000  # Tool loop cap — meta_cognition detects runaway earlier
    enable_persona_drift_guard: bool = True
    persona_drift_threshold: float = 0.32
    persona_drift_max_regens: int = 5


@dataclass
class OpenClawBridgeConfig:
    """OpenClaw bridge endpoint settings for delegated execution."""

    base_url: str = ""  # e.g. https://bridge.example.com
    task_path: str = "/v1/responses"  # Relative path on bridge endpoint
    status_path: str = "/v1/tasks/{task_id}"  # Legacy async status endpoint template
    model: str = "openclaw"  # OpenResponses model/agent selector
    agent_id: str = "main"  # Optional x-openclaw-agent-id header
    api_key: str = ""  # Optional bearer token
    timeout_seconds: int = 180
    verify_tls: bool = True
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    poll_interval_seconds: float = 1.5
    max_poll_seconds: int = 180
    skills_dir: str = ""  # Optional local OpenClaw skills import path
    auto_import_skills: bool = True  # Import OpenClaw SKILL.md entries into marketplace


# Resource limits are maxed and identical on both modes. Modes differ only on
# governance posture (trust, approvals, self-evolution, compliance). Users can
# toggle approvals independently via --approvals/--no-approvals or the
# PREDACORE_APPROVALS_REQUIRED env var.
_MAXED_RESOURCES_SECURITY = {
    "docker_sandbox": True,
    "max_concurrent_tasks": 100,
    "task_timeout_seconds": 3600,
}
_MAXED_RESOURCES_LAUNCH = {
    "max_spawn_depth": 16,
    "max_spawn_fanout": 64,
    "max_tool_iterations": 1000,
    "enable_persona_drift_guard": True,
    "persona_drift_max_regens": 5,
}

PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "enterprise": {
        "channels": {"enabled": ["webchat"]},
        "security": {
            "trust_level": "normal",
            **_MAXED_RESOURCES_SECURITY,
        },
        "launch": {
            "profile": "enterprise",
            "approvals_required": True,
            "egm_mode": "strict",
            "default_code_network": False,
            "enable_openclaw_bridge": False,
            "enable_plugin_marketplace": False,
            "enable_self_evolution": False,
            "persona_drift_threshold": 0.32,
            **_MAXED_RESOURCES_LAUNCH,
        },
    },
    "beast": {
        "channels": {"enabled": ["webchat"]},
        "security": {
            "trust_level": "yolo",
            **_MAXED_RESOURCES_SECURITY,
        },
        "launch": {
            "profile": "beast",
            "approvals_required": False,
            "egm_mode": "off",
            "default_code_network": True,
            "enable_openclaw_bridge": True,
            "enable_plugin_marketplace": True,
            "enable_self_evolution": True,
            "persona_drift_threshold": 0.60,
            **_MAXED_RESOURCES_LAUNCH,
        },
    },
}


@dataclass
class MemoryConfig:
    """Memory and knowledge settings."""

    persistence_dir: str = ""  # Auto-set to DEFAULT_MEMORY_DIR
    enable_knowledge_graph: bool = True
    enable_vector_store: bool = True
    working_memory_capacity: int = 7
    decay_rate: float = 0.01
    consolidation_threshold: int = 3
    # Phase B+C — features ported from lab MCP. All default ON: matches the
    # lab's mcp_server.py behavior so predacore-public agents get the same
    # memory hygiene by default. Disable per-flag via PREDACORE_MEMORY_*.
    enable_healer: bool = True
    scan_for_secrets: bool = True
    eager_warmup: bool = True


@dataclass
class OperatorsConfig:
    """Desktop and mobile operator settings."""

    # Desktop macro limits
    macro_max_steps: int = 50
    macro_max_depth: int = 3

    # AX tree traversal
    ax_default_depth: int = 4
    ax_max_depth: int = 8
    ax_default_children: int = 25
    ax_max_children: int = 100

    # Screenshot
    screenshot_max_b64_bytes: int = 10_000_000  # 10MB

    # Input limits
    sleep_max_seconds: float = 60.0
    scroll_max_amount: int = 100

    # Android
    android_macro_max_steps: int = 50
    android_screen_record_max_seconds: int = 180


@dataclass
class PredaCoreConfig:
    """Master configuration for PredaCore."""

    # ── Identity ──
    name: str = "PredaCore"
    version: str = "0.1.0"
    agent: str = DEFAULT_AGENT  # Active agent (folder name under ~/.predacore/agents/)

    # ── Paths ──
    home_dir: str = ""
    sessions_dir: str = ""
    skills_dir: str = ""
    logs_dir: str = ""

    # ── Sub-configs ──
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent_llm: AgentLLMConfig = field(default_factory=AgentLLMConfig)
    channels: ChannelConfig = field(default_factory=ChannelConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    launch: LaunchProfileConfig = field(default_factory=LaunchProfileConfig)
    openclaw: OpenClawBridgeConfig = field(default_factory=OpenClawBridgeConfig)
    operators: OperatorsConfig = field(default_factory=OperatorsConfig)

    # ── MCP (Model Context Protocol) servers ──
    # Each entry: {"name": "...", "command": [...] | "...", "env": {}, "cwd": "...", "disabled": false}
    # The daemon spawns each server on startup, enumerates its tools, and
    # mounts them into HANDLER_MAP as ``mcp_<server>_<tool>`` so the LLM
    # sees them as first-class tools. Manage via the mcp_add / mcp_remove /
    # mcp_restart / mcp_list tools in chat, or edit config.yaml directly.
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)

    @property
    def agents_dir(self) -> str:
        """Path to the agents directory (~/.predacore/agents/)."""
        return str(Path(self.home_dir) / "agents")

    @property
    def agent_dir(self) -> str:
        """Path to the active agent's directory (~/.predacore/agents/{agent}/)."""
        return str(Path(self.home_dir) / "agents" / self.agent)

    @property
    def flame_dir(self) -> str:
        """Path to the Flame skill network directory (~/.predacore/flame/)."""
        return str(Path(self.home_dir) / "flame")

    def __post_init__(self):
        """Set computed defaults after init."""
        if not self.home_dir:
            self.home_dir = str(DEFAULT_HOME)
        if not self.sessions_dir:
            self.sessions_dir = str(Path(self.home_dir) / "sessions")
        if not self.skills_dir:
            self.skills_dir = str(Path(self.home_dir) / "skills")
        if not self.logs_dir:
            self.logs_dir = str(Path(self.home_dir) / "logs")
        if not self.memory.persistence_dir:
            self.memory.persistence_dir = str(Path(self.home_dir) / "memory")
        if not self.daemon.db_socket_path:
            self.daemon.db_socket_path = str(Path(self.home_dir) / "db.sock")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_bool(raw: str) -> bool:
    """Parse env bool values consistently."""
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def _parse_csv(raw: str) -> list[str]:
    """Parse comma-separated env values into a normalized list."""
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _safe_int(raw: str, default: int = 0) -> int:
    """Parse int from env value with safe fallback."""
    try:
        return int(raw)
    except (ValueError, TypeError):
        logger.warning("Invalid integer value '%s'; using default %d", raw, default)
        return default


def _safe_float(raw: str, default: float = 0.0) -> float:
    """Parse float from env value with safe fallback."""
    try:
        return float(raw)
    except (ValueError, TypeError):
        logger.warning("Invalid float value '%s'; using default %s", raw, default)
        return default


def _validate_port(port: int) -> int:
    """Clamp port to valid TCP range (1–65535)."""
    if port < 1 or port > 65535:
        logger.warning("Port %d out of range (1–65535); clamping", port)
        return max(1, min(port, 65535))
    return port


def _resolve_profile_name(merged: dict[str, Any], profile_override: str | None) -> str:
    """Resolve requested launch profile from override or merged config."""
    if profile_override:
        return profile_override.strip()
    launch = merged.get("launch", {})
    if isinstance(launch, dict):
        candidate = launch.get("profile")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return DEFAULT_PROFILE


def _get_profile_defaults(profile_name: str) -> dict[str, Any]:
    """Return launch profile defaults with safe fallback."""
    if profile_name in PROFILE_PRESETS:
        return PROFILE_PRESETS[profile_name]
    logger.warning(
        "Unknown launch profile '%s'; falling back to '%s'",
        profile_name,
        DEFAULT_PROFILE,
    )
    return PROFILE_PRESETS[DEFAULT_PROFILE]


def _sync_runtime_policy_env(cfg: PredaCoreConfig) -> None:
    """
    Mirror config launch policy into runtime env consumed by service modules.
    Keeps policy deterministic regardless of launcher entrypoint.
    """
    os.environ["PREDACORE_PROFILE"] = cfg.launch.profile
    os.environ["APPROVALS_REQUIRED"] = "1" if cfg.launch.approvals_required else "0"
    os.environ["EGM_MODE"] = str(cfg.launch.egm_mode) if cfg.launch.egm_mode else "off"
    os.environ["DEFAULT_CODE_NETWORK"] = "1" if cfg.launch.default_code_network else "0"
    os.environ["MAX_TOOL_ITERATIONS"] = str(cfg.launch.max_tool_iterations)
    os.environ["PREDACORE_ENABLE_PERSONA_DRIFT_GUARD"] = (
        "1" if cfg.launch.enable_persona_drift_guard else "0"
    )
    os.environ["PREDACORE_PERSONA_DRIFT_THRESHOLD"] = str(
        cfg.launch.persona_drift_threshold
    )
    os.environ["PREDACORE_PERSONA_DRIFT_MAX_REGENS"] = str(
        cfg.launch.persona_drift_max_regens
    )


def _env_overrides() -> dict[str, Any]:
    """Read PREDACORE_* environment variables and map to config keys."""
    overrides: dict[str, Any] = {}

    env_map = {
        "PREDACORE_NAME": ("name",),
        "PREDACORE_HOME": ("home_dir",),
        "PREDACORE_AGENT": ("agent",),
        "PREDACORE_CHANNELS": ("channels", "enabled"),
        "PREDACORE_CHANNELS_ENABLED": ("channels", "enabled"),
        "PREDACORE_PROFILE": ("launch", "profile"),
        "PREDACORE_APPROVALS_REQUIRED": ("launch", "approvals_required"),
        "PREDACORE_EGM_MODE": ("launch", "egm_mode"),
        "PREDACORE_DEFAULT_CODE_NETWORK": ("launch", "default_code_network"),
        "PREDACORE_ENABLE_OPENCLAW_BRIDGE": ("launch", "enable_openclaw_bridge"),
        "PREDACORE_ENABLE_PLUGIN_MARKETPLACE": ("launch", "enable_plugin_marketplace"),
        "PREDACORE_ENABLE_SELF_EVOLUTION": ("launch", "enable_self_evolution"),
        "PREDACORE_MAX_SPAWN_DEPTH": ("launch", "max_spawn_depth"),
        "PREDACORE_MAX_SPAWN_FANOUT": ("launch", "max_spawn_fanout"),
        "PREDACORE_MAX_TOOL_ITERATIONS": ("launch", "max_tool_iterations"),
        "PREDACORE_ENABLE_PERSONA_DRIFT_GUARD": (
            "launch",
            "enable_persona_drift_guard",
        ),
        "PREDACORE_PERSONA_DRIFT_THRESHOLD": ("launch", "persona_drift_threshold"),
        "PREDACORE_PERSONA_DRIFT_MAX_REGENS": ("launch", "persona_drift_max_regens"),
        "PREDACORE_TRUST_LEVEL": ("security", "trust_level"),
        "OPENCLAW_BRIDGE_URL": ("openclaw", "base_url"),
        "OPENCLAW_BRIDGE_TASK_PATH": ("openclaw", "task_path"),
        "OPENCLAW_BRIDGE_STATUS_PATH": ("openclaw", "status_path"),
        "OPENCLAW_BRIDGE_MODEL": ("openclaw", "model"),
        "OPENCLAW_BRIDGE_AGENT_ID": ("openclaw", "agent_id"),
        "OPENCLAW_BRIDGE_API_KEY": ("openclaw", "api_key"),
        "OPENCLAW_BRIDGE_TIMEOUT": ("openclaw", "timeout_seconds"),
        "OPENCLAW_BRIDGE_VERIFY_TLS": ("openclaw", "verify_tls"),
        "OPENCLAW_BRIDGE_MAX_RETRIES": ("openclaw", "max_retries"),
        "OPENCLAW_BRIDGE_RETRY_BACKOFF": ("openclaw", "retry_backoff_seconds"),
        "OPENCLAW_BRIDGE_POLL_INTERVAL": ("openclaw", "poll_interval_seconds"),
        "OPENCLAW_BRIDGE_MAX_POLL_SECONDS": ("openclaw", "max_poll_seconds"),
        "OPENCLAW_SKILLS_DIR": ("openclaw", "skills_dir"),
        "OPENCLAW_AUTO_IMPORT_SKILLS": ("openclaw", "auto_import_skills"),
        "LLM_PROVIDER": ("llm", "provider"),
        "LLM_MODEL": ("llm", "model"),
        "LLM_FALLBACK_MODELS": ("llm", "fallback_models"),
        "LLM_FALLBACK_PROVIDERS": ("llm", "fallback_providers"),
        "LLM_TEMPERATURE": ("llm", "temperature"),
        "LLM_REASONING": ("llm", "reasoning_effort"),
        "PREDACORE_AGENT_LLM_PROVIDER": ("agent_llm", "provider"),
        "PREDACORE_AGENT_LLM_MODEL": ("agent_llm", "model"),
        "PREDACORE_AGENT_LLM_FALLBACK_MODELS": ("agent_llm", "fallback_models"),
        "PREDACORE_AGENT_LLM_FALLBACK_PROVIDERS": ("agent_llm", "fallback_providers"),
        "PREDACORE_AGENT_LLM_TEMPERATURE": ("agent_llm", "temperature"),
        "PREDACORE_AGENT_LLM_REASONING": ("agent_llm", "reasoning_effort"),
        # Provider API keys — only the active provider's key is applied.
        # Handled separately below via _PROVIDER_KEY_MAP.
        # "OPENAI_API_KEY": ("llm", "api_key"),  # etc.
        "OLLAMA_HOST": ("llm", "base_url"),
        "PREDACORE_MACRO_MAX_STEPS": ("operators", "macro_max_steps"),
        "PREDACORE_MACRO_MAX_DEPTH": ("operators", "macro_max_depth"),
        "PREDACORE_AX_DEFAULT_DEPTH": ("operators", "ax_default_depth"),
        "PREDACORE_AX_MAX_DEPTH": ("operators", "ax_max_depth"),
        "PREDACORE_AX_MAX_CHILDREN": ("operators", "ax_max_children"),
        "PREDACORE_SCREENSHOT_MAX_B64": ("operators", "screenshot_max_b64_bytes"),
        "PREDACORE_SLEEP_MAX_SECONDS": ("operators", "sleep_max_seconds"),
        "PREDACORE_SCROLL_MAX": ("operators", "scroll_max_amount"),
        "TELEGRAM_BOT_TOKEN": ("channels", "telegram_token"),
        "DISCORD_BOT_TOKEN": ("channels", "discord_token"),
        "WHATSAPP_TOKEN": ("channels", "whatsapp_token"),
        "WEBCHAT_PORT": ("channels", "webchat_port"),
        "WEBHOOK_PORT": ("daemon", "webhook_port"),
        "WEBHOOK_SECRET": ("daemon", "webhook_secret"),
        "PREDACORE_DB_SOCKET": ("daemon", "db_socket_path"),
        # Memory subsystem (Phase B+C — Healer, secret-scan, warmup)
        "PREDACORE_MEMORY_ENABLE_HEALER": ("memory", "enable_healer"),
        "PREDACORE_MEMORY_SCAN_SECRETS": ("memory", "scan_for_secrets"),
        "PREDACORE_MEMORY_EAGER_WARMUP": ("memory", "eager_warmup"),
    }

    for env_var, path in env_map.items():
        value = os.environ.get(env_var)
        if value is not None:
            # Build nested dict from path
            current = overrides
            for key in path[:-1]:
                current = current.setdefault(key, {})
            # Type coerce
            final_key = path[-1]
            if env_var in ("PREDACORE_CHANNELS", "PREDACORE_CHANNELS_ENABLED"):
                current[final_key] = _parse_csv(value)
            elif final_key in (
                "temperature",
                "retry_backoff_seconds",
                "poll_interval_seconds",
                "persona_drift_threshold",
            ):
                current[final_key] = _safe_float(value)
            elif final_key in (
                "webchat_port",
                "webhook_port",
                "max_spawn_depth",
                "max_spawn_fanout",
                "max_tool_iterations",
                "persona_drift_max_regens",
                "timeout_seconds",
                "max_retries",
                "max_poll_seconds",
            ):
                current[final_key] = _safe_int(value)
            elif final_key in (
                "approvals_required",
                "default_code_network",
                "enable_openclaw_bridge",
                "enable_plugin_marketplace",
                "enable_self_evolution",
                "enable_persona_drift_guard",
                "verify_tls",
                "auto_import_skills",
                "enable_healer",
                "scan_for_secrets",
                "eager_warmup",
            ):
                current[final_key] = _parse_bool(value)
            elif final_key in ("fallback_models", "fallback_providers"):
                current[final_key] = _parse_csv(value)
            else:
                current[final_key] = value

    # ── Provider-specific API key resolution ──
    # Only apply the key that matches the active provider, so keys don't
    # clobber each other (e.g. SAMBANOVA_API_KEY overwriting anthropic's key).
    _PROVIDER_KEY_MAP: dict[str, str] = {
        "anthropic": "ANTHROPIC_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "azure": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "xai": "XAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
        "together": "TOGETHER_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "sambanova": "SAMBANOVA_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        "nvidia": "NVIDIA_API_KEY",
    }
    active_provider = overrides.get("llm", {}).get("provider", "")
    provider_env_var = _PROVIDER_KEY_MAP.get(active_provider, "")
    if provider_env_var:
        key_val = os.environ.get(provider_env_var)
        if key_val:
            overrides.setdefault("llm", {})["api_key"] = key_val

    return overrides


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load YAML config file if it exists."""
    if not path.exists():
        return {}
    try:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        logger.info("Loaded config from %s", path)
        return data
    except ImportError:
        logger.warning("PyYAML not installed; skipping config file %s", path)
        return {}
    except (OSError, ValueError) as e:
        logger.warning("Failed to load config from %s: %s", path, e)
        return {}


def _dict_to_config(data: dict[str, Any]) -> PredaCoreConfig:
    """Convert a flat/nested dict to PredaCoreConfig with sub-dataclasses."""
    llm_data = data.pop("llm", {})
    agent_llm_data = data.pop("agent_llm", {})
    channels_data = data.pop("channels", {})
    security_data = data.pop("security", {})
    daemon_data = data.pop("daemon", {})
    memory_data = data.pop("memory", {})
    launch_data = data.pop("launch", {})
    openclaw_data = data.pop("openclaw", {})
    operators_data = data.pop("operators", {})

    return PredaCoreConfig(
        **{k: v for k, v in data.items() if k in PredaCoreConfig.__dataclass_fields__},
        llm=LLMConfig(
            **{k: v for k, v in llm_data.items() if k in LLMConfig.__dataclass_fields__}
        ),
        agent_llm=AgentLLMConfig(
            **{
                k: v
                for k, v in agent_llm_data.items()
                if k in AgentLLMConfig.__dataclass_fields__
            }
        ),
        channels=ChannelConfig(
            **{
                k: v
                for k, v in channels_data.items()
                if k in ChannelConfig.__dataclass_fields__
            }
        ),
        security=SecurityConfig(
            **{
                k: v
                for k, v in security_data.items()
                if k in SecurityConfig.__dataclass_fields__
            }
        ),
        daemon=DaemonConfig(
            **{
                k: v
                for k, v in daemon_data.items()
                if k in DaemonConfig.__dataclass_fields__
            }
        ),
        memory=MemoryConfig(
            **{
                k: v
                for k, v in memory_data.items()
                if k in MemoryConfig.__dataclass_fields__
            }
        ),
        launch=LaunchProfileConfig(
            **{
                k: v
                for k, v in launch_data.items()
                if k in LaunchProfileConfig.__dataclass_fields__
            }
        ),
        openclaw=OpenClawBridgeConfig(
            **{
                k: v
                for k, v in openclaw_data.items()
                if k in OpenClawBridgeConfig.__dataclass_fields__
            }
        ),
        operators=OperatorsConfig(
            **{
                k: v
                for k, v in operators_data.items()
                if k in OperatorsConfig.__dataclass_fields__
            }
        ),
    )


def load_config(
    config_path: str | None = None, profile_override: str | None = None
) -> PredaCoreConfig:
    """
    Load PredaCore configuration with layered precedence:
      defaults → config.yaml → environment variables
    """
    # 1) Start with empty dict (dataclass defaults apply in __post_init__)
    base: dict[str, Any] = {}

    # 2) Load YAML config
    yaml_path = Path(config_path) if config_path else DEFAULT_CONFIG_FILE
    yaml_data = _load_yaml_config(yaml_path)
    merged = _deep_merge(base, yaml_data)

    # 3) Apply environment variable overrides
    env_data = _env_overrides()
    merged = _deep_merge(merged, env_data)

    # 4) Apply launch profile defaults before object construction.
    profile_name = _resolve_profile_name(merged, profile_override)
    merged = _deep_merge(_get_profile_defaults(profile_name), merged)
    launch_section = merged.setdefault("launch", {})
    if isinstance(launch_section, dict):
        launch_section["profile"] = (
            profile_name if profile_name in PROFILE_PRESETS else DEFAULT_PROFILE
        )

    # 5) Convert to config object
    cfg = _dict_to_config(merged)

    # 6) Ensure directories exist
    for dir_path in [
        cfg.home_dir,
        cfg.sessions_dir,
        cfg.skills_dir,
        cfg.logs_dir,
        cfg.memory.persistence_dir,
        cfg.agent_dir,
        cfg.flame_dir,
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # 7) Warn if config file is world-readable (may contain API key refs)
    if yaml_path.exists():
        try:
            mode = yaml_path.stat().st_mode & 0o777
            if mode & 0o044:  # readable by group or others
                logger.warning(
                    "Config file %s has permissive permissions (%o). "
                    "Consider: chmod 600 %s",
                    yaml_path, mode, yaml_path,
                )
        except OSError:
            pass

    _sync_runtime_policy_env(cfg)

    logger.info(
        "PredaCore config loaded: profile=%s, trust=%s, approvals=%s, llm=%s",
        cfg.launch.profile,
        cfg.security.trust_level,
        "on" if cfg.launch.approvals_required else "off",
        cfg.llm.provider,
    )
    return cfg


def save_default_config(
    path: str | None = None,
    *,
    provider: str = "gemini-cli",
    trust_level: str | None = None,
    model: str = "",
    channels: list[str] | None = None,
    profile: str = DEFAULT_PROFILE,
) -> Path:
    """Write config.yaml for setup, applying user's choices.

    Args:
        path: Destination config path (defaults to DEFAULT_CONFIG_FILE).
        provider: LLM provider name.
        trust_level: yolo | normal. If None, derived from profile
            (enterprise→normal, beast→yolo).
        model: Optional explicit model name (empty = auto-detect).
        channels: List of channel names to enable. Defaults to ["webchat"].
        profile: Launch profile — enterprise | beast (default: enterprise).
    """
    # Derive trust_level from profile if not explicitly set.
    if trust_level is None:
        trust_level = "yolo" if profile == "beast" else "normal"
    target = Path(path) if path else DEFAULT_CONFIG_FILE
    target.parent.mkdir(parents=True, exist_ok=True)

    # Normalize channel list — webchat is always enabled (required for
    # `predacore chat` to attach its WebSocket client), dedupe, validate.
    enabled_channels = list(channels) if channels else ["webchat"]
    if "webchat" not in enabled_channels:
        enabled_channels.insert(0, "webchat")
    _known_channels = {"webchat", "telegram", "discord", "whatsapp", "signal", "imessage", "email", "slack"}
    enabled_channels = [c for c in enabled_channels if c in _known_channels]

    # Build the YAML channel list block
    _channel_lines = "\n".join(f"    - {c}" for c in enabled_channels)
    _commented_lines = "\n".join(
        f"    # - {c}" for c in _known_channels if c not in enabled_channels
    )
    _channels_block = _channel_lines
    if _commented_lines:
        _channels_block = f"{_channel_lines}\n{_commented_lines}"

    default_yaml = f"""\
# ──────────────────────────────────────────────────────────
# Prometheus Configuration
# ──────────────────────────────────────────────────────────

name: PredaCore
agent: default             # Active agent (folder under ~/.predacore/agents/)

llm:
  provider: {provider}  # openai | gemini | gemini-cli | anthropic | groq | xai
                        # | deepseek | cerebras | together | openrouter | sambanova
                        # | mistral | fireworks | nvidia | zhipu | ollama
  model: "{model}"      # auto-detect from provider if empty
  fallback_models: []   # optional ordered model fallbacks for the chosen provider
  fallback_providers: []  # optional. Leave empty to only use the primary.
                          # e.g. [gemini, openrouter] — rate limits on primary
                          # will ask you to switch rather than silently reroute.
  temperature: 0.7
  reasoning_effort: medium

agent_llm:
  provider: ""          # optional shared cognition provider for all agents / DAF workers
  model: ""
  fallback_providers: []

channels:
  enabled:
{_channels_block}

security:
  trust_level: {trust_level}   # yolo | normal
  # Every other security setting inherits from the launch profile preset.

daemon:
  enabled: false
  webhook_port: 8765

memory:
  enable_knowledge_graph: true
  enable_vector_store: true
  working_memory_capacity: 7

launch:
  profile: {profile}        # enterprise | beast — all other launch settings
                            # (approvals_required, egm_mode, self_evolution,
                            # code_network, etc.) cascade from the profile preset.
                            # Override any of them below only if you need to.

openclaw:
  base_url: ""              # set OPENCLAW_BRIDGE_URL for runtime
  task_path: /v1/responses
  status_path: /v1/tasks/{{task_id}}   # only used by legacy async task APIs
  model: openclaw
  agent_id: main
  timeout_seconds: 180
  verify_tls: true
  max_retries: 2
  retry_backoff_seconds: 1.0
  poll_interval_seconds: 1.5
  max_poll_seconds: 180
  skills_dir: ""            # optional override for installed openclaw skills
  auto_import_skills: true
"""
    target.write_text(default_yaml)
    # Restrict permissions — config may contain secrets (API keys, tokens)
    try:
        os.chmod(str(target), 0o600)
    except OSError:
        logger.warning("Could not set restrictive permissions on %s", target)
    logger.info("Default config written to %s", target)
    return target
