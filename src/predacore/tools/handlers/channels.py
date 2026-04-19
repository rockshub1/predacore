"""Channel management handlers — configure + install channels at runtime.

These tools let the agent (or a user via slash command) manage channels
without leaving the chat:

- ``channel_configure`` — enable / disable / set-token / status for an
  already-discoverable channel (built-in or installed plugin).
- ``channel_install`` — ``pip install`` a third-party channel package, then
  rescan the registry so the new adapter is immediately usable.

Both tools go through the ``ChannelRegistry`` (see ``channels/registry.py``)
so they work uniformly for built-in channels, third-party plugins, and
user-local ``~/.predacore/channels/*.py`` adapters.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

from ._context import (
    ToolContext,
    ToolError,
    ToolErrorKind,
    invalid_param,
    missing_param,
)

# Map channel name -> (.env var, config.yaml field)
# Only channels that need a secret appear here. Webchat doesn't need one.
_SECRET_MAP: dict[str, tuple[str, str]] = {
    "telegram": ("TELEGRAM_BOT_TOKEN", "telegram_token"),
    "discord":  ("DISCORD_BOT_TOKEN",  "discord_token"),
    "whatsapp": ("WHATSAPP_TOKEN",     "whatsapp_token"),
}

# Regex for safe package names — rejects shell-injection attempts in
# channel_install. Matches PEP 503 + optional version specifier / extras.
_PKG_NAME_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?"
    r"([<>=!~]=?\s*[A-Za-z0-9._*+-]+)?$"
)

# Known secret env-var names — LLM providers + channel tokens.
# ``secret_set`` accepts any name on this list verbatim, plus any name
# matching ``*_API_KEY`` / ``*_TOKEN`` / ``*_SECRET`` at length <= 64.
# This keeps the attack surface small: the agent can't write arbitrary
# env vars via prompt injection, only recognized-shape secrets.
_KNOWN_SECRETS: frozenset[str] = frozenset({
    # LLM providers
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "GROQ_API_KEY", "DEEPSEEK_API_KEY", "SAMBANOVA_API_KEY", "MISTRAL_API_KEY",
    "XAI_API_KEY", "COHERE_API_KEY", "TOGETHER_API_KEY", "PERPLEXITY_API_KEY",
    "OPENROUTER_API_KEY",
    # Channels (also writable via channel_configure)
    "TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "WHATSAPP_TOKEN", "WHATSAPP_APP_SECRET",
    "SLACK_APP_TOKEN", "SLACK_BOT_TOKEN",
    "SIGNAL_API_URL", "SIGNAL_NUMBER",
    "EMAIL_USERNAME", "EMAIL_PASSWORD", "EMAIL_FROM_ADDRESS",
    "IMAP_HOST", "IMAP_PORT", "SMTP_HOST", "SMTP_PORT",
    # Auth
    "PREDACORE_JWT_SECRET",
})
_SECRET_SHAPE_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}_(?:API_KEY|TOKEN|SECRET)$")


# ── channel_configure ─────────────────────────────────────────────────


async def handle_channel_configure(args: dict[str, Any], ctx: ToolContext) -> str:
    """Enable / disable / set-token / status for a channel.

    Args:
        action: one of ``"add"``, ``"remove"``, ``"set_token"``, ``"status"``.
        channel: required for all actions except ``"status"``.
        token: required for ``"add"`` if the channel needs a secret, and
            always required for ``"set_token"``. Ignored for ``"remove"``.
    """
    from ...channels.registry import get_registry

    action = (args.get("action") or "").strip().lower()
    if not action:
        raise missing_param("action", tool="channel_configure")
    if action not in ("add", "remove", "set_token", "status"):
        raise invalid_param(
            "action",
            f"unknown action: {action!r} "
            "(valid: add, remove, set_token, status)",
            tool="channel_configure",
        )

    home = Path(getattr(ctx.config, "home_dir", Path.home() / ".predacore")).expanduser()
    registry = get_registry(home)
    registry.scan()

    if action == "status":
        return _channel_status(ctx, registry, home)

    channel = (args.get("channel") or "").strip().lower()
    if not channel:
        raise missing_param("channel", tool="channel_configure")
    if not registry.has(channel):
        raise ToolError(
            f"No adapter registered for channel {channel!r}. "
            f"Available: {', '.join(registry.available()) or 'none'}. "
            "Install a new channel type with `channel_install`.",
            kind=ToolErrorKind.NOT_FOUND,
            tool_name="channel_configure",
            suggestion="channel_install package=predacore-<name>",
        )

    if action == "add":
        token = (args.get("token") or "").strip()
        secret_env = _SECRET_MAP.get(channel)
        if secret_env is not None and not token:
            raise missing_param(
                "token",
                tool="channel_configure",
            )
        result = _add_channel(ctx, channel, token, home)
        return json.dumps(result, indent=2)

    if action == "remove":
        result = _remove_channel(ctx, channel, home)
        return json.dumps(result, indent=2)

    if action == "set_token":
        token = (args.get("token") or "").strip()
        if not token:
            raise missing_param("token", tool="channel_configure")
        if channel not in _SECRET_MAP:
            raise invalid_param(
                "channel",
                f"Channel {channel!r} does not take a token.",
                tool="channel_configure",
            )
        result = _set_token(channel, token, home)
        return json.dumps(result, indent=2)

    raise ToolError(  # unreachable — kept for linter completeness
        "unknown action",
        kind=ToolErrorKind.INVALID_PARAM,
        tool_name="channel_configure",
    )


def _channel_status(ctx: ToolContext, registry, home: Path) -> str:
    enabled = set(getattr(ctx.config.channels, "enabled", []) or [])
    env_path = home / ".env"
    env_content = env_path.read_text() if env_path.exists() else ""

    rows = []
    for entry in registry.describe():
        name = entry["name"]
        token_var = _SECRET_MAP.get(name, (None, None))[0]
        token_present = bool(token_var) and (
            bool(os.environ.get(token_var)) or f"{token_var}=" in env_content
        )
        rows.append({
            "channel": name,
            "source": entry["source"],
            "enabled": name in enabled,
            "token_set": token_present if token_var else None,  # None = no token needed
        })
    return json.dumps({"status": "ok", "channels": rows}, indent=2)


def _add_channel(
    ctx: ToolContext, channel: str, token: str, home: Path,
) -> dict[str, Any]:
    """Enable a channel + write its token to ~/.predacore/.env."""
    actions: list[str] = []

    if token and channel in _SECRET_MAP:
        env_var, _ = _SECRET_MAP[channel]
        _write_env_secret(home, env_var, token)
        os.environ[env_var] = token  # live process sees the new value immediately
        actions.append(f"wrote {env_var} to {home/'.env'}")

    enabled = list(getattr(ctx.config.channels, "enabled", []) or [])
    if channel not in enabled:
        enabled.append(channel)
        _update_config_enabled_channels(home, enabled)
        ctx.config.channels.enabled = enabled
        actions.append(f"enabled {channel} in config.yaml")

    actions.append("daemon restart required to pick up the new channel")
    return {"status": "ok", "channel": channel, "actions": actions}


def _remove_channel(
    ctx: ToolContext, channel: str, home: Path,
) -> dict[str, Any]:
    """Disable a channel (config only — secrets stay unless explicitly cleared)."""
    enabled = list(getattr(ctx.config.channels, "enabled", []) or [])
    if channel in enabled:
        enabled.remove(channel)
        _update_config_enabled_channels(home, enabled)
        ctx.config.channels.enabled = enabled
        return {
            "status": "ok",
            "channel": channel,
            "actions": [
                f"disabled {channel} in config.yaml",
                "(token left in .env — use `set_token` with empty value to clear)",
                "daemon restart required to stop the adapter",
            ],
        }
    return {
        "status": "ok",
        "channel": channel,
        "actions": [f"{channel} was not enabled"],
    }


def _set_token(channel: str, token: str, home: Path) -> dict[str, Any]:
    env_var, _ = _SECRET_MAP[channel]
    _write_env_secret(home, env_var, token)
    os.environ[env_var] = token
    return {
        "status": "ok",
        "channel": channel,
        "actions": [f"updated {env_var} in {home/'.env'}"],
    }


# ── channel_install ───────────────────────────────────────────────────


async def handle_channel_install(args: dict[str, Any], ctx: ToolContext) -> str:
    """``pip install`` a third-party channel package, then rescan the registry.

    Trust policy:
    - yolo     → runs without prompting
    - normal   → approval via the standard tool-confirm path (dispatched
                 by the executor, not here)
    - paranoid → blocked unless the user enables it in config

    Args:
        package: name of the pypi package (e.g. ``predacore-slack``).
        upgrade: if true, passes ``--upgrade`` to pip. Default false.
    """
    from ...channels.registry import get_registry

    package = (args.get("package") or "").strip()
    if not package:
        raise missing_param("package", tool="channel_install")
    if not _PKG_NAME_RE.match(package):
        raise invalid_param(
            "package",
            f"rejected unsafe package spec: {package!r} "
            "(allowed: a PEP 503 name with optional version/extras)",
            tool="channel_install",
        )

    trust = getattr(ctx.config.security, "trust_level", "normal")
    if trust == "paranoid":
        raise ToolError(
            "channel_install is disabled in paranoid trust mode. "
            "Install with `pip install " + package + "` manually and restart the daemon.",
            kind=ToolErrorKind.BLOCKED,
            tool_name="channel_install",
            suggestion="Lower trust to normal/yolo, or install manually.",
        )

    upgrade = bool(args.get("upgrade", False))

    cmd = [sys.executable, "-m", "pip", "install", "--no-input"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append(package)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
    except asyncio.TimeoutError:
        raise ToolError(
            f"pip install {package} timed out after 180s",
            kind=ToolErrorKind.TIMEOUT,
            tool_name="channel_install",
        ) from None
    except FileNotFoundError as exc:
        raise ToolError(
            f"Couldn't run pip: {exc}",
            kind=ToolErrorKind.UNAVAILABLE,
            tool_name="channel_install",
        ) from exc

    if proc.returncode != 0:
        tail = (stderr.decode("utf-8", "replace") or stdout.decode("utf-8", "replace"))[-400:]
        raise ToolError(
            f"pip install {package} failed (rc={proc.returncode}): {tail.strip()}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="channel_install",
        )

    # Refresh the registry so any new channel_name becomes visible.
    home = Path(getattr(ctx.config, "home_dir", Path.home() / ".predacore")).expanduser()
    registry = get_registry(home)
    registry.scan(refresh=True)

    return json.dumps({
        "status": "ok",
        "package": package,
        "channels_now_available": registry.available(),
        "next_step": (
            "Call channel_configure with action=\"add\" and the new "
            "channel's name + token to enable it."
        ),
    }, indent=2)


# ── secret_set / secret_list ──────────────────────────────────────────


def _is_allowed_secret_name(name: str) -> bool:
    """Allow known secret names + anything that looks like a secret env var."""
    if name in _KNOWN_SECRETS:
        return True
    return bool(_SECRET_SHAPE_RE.match(name))


async def handle_secret_set(args: dict[str, Any], ctx: ToolContext) -> str:
    """Write a named secret to ``~/.predacore/.env`` (chmod 600).

    Args:
        name: env var name (e.g. ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``,
            ``TELEGRAM_BOT_TOKEN``). Must either be a recognized provider/channel
            secret or match ``*_API_KEY`` / ``*_TOKEN`` / ``*_SECRET`` shape.
        value: the secret value. Required.

    The new value is also set in ``os.environ`` so the running process picks
    it up immediately — the daemon does not need to restart.
    """
    name = (args.get("name") or "").strip()
    value = args.get("value") or ""
    if not name:
        raise missing_param("name", tool="secret_set")
    if not value:
        raise missing_param("value", tool="secret_set")

    if not _is_allowed_secret_name(name):
        raise invalid_param(
            "name",
            f"refused to write unfamiliar env var {name!r}. "
            "Allowed: recognized provider/channel secrets, or any "
            "*_API_KEY / *_TOKEN / *_SECRET name (uppercase, ≤64 chars).",
            tool="secret_set",
        )

    # Soft trust check — writing secrets is sensitive.
    trust = getattr(ctx.config.security, "trust_level", "normal")
    if trust == "paranoid":
        raise ToolError(
            f"secret_set is disabled in paranoid trust mode "
            f"(would write {name} to ~/.predacore/.env).",
            kind=ToolErrorKind.BLOCKED,
            tool_name="secret_set",
            suggestion="Add the secret manually, or lower trust to normal/yolo.",
        )

    home = Path(getattr(ctx.config, "home_dir", Path.home() / ".predacore")).expanduser()
    _write_env_secret(home, name, value)
    os.environ[name] = value  # live — LLM/channel picks it up on next use

    return json.dumps({
        "status": "ok",
        "name": name,
        "actions": [f"wrote {name} to {home/'.env'} (chmod 600)"],
        "live": "env var is active in the running process — no restart needed",
    }, indent=2)


async def handle_secret_list(args: dict[str, Any], ctx: ToolContext) -> str:
    """List which known secrets are currently set (names only, values redacted)."""
    home = Path(getattr(ctx.config, "home_dir", Path.home() / ".predacore")).expanduser()
    env_path = home / ".env"
    env_content = env_path.read_text(encoding="utf-8") if env_path.exists() else ""

    seen: list[dict[str, Any]] = []
    for var in sorted(_KNOWN_SECRETS):
        in_env = bool(os.environ.get(var))
        in_file = f"{var}=" in env_content
        if in_env or in_file:
            seen.append({
                "name": var,
                "in_env_file": in_file,
                "in_process_env": in_env,
            })

    return json.dumps({
        "status": "ok",
        "secrets_set": seen,
        "count": len(seen),
    }, indent=2)


# ── File-write helpers ────────────────────────────────────────────────


def _write_env_secret(home: Path, var: str, value: str) -> None:
    """Append or replace ``VAR=VALUE`` in ``~/.predacore/.env`` (chmod 600)."""
    home.mkdir(parents=True, exist_ok=True)
    env_path = home / ".env"
    lines: list[str] = []
    if env_path.exists():
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []

    replaced = False
    for i, line in enumerate(lines):
        if line.startswith(f"{var}="):
            lines[i] = f"{var}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{var}={value}")

    # Atomic write + chmod 600
    fd, tmp = tempfile.mkstemp(dir=str(home), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, str(env_path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.chmod(str(env_path), stat.S_IRUSR | stat.S_IWUSR)


def _update_config_enabled_channels(home: Path, enabled: list[str]) -> None:
    """Rewrite ``channels.enabled`` in ``~/.predacore/config.yaml``.

    Uses PyYAML so structure stays intact even if the file was hand-edited.
    """
    import yaml  # core dep — safe to import eagerly

    cfg_path = home / "config.yaml"
    home.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if cfg_path.exists():
        try:
            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except yaml.YAMLError:
            data = {}

    channels_section = data.get("channels")
    if not isinstance(channels_section, dict):
        channels_section = {}
        data["channels"] = channels_section
    channels_section["enabled"] = sorted(set(enabled))

    fd, tmp = tempfile.mkstemp(dir=str(home), suffix=".yaml.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
        os.replace(tmp, str(cfg_path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
