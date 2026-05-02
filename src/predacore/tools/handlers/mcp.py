"""
MCP handlers — dynamic tool registration + runtime management tools.

Two kinds of handlers live here:

1. **Per-MCP-tool closures.** When the registry mounts a discovered MCP
   tool, we synthesize a handler that calls the registry with that
   specific ``(server, tool)`` pair. Each mounted tool gets its own
   entry in ``HANDLER_MAP`` so the LLM-facing tool list and the executor
   path stay uniform — no meta-``mcp_call`` dispatch needed.

2. **Management handlers.** ``mcp_list``, ``mcp_add``, ``mcp_remove``,
   ``mcp_restart`` let the agent (or a user in chat) manage MCP servers
   conversationally, just like ``channel_configure`` / ``channel_install``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._context import (
    ToolContext,
    ToolError,
    ToolErrorKind,
    invalid_param,
    missing_param,
)

if TYPE_CHECKING:
    from ...services.mcp_registry import _MountedTool

logger = logging.getLogger(__name__)

# ── Per-tool handler synthesis ───────────────────────────────────────


def make_mcp_tool_handler(exposed_name: str) -> Callable[..., Any]:
    """Build a handler closure that calls the registry for this exact tool.

    The closure captures only the tool's *exposed name*, which is stable
    for the lifetime of the registration. Re-resolving the underlying
    ``(server, remote_name)`` on every call means unmount/remount works
    without leaking dangling closures.
    """
    async def _handler(args: dict[str, Any], ctx: ToolContext) -> str:
        from ...services.mcp_registry import get_registry

        registry = get_registry()
        try:
            result = await registry.call(exposed_name, args or {})
        except Exception as exc:
            raise ToolError(
                f"MCP call failed: {exc}",
                kind=ToolErrorKind.EXECUTION,
                tool_name=exposed_name,
            ) from exc
        return _format_mcp_result(result)
    _handler.__name__ = f"handle_{exposed_name}"
    _handler.__qualname__ = _handler.__name__
    return _handler


def make_mcp_tool_schema(mt: _MountedTool) -> dict[str, Any]:
    """Convert a mounted MCP tool into an LLM-facing tool-definition dict."""
    description = mt.tool.description.strip() or f"MCP tool from server '{mt.server}'"
    # Prepend the origin so the LLM knows this is a plugin tool (useful for
    # trust / consent decisions and for the agent to name the source).
    labelled = f"[mcp:{mt.server}] {description}"
    return {
        "name": mt.exposed_name,
        "description": labelled[:1024],
        "parameters": mt.tool.input_schema or {"type": "object"},
    }


def _format_mcp_result(result: dict[str, Any]) -> str:
    """Flatten MCP's CallToolResult into a string the LLM can consume."""
    is_error = bool(result.get("isError", False))
    content = result.get("content") or []
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        t = block.get("type", "")
        if t == "text":
            parts.append(str(block.get("text", "")))
        elif t == "image":
            parts.append(f"[image:{block.get('mimeType', 'unknown')}]")
        elif t == "resource":
            parts.append(f"[resource:{block.get('uri', '')}]")
        else:
            parts.append(f"[{t}]")
    body = "\n".join(p for p in parts if p).strip()
    if not body:
        body = "(no content)"
    return ("[mcp error] " + body) if is_error else body


# ── Management handlers ───────────────────────────────────────────────

# Match npm and pip package names. Prevents shell-injection via the
# `pip install` / `npm install -g` path in `mcp_add`.
_NPM_NAME_RE = re.compile(r"^(@[a-z0-9][\w._-]*/)?[a-z0-9][\w._-]*$", re.IGNORECASE)
_PIP_NAME_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?"
    r"([<>=!~]=?\s*[A-Za-z0-9._*+-]+)?$"
)


async def _run_installer(argv: list[str], *, label: str, timeout_s: float = 300.0) -> None:
    """Run a package-install command asynchronously, freeing the event loop.

    Previous versions called ``subprocess.run`` synchronously inside an
    async handler, which pinned the whole daemon for the entire install
    (up to 5 min on a slow network). With a real async subprocess, other
    channels, MCP servers, and background tasks stay responsive during
    pip/npm install.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        raise ToolError(
            f"{label} timed out after {int(timeout_s)}s",
            kind=ToolErrorKind.TIMEOUT,
            tool_name="mcp_add",
        ) from None
    except FileNotFoundError as exc:
        raise ToolError(
            f"{label}: executable not found ({exc})",
            kind=ToolErrorKind.UNAVAILABLE,
            tool_name="mcp_add",
        ) from exc
    if proc.returncode != 0:
        tail = (stderr.decode("utf-8", "replace") or stdout.decode("utf-8", "replace"))[-300:]
        raise ToolError(
            f"{label} failed (rc={proc.returncode}): {tail.strip()}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="mcp_add",
        )


async def handle_mcp_list(args: dict[str, Any], ctx: ToolContext) -> str:
    """Show every configured MCP server and the tools it currently exposes."""
    from ...services.mcp_registry import get_registry

    registry = get_registry()
    servers = registry.list_servers()
    return json.dumps(
        {"status": "ok", "count": len(servers), "servers": servers},
        indent=2,
    )


async def handle_mcp_add(args: dict[str, Any], ctx: ToolContext) -> str:
    """Add a new MCP server (and optionally install its backing package).

    Args:
        name: short identifier (e.g. ``"filesystem"``). Used in the exposed
            tool prefix ``mcp_<name>_<tool>``.
        command: argv list or shell string that launches the server.
        env: optional dict of env vars to set for the subprocess. ``${VAR}``
            placeholders are expanded against the main process env.
        install: optional dict ``{"npm": "pkg"}`` or ``{"pip": "pkg"}`` — if
            set, we run the appropriate installer first so the subsequent
            ``command`` can actually resolve.
        persist: if true (default), append this spec to ``config.yaml`` so
            it survives restarts. Set false for ephemeral experiments.
    """
    from ...services.mcp_registry import MCPServerSpec, get_registry

    name = (args.get("name") or "").strip()
    if not name:
        raise missing_param("name", tool="mcp_add")
    command = args.get("command")
    if not command:
        raise missing_param("command", tool="mcp_add")

    # Optional install step — the reason mcp_add is separate from just
    # appending config. Many MCP servers ship as `npx …` invocations of
    # packages the user hasn't installed globally yet.
    install = args.get("install") or {}
    if isinstance(install, dict):
        # mcp_add is in WRITE_TOOLS — the dispatcher confirms it via the
        # ask_everytime policy. No additional handler-side trust gate.
        if "npm" in install:
            pkg = str(install["npm"])
            if not _NPM_NAME_RE.match(pkg):
                raise invalid_param("install.npm", f"invalid npm name: {pkg!r}", tool="mcp_add")
            if not shutil.which("npm"):
                raise ToolError(
                    "npm not in PATH — install Node.js first.",
                    kind=ToolErrorKind.UNAVAILABLE,
                    tool_name="mcp_add",
                )
            await _run_installer(
                ["npm", "install", "-g", pkg],
                label=f"npm install -g {pkg}",
            )
        if "pip" in install:
            pkg = str(install["pip"])
            if not _PIP_NAME_RE.match(pkg):
                raise invalid_param("install.pip", f"invalid pip spec: {pkg!r}", tool="mcp_add")
            await _run_installer(
                [sys.executable, "-m", "pip", "install", "--no-input", pkg],
                label=f"pip install {pkg}",
            )

    spec_dict: dict[str, Any] = {
        "name": name,
        "command": command,
    }
    if "env" in args and isinstance(args["env"], dict):
        spec_dict["env"] = args["env"]
    if "cwd" in args:
        spec_dict["cwd"] = str(args["cwd"])
    if "description" in args:
        spec_dict["description"] = str(args["description"])

    try:
        spec = MCPServerSpec.from_dict(spec_dict)
    except ValueError as exc:
        raise invalid_param("command", str(exc), tool="mcp_add") from exc

    registry = get_registry()
    result = await registry.add_server(spec)
    if result.get("status") != "ok":
        raise ToolError(
            str(result.get("error", "mcp_add failed")),
            kind=ToolErrorKind.EXECUTION,
            tool_name="mcp_add",
        )

    if args.get("persist", True):
        home = Path(getattr(ctx.config, "home_dir", Path.home() / ".predacore")).expanduser()
        _persist_server(home, spec_dict)
        result["persisted"] = True

    result["note"] = (
        "Tools are live now; they'll also load on daemon restart."
    )
    return json.dumps(result, indent=2)


async def handle_mcp_remove(args: dict[str, Any], ctx: ToolContext) -> str:
    """Tear down an MCP server and drop it from ``config.yaml``."""
    from ...services.mcp_registry import get_registry

    name = (args.get("name") or "").strip()
    if not name:
        raise missing_param("name", tool="mcp_remove")

    registry = get_registry()
    result = await registry.remove_server(name)
    if result.get("status") != "ok":
        raise ToolError(
            str(result.get("error", f"mcp_remove {name} failed")),
            kind=ToolErrorKind.NOT_FOUND,
            tool_name="mcp_remove",
        )

    if args.get("persist", True):
        home = Path(getattr(ctx.config, "home_dir", Path.home() / ".predacore")).expanduser()
        _unpersist_server(home, name)
        result["persisted"] = True

    return json.dumps(result, indent=2)


async def handle_mcp_restart(args: dict[str, Any], ctx: ToolContext) -> str:
    """Bounce a running MCP server (useful after a config or token change)."""
    from ...services.mcp_registry import get_registry

    name = (args.get("name") or "").strip()
    if not name:
        raise missing_param("name", tool="mcp_restart")

    registry = get_registry()
    result = await registry.restart(name)
    if result.get("status") != "ok":
        raise ToolError(
            str(result.get("error", f"mcp_restart {name} failed")),
            kind=ToolErrorKind.EXECUTION,
            tool_name="mcp_restart",
        )
    return json.dumps(result, indent=2)


# ── config.yaml writers (atomic + chmod-preserving) ──────────────────


def _load_config_yaml(home: Path) -> dict[str, Any]:
    import yaml
    path = home / "config.yaml"
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _save_config_yaml(home: Path, data: dict[str, Any]) -> None:
    import yaml
    path = home / "config.yaml"
    home.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(home), suffix=".yaml.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _persist_server(home: Path, spec_dict: dict[str, Any]) -> None:
    data = _load_config_yaml(home)
    servers = data.get("mcp_servers")
    if not isinstance(servers, list):
        servers = []
    # Remove any prior entry with the same name before appending.
    servers = [s for s in servers if not (isinstance(s, dict) and s.get("name") == spec_dict["name"])]
    servers.append(spec_dict)
    data["mcp_servers"] = servers
    _save_config_yaml(home, data)


def _unpersist_server(home: Path, name: str) -> None:
    data = _load_config_yaml(home)
    servers = data.get("mcp_servers")
    if not isinstance(servers, list):
        return
    before = len(servers)
    data["mcp_servers"] = [
        s for s in servers if not (isinstance(s, dict) and s.get("name") == name)
    ]
    if len(data["mcp_servers"]) != before:
        _save_config_yaml(home, data)
