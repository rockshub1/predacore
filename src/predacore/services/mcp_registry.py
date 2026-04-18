"""
MCP server registry — lifecycle + tool discovery for configured MCP servers.

Given a list of server specs in ``config.yaml``, this module:

1. Spawns each server as an ``MCPClient`` subprocess.
2. Runs the ``tools/list`` handshake so we know what each server offers.
3. Registers every discovered tool into PredaCore's ``HANDLER_MAP`` under
   the name ``mcp_<server>_<tool>`` so the LLM sees them as first-class
   tools (no meta-``mcp_call`` dispatching required).
4. Generates matching tool-schema dicts for the LLM prompt, which the
   core injects into its ``_tool_definitions`` list on boot.
5. Handles shutdown: stops every client, clears registered tools.

Config shape (``config.yaml``):

    mcp_servers:
      - name: filesystem
        command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "$HOME"]
      - name: github
        command: ["npx", "-y", "@modelcontextprotocol/server-github"]
        env:
          GITHUB_PERSONAL_ACCESS_TOKEN: ${GITHUB_TOKEN}
      - name: my-local
        command: "python /opt/mcp/my_server.py"
        cwd: /opt/mcp

Anything the user adds via the ``mcp_add`` tool is appended here.

Naming convention for the exposed tool names:

    mcp_<server_slug>_<tool_name>

``server_slug`` is the config's ``name`` field, lowercased with non-word
characters turned into underscores, so ``"GitHub Repo"`` becomes
``"github_repo"``. ``tool_name`` is passed through verbatim unless it
contains characters that would confuse the LLM's tool-call parser, in
which case we normalize it the same way.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from .mcp_client import (
    MCPClient,
    MCPClientError,
    MCPTool,
    expand_env,
    parse_command,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_NAME_SAFE_RE = re.compile(r"[^a-z0-9_]")


def _slug(name: str) -> str:
    """Normalize an arbitrary label into a tool-name-safe slug."""
    return _NAME_SAFE_RE.sub("_", (name or "").lower()).strip("_") or "unnamed"


@dataclass
class MCPServerSpec:
    """Configuration for one MCP server."""

    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    disabled: bool = False
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MCPServerSpec:
        name = str(d.get("name") or "").strip()
        if not name:
            raise ValueError("MCP server spec missing 'name'")
        raw_cmd = d.get("command")
        if not raw_cmd:
            raise ValueError(f"MCP server {name!r}: missing 'command'")
        return cls(
            name=name,
            command=parse_command(raw_cmd),
            env=expand_env(d.get("env") or {}),
            cwd=(str(d["cwd"]) if d.get("cwd") else None),
            disabled=bool(d.get("disabled", False)),
            description=str(d.get("description") or ""),
        )


@dataclass
class _MountedTool:
    """One MCP tool mounted into PredaCore's tool-call layer."""

    server: str
    exposed_name: str   # e.g. "mcp_fs_read_file"
    remote_name: str    # e.g. "read_file"  (what we send back to the server)
    tool: MCPTool


class MCPRegistry:
    """Owns the set of running MCP clients and the tools they expose."""

    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}
        self._specs: dict[str, MCPServerSpec] = {}
        self._tools: dict[str, _MountedTool] = {}
        # Callbacks — set once by the daemon on startup so we can mutate
        # the tool layer without importing it here (avoids circular deps).
        self._on_mount: Callable[[_MountedTool], None] | None = None
        self._on_unmount: Callable[[_MountedTool], None] | None = None
        self._lock = asyncio.Lock()

    # ── Registration hooks (wired by daemon on startup) ──────────────

    def set_mount_callbacks(
        self,
        on_mount: "Callable[[_MountedTool], None]",
        on_unmount: "Callable[[_MountedTool], None]",
    ) -> None:
        self._on_mount = on_mount
        self._on_unmount = on_unmount

    # ── Starting / stopping ───────────────────────────────────────────

    async def start_all(self, specs: list[MCPServerSpec]) -> list[dict[str, Any]]:
        """Bring up every spec in parallel. Returns per-server status dicts.

        Failed servers don't block the others — the daemon still comes up.
        """
        self._specs = {s.name: s for s in specs if not s.disabled}
        if not self._specs:
            logger.info("No MCP servers configured")
            return []

        results = await asyncio.gather(
            *(self._start_one(spec) for spec in self._specs.values()),
            return_exceptions=True,
        )
        report: list[dict[str, Any]] = []
        for spec, res in zip(self._specs.values(), results):
            if isinstance(res, Exception):
                report.append({
                    "server": spec.name, "status": "error", "error": str(res),
                })
                logger.error("MCP %s failed to start: %s", spec.name, res)
            else:
                report.append({
                    "server": spec.name,
                    "status": "ok",
                    "tools": res,
                })
        return report

    async def _start_one(self, spec: MCPServerSpec) -> int:
        """Start one server and mount its tools. Returns tool count on success."""
        client = MCPClient(spec.name, spec.command, env=spec.env, cwd=spec.cwd)
        await client.start()
        try:
            tools = await client.list_tools()
        except Exception:
            await client.close()
            raise
        self._clients[spec.name] = client

        slug = _slug(spec.name)
        mounted_count = 0
        for t in tools:
            exposed = f"mcp_{slug}_{_slug(t.name)}"
            mt = _MountedTool(
                server=spec.name,
                exposed_name=exposed,
                remote_name=t.name,
                tool=t,
            )
            self._tools[exposed] = mt
            if self._on_mount is not None:
                try:
                    self._on_mount(mt)
                except Exception as exc:
                    logger.error("MCP mount hook failed for %s: %s", exposed, exc)
            mounted_count += 1
        return mounted_count

    async def stop_all(self) -> None:
        async with self._lock:
            for mt in list(self._tools.values()):
                if self._on_unmount is not None:
                    try:
                        self._on_unmount(mt)
                    except Exception:
                        pass
            self._tools.clear()

            await asyncio.gather(
                *(c.close() for c in self._clients.values()),
                return_exceptions=True,
            )
            self._clients.clear()
            self._specs.clear()

    async def restart(self, server: str) -> dict[str, Any]:
        """Tear down + bring up one server."""
        async with self._lock:
            spec = self._specs.get(server)
            if spec is None:
                return {"status": "error", "error": f"unknown server: {server}"}
            # unmount + close
            await self._remove_server(server)
            try:
                count = await self._start_one(spec)
            except MCPClientError as exc:
                return {"status": "error", "error": str(exc)}
            self._specs[server] = spec
            return {"status": "ok", "server": server, "tools_mounted": count}

    async def add_server(self, spec: MCPServerSpec) -> dict[str, Any]:
        """Register and start a new server at runtime."""
        async with self._lock:
            if spec.name in self._specs:
                return {"status": "error", "error": f"server already registered: {spec.name}"}
            if spec.disabled:
                self._specs[spec.name] = spec
                return {"status": "ok", "server": spec.name, "note": "disabled"}
            try:
                count = await self._start_one(spec)
            except MCPClientError as exc:
                return {"status": "error", "error": str(exc)}
            self._specs[spec.name] = spec
            return {"status": "ok", "server": spec.name, "tools_mounted": count}

    async def remove_server(self, server: str) -> dict[str, Any]:
        async with self._lock:
            if server not in self._specs:
                return {"status": "error", "error": f"unknown server: {server}"}
            await self._remove_server(server)
            self._specs.pop(server, None)
            return {"status": "ok", "server": server}

    async def _remove_server(self, server: str) -> None:
        # Unmount tools belonging to this server
        for exposed, mt in list(self._tools.items()):
            if mt.server != server:
                continue
            self._tools.pop(exposed, None)
            if self._on_unmount is not None:
                try:
                    self._on_unmount(mt)
                except Exception:
                    pass
        client = self._clients.pop(server, None)
        if client is not None:
            await client.close()

    # ── Queries ───────────────────────────────────────────────────────

    def list_servers(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name, spec in sorted(self._specs.items()):
            client = self._clients.get(name)
            tools = [
                mt.exposed_name for mt in self._tools.values() if mt.server == name
            ]
            out.append({
                "name": name,
                "running": bool(client and client.running),
                "disabled": spec.disabled,
                "command": spec.command,
                "description": spec.description,
                "tools": sorted(tools),
            })
        return out

    def list_tools(self) -> list[_MountedTool]:
        return list(self._tools.values())

    def describe_tool(self, exposed_name: str) -> _MountedTool | None:
        return self._tools.get(exposed_name)

    # ── Invocation ────────────────────────────────────────────────────

    async def call(
        self,
        exposed_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Route a tool call to the right client by its exposed name."""
        mt = self._tools.get(exposed_name)
        if mt is None:
            raise MCPClientError(f"No MCP tool registered as {exposed_name!r}")
        client = self._clients.get(mt.server)
        if client is None or not client.running:
            raise MCPClientError(f"MCP server {mt.server!r} is not running")
        return await client.call_tool(mt.remote_name, arguments, timeout=timeout)


# ── Module-level singleton (one registry per process) ────────────────


_registry: MCPRegistry | None = None


def get_registry() -> MCPRegistry:
    global _registry
    if _registry is None:
        _registry = MCPRegistry()
    return _registry


def reset_registry() -> None:
    global _registry
    _registry = None
