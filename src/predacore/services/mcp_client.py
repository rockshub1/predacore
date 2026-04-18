"""
MCP (Model Context Protocol) client — stdio transport.

PredaCore uses this to consume third-party MCP servers. Every MCP tool in
the ecosystem (Anthropic's filesystem/github/slack/etc., plus community
servers) becomes callable by the agent without writing a PredaCore wrapper.

We implement the *client* side only. PredaCore no longer runs its own MCP
server — all tool calling internally goes through the PredaCore SDK — but
we do want to reach out and use other people's MCP servers.

Protocol reference: https://modelcontextprotocol.io/specification

Transport is JSON-RPC 2.0 over a subprocess's stdin/stdout. Each message is
a single UTF-8 line (newline-delimited JSON, or "NDJSON"). Framing here is
plain line-based — the server we spawn is expected to speak NDJSON. (The
spec also defines a Content-Length framing, but every production server
we'd connect to uses NDJSON in practice.)

Usage:

    client = MCPClient("fs", ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
    await client.start()
    tools = await client.list_tools()
    result = await client.call_tool("read_file", {"path": "/tmp/foo.txt"})
    await client.close()
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Protocol constants
_PROTOCOL_VERSION = "2025-06-18"  # Most recent spec revision as of writing
_CLIENT_INFO = {"name": "predacore", "version": "0.1.0"}
_DEFAULT_TIMEOUT_S = 30.0
_INITIALIZE_TIMEOUT_S = 10.0


class MCPClientError(Exception):
    """Raised when an MCP server returns a JSON-RPC error or crashes."""


@dataclass
class MCPTool:
    """A single tool exposed by an MCP server."""

    name: str                     # canonical tool name as the server reports it
    description: str              # human-readable, passed through to the LLM
    input_schema: dict[str, Any]  # JSON schema for arguments (``tools/call`` params)


class MCPClient:
    """One subprocess, speaking JSON-RPC 2.0 over stdio, implementing the
    MCP client role.

    Thread-safe for single-event-loop use. Concurrent requests are allowed —
    each gets its own JSON-RPC id and its own future.
    """

    def __init__(
        self,
        name: str,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        default_timeout: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        if not command:
            raise ValueError("command must be a non-empty list of argv")
        self.name = name
        self.command = list(command)
        self.env = dict(env) if env else None
        self.cwd = cwd
        self.default_timeout = default_timeout

        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._next_id = 1
        self._started = False
        self._closed = False
        self._send_lock = asyncio.Lock()

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Spawn the subprocess and run the MCP initialize handshake."""
        if self._started:
            return
        self._started = True

        merged_env = os.environ.copy()
        if self.env:
            merged_env.update(self.env)

        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=merged_env,
                cwd=self.cwd,
            )
        except FileNotFoundError as exc:
            self._started = False
            raise MCPClientError(
                f"MCP server {self.name!r} command not found: {self.command[0]!r}"
            ) from exc

        self._reader_task = asyncio.create_task(
            self._reader_loop(), name=f"mcp-reader:{self.name}",
        )
        self._stderr_task = asyncio.create_task(
            self._stderr_drain_loop(), name=f"mcp-stderr:{self.name}",
        )

        # MCP initialize — the first round-trip.
        try:
            await asyncio.wait_for(
                self._initialize(),
                timeout=_INITIALIZE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            await self.close()
            raise MCPClientError(
                f"MCP server {self.name!r} did not respond to initialize "
                f"within {_INITIALIZE_TIMEOUT_S}s"
            ) from None
        except Exception:
            await self.close()
            raise

        logger.info("MCP server started: %s (%s)", self.name, " ".join(self.command))

    async def _initialize(self) -> None:
        await self._request("initialize", {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {},  # we don't need any client-side capabilities
            "clientInfo": _CLIENT_INFO,
        }, timeout=_INITIALIZE_TIMEOUT_S)
        # The client MUST send `initialized` after initialize succeeds.
        await self._notify("notifications/initialized")

    async def close(self) -> None:
        """Terminate the subprocess and cancel pending requests."""
        if self._closed:
            return
        self._closed = True

        if self._proc is not None and self._proc.returncode is None:
            try:
                self._proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                try:
                    self._proc.kill()
                except ProcessLookupError:
                    pass

        for task in (self._reader_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(MCPClientError(
                    f"MCP server {self.name!r} closed before responding"
                ))
        self._pending.clear()
        logger.info("MCP server closed: %s", self.name)

    @property
    def running(self) -> bool:
        return (
            self._started
            and not self._closed
            and self._proc is not None
            and self._proc.returncode is None
        )

    # ── Public protocol methods ──────────────────────────────────────

    async def list_tools(self) -> list[MCPTool]:
        """Return the full tool catalog from the server."""
        result = await self._request("tools/list", {})
        raw = result.get("tools") or []
        out: list[MCPTool] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or ""
            if not name:
                continue
            out.append(MCPTool(
                name=name,
                description=item.get("description") or "",
                input_schema=item.get("inputSchema") or {"type": "object"},
            ))
        return out

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Invoke ``tool_name`` and return the raw MCP result dict.

        The result follows the spec's ``CallToolResult`` shape:
        ``{"content": [{"type": "text", "text": "..."}, ...], "isError": bool}``.
        """
        return await self._request(
            "tools/call",
            {"name": tool_name, "arguments": arguments or {}},
            timeout=timeout,
        )

    # ── JSON-RPC plumbing ────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if not self._started or self._closed:
            raise MCPClientError(f"MCP server {self.name!r} is not running")

        req_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut

        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        try:
            await self._send_line(json.dumps(payload, ensure_ascii=False))
        except Exception:
            self._pending.pop(req_id, None)
            raise

        try:
            return await asyncio.wait_for(
                fut, timeout=timeout or self.default_timeout,
            )
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise MCPClientError(
                f"MCP server {self.name!r} request {method!r} timed out"
            ) from None

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            payload["params"] = params
        await self._send_line(json.dumps(payload, ensure_ascii=False))

    async def _send_line(self, line: str) -> None:
        async with self._send_lock:
            assert self._proc and self._proc.stdin
            self._proc.stdin.write((line + "\n").encode("utf-8"))
            await self._proc.stdin.drain()

    async def _reader_loop(self) -> None:
        assert self._proc and self._proc.stdout
        stdout = self._proc.stdout
        while True:
            try:
                line = await stdout.readline()
            except Exception:
                break
            if not line:
                break  # stdout closed — server exited
            try:
                msg = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                logger.warning(
                    "MCP %s: non-JSON output: %r (%s)",
                    self.name, line[:200], exc,
                )
                continue
            self._dispatch(msg)

    def _dispatch(self, msg: dict[str, Any]) -> None:
        # Response to a pending request?
        if "id" in msg and ("result" in msg or "error" in msg):
            fut = self._pending.pop(msg["id"], None)
            if fut is None or fut.done():
                return
            if "error" in msg:
                err = msg["error"] or {}
                fut.set_exception(MCPClientError(
                    f"{self.name}: {err.get('message') or 'unknown error'} "
                    f"(code={err.get('code')})"
                ))
            else:
                result = msg["result"]
                fut.set_result(result if isinstance(result, dict) else {"value": result})
            return

        # Server-initiated notifications — we ignore most for now.
        # Logging, progress, cancellation notifications land here; the
        # framework doesn't propagate them upward. Extend later if needed.
        method = msg.get("method", "?")
        logger.debug("MCP %s notification: %s", self.name, method)

    async def _stderr_drain_loop(self) -> None:
        """Forward the server's stderr to our logs — useful for debugging."""
        assert self._proc and self._proc.stderr
        while True:
            try:
                line = await self._proc.stderr.readline()
            except Exception:
                break
            if not line:
                break
            try:
                text = line.decode("utf-8", "replace").rstrip()
            except Exception:
                continue
            if text:
                logger.debug("[mcp:%s] %s", self.name, text)


# ── Helpers ──────────────────────────────────────────────────────────


def parse_command(value: str | list[str]) -> list[str]:
    """Accept either a shell-ish string or a list of argv tokens."""
    if isinstance(value, list):
        return [str(v) for v in value]
    return shlex.split(value)


def expand_env(env: dict[str, str] | None) -> dict[str, str]:
    """Resolve ``${VAR}`` placeholders in each value against os.environ.

    Useful for MCP server configs that reference secrets from ``.env``:

        env:
          GITHUB_PERSONAL_ACCESS_TOKEN: ${GITHUB_TOKEN}
    """
    if not env:
        return {}
    out: dict[str, str] = {}
    for k, v in env.items():
        out[k] = os.path.expandvars(str(v))
    return out
