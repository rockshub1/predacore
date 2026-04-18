"""
PredaCore DB Client — async and sync clients for the DB socket service.

Provides two classes:

    DBClient      — async client, connects to the Unix domain socket
                    exposed by DBServer and speaks the same length-prefixed
                    JSON-RPC protocol.

    DBClientSync  — thin synchronous wrapper around DBClient for callers
                    that cannot use ``await`` (e.g. IdentityService,
                    ApprovalHistory).  Runs its own background event loop
                    in a daemon thread.

Usage (async):
    async with DBClient() as db:
        rows = await db.query("main", "SELECT * FROM users")

Usage (sync):
    db = DBClientSync()
    db.connect()
    rows = db.query("main", "SELECT * FROM users")
    db.close()
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import struct
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


def _encode_params(params: Optional[list]) -> Optional[list]:
    """Encode bytes values in SQL params as base64 for JSON transport."""
    if not params:
        return params
    encoded = []
    for p in params:
        if isinstance(p, (bytes, bytearray, memoryview)):
            encoded.append({"__bytes__": base64.b64encode(bytes(p)).decode("ascii")})
        else:
            encoded.append(p)
    return encoded


def _decode_bytes_in_obj(obj: Any) -> Any:
    """Recursively decode __bytes__ markers back to bytes."""
    if isinstance(obj, dict):
        if "__bytes__" in obj and len(obj) == 1:
            return base64.b64decode(obj["__bytes__"])
        return {k: _decode_bytes_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_bytes_in_obj(item) for item in obj]
    return obj

log = logging.getLogger(__name__)

_DEFAULT_SOCKET = str(Path.home() / ".predacore" / "db.sock")
_HEADER = struct.Struct("!I")  # 4-byte big-endian unsigned int


class DBClient:
    """Asynchronous client for the PredaCore DB socket service."""

    def __init__(self, socket_path: Optional[str] = None) -> None:
        self._socket_path: str = (
            os.environ.get("PREDACORE_DB_SOCKET")
            or socket_path
            or _DEFAULT_SOCKET
        )
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._req_id: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the Unix socket connection to DBServer."""
        self._reader, self._writer = await asyncio.open_unix_connection(
            self._socket_path
        )
        log.info("DBClient connected to %s", self._socket_path)

    async def close(self) -> None:
        """Close the socket connection."""
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None
            log.info("DBClient disconnected")

    async def __aenter__(self) -> "DBClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    @classmethod
    async def is_available(cls, socket_path: Optional[str] = None) -> bool:
        """Return True if the socket exists and a ping succeeds."""
        path = (
            os.environ.get("PREDACORE_DB_SOCKET")
            or socket_path
            or _DEFAULT_SOCKET
        )
        if not Path(path).exists():
            return False
        try:
            client = cls(socket_path=path)
            await client.connect()
            try:
                resp = await client.ping()
                return resp.get("ok", False) is True
            finally:
                await client.close()
        except (OSError, ConnectionRefusedError, asyncio.TimeoutError):
            return False

    # ------------------------------------------------------------------
    # RPC helpers
    # ------------------------------------------------------------------

    async def _ensure_connected(self) -> None:
        """Reconnect if the underlying transport is gone."""
        if self._writer is None or self._writer.is_closing():
            log.info("DBClient reconnecting to %s", self._socket_path)
            await self.connect()

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    async def _call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Send a JSON-RPC request and return the result (or raise)."""
        async with self._lock:
            await self._ensure_connected()
            assert self._reader is not None and self._writer is not None

            # Encode bytes values in SQL params for JSON transport
            safe_params = dict(params or {})
            if "params" in safe_params:
                safe_params["params"] = _encode_params(safe_params["params"])

            req_id = self._next_id()
            request = {"method": method, "params": safe_params, "id": req_id}
            payload = json.dumps(request, default=str).encode("utf-8")
            self._writer.write(_HEADER.pack(len(payload)) + payload)
            await self._writer.drain()

            try:
                header = await self._reader.readexactly(_HEADER.size)
            except asyncio.IncompleteReadError:
                # Connection dropped — reconnect and retry once.
                await self.close()
                await self.connect()
                assert self._reader is not None and self._writer is not None
                req_id = self._next_id()
                request["id"] = req_id
                payload = json.dumps(request, default=str).encode("utf-8")
                self._writer.write(_HEADER.pack(len(payload)) + payload)
                await self._writer.drain()
                header = await self._reader.readexactly(_HEADER.size)

            (length,) = _HEADER.unpack(header)
            body = await self._reader.readexactly(length)
            response = json.loads(body.decode("utf-8"))

            if "error" in response:
                raise RuntimeError(response["error"])
            # Decode any base64-encoded bytes values back
            return _decode_bytes_in_obj(response.get("result"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        db_name: str,
        sql: str,
        params: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Execute a write statement. Returns ``{rowcount, lastrowid}``."""
        return await self._call(
            "execute",
            {"db_name": db_name, "sql": sql, "params": params},
        )

    async def query(
        self,
        db_name: str,
        sql: str,
        params: Optional[list] = None,
    ) -> List[list]:
        """Run a read query. Returns a list of row-tuples (as lists)."""
        result = await self._call(
            "query",
            {"db_name": db_name, "sql": sql, "params": params},
        )
        return result["rows"]

    async def query_dicts(
        self,
        db_name: str,
        sql: str,
        params: Optional[list] = None,
    ) -> List[Dict[str, Any]]:
        """Run a read query. Returns a list of dicts keyed by column name."""
        result = await self._call(
            "query_dicts",
            {"db_name": db_name, "sql": sql, "params": params},
        )
        return result["rows"]

    async def executescript(self, db_name: str, sql: str) -> Dict[str, Any]:
        """Execute a multi-statement SQL script. Returns ``{ok: true}``."""
        return await self._call(
            "executescript",
            {"db_name": db_name, "sql": sql},
        )

    async def ping(self) -> Dict[str, Any]:
        """Ping the server. Returns ``{ok: true}``."""
        return await self._call("ping")


# ======================================================================
# Synchronous wrapper
# ======================================================================


class DBClientSync:
    """Synchronous wrapper around :class:`DBClient`.

    Spins up a daemon thread with its own event loop so that synchronous
    callers (IdentityService, ApprovalHistory, etc.) can talk to the
    DBServer without needing ``await``.
    """

    def __init__(self, socket_path: Optional[str] = None) -> None:
        self._socket_path = socket_path
        self._client: Optional[DBClient] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _start_loop(self) -> None:
        """Entry point for the background thread."""
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def connect(self) -> None:
        """Start the background loop and connect the async client."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._start_loop, daemon=True, name="dbclient-sync"
        )
        self._thread.start()

        self._client = DBClient(socket_path=self._socket_path)
        future = asyncio.run_coroutine_threadsafe(
            self._client.connect(), self._loop
        )
        future.result(timeout=10)

    def close(self) -> None:
        """Disconnect the client and tear down the background loop."""
        if self._client is not None and self._loop is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._client.close(), self._loop
                )
                future.result(timeout=5)
            except Exception:
                pass
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._client = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run(self, coro: Any) -> Any:
        """Submit a coroutine to the background loop and block for result."""
        if self._loop is None or self._client is None:
            raise RuntimeError("DBClientSync is not connected — call connect() first")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=30)

    # ------------------------------------------------------------------
    # Public API (mirrors DBClient)
    # ------------------------------------------------------------------

    def execute(
        self,
        db_name: str,
        sql: str,
        params: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Execute a write statement. Returns ``{rowcount, lastrowid}``."""
        assert self._client is not None
        return self._run(self._client.execute(db_name, sql, params))

    def query(
        self,
        db_name: str,
        sql: str,
        params: Optional[list] = None,
    ) -> List[list]:
        """Run a read query. Returns a list of row-tuples (as lists)."""
        assert self._client is not None
        return self._run(self._client.query(db_name, sql, params))

    def query_dicts(
        self,
        db_name: str,
        sql: str,
        params: Optional[list] = None,
    ) -> List[Dict[str, Any]]:
        """Run a read query. Returns a list of dicts keyed by column name."""
        assert self._client is not None
        return self._run(self._client.query_dicts(db_name, sql, params))

    def executescript(self, db_name: str, sql: str) -> Dict[str, Any]:
        """Execute a multi-statement SQL script. Returns ``{ok: true}``."""
        assert self._client is not None
        return self._run(self._client.executescript(db_name, sql))

    def ping(self) -> Dict[str, Any]:
        """Ping the server. Returns ``{ok: true}``."""
        assert self._client is not None
        return self._run(self._client.ping())
