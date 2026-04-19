"""
PredaCore DB Adapter — transparent socket-or-local SQLite access.

Provides a single :class:`DBAdapter` that tries to reach the running
:class:`~predacore.services.db_server.DBServer` over its Unix socket.  If
the socket is unavailable (server not running, socket missing, etc.) it
falls back automatically to direct SQLite connections opened in the
current process, so callers never need to care which path is active.

Usage:
    adapter = DBAdapter(
        socket_path="~/.predacore/db.sock",
        db_registry={"main": "/data/main.db"},
    )
    rows = await adapter.query("main", "SELECT 1")
    print(adapter.is_using_socket)  # True or False
"""
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_SOCKET = str(Path.home() / ".predacore" / "db.sock")


class DBAdapter:
    """Transparent DB access: prefer socket, fall back to direct SQLite.

    The adapter is safe to use from multiple threads when operating in
    direct-fallback mode — each thread receives its own SQLite
    connection via thread-local storage.
    """

    def __init__(
        self,
        socket_path: str | None = None,
        db_registry: dict[str, str] | None = None,
    ) -> None:
        self._socket_path: str = (
            os.environ.get("PREDACORE_DB_SOCKET")
            or socket_path
            or _DEFAULT_SOCKET
        )
        self._db_registry: dict[str, str] = dict(db_registry or {})

        # Socket client (lazy)
        self._client: Any = None  # DBClient | None
        self._client_lock: asyncio.Lock = asyncio.Lock()
        self._using_socket: bool = False

        # Direct-fallback thread-local connections
        self._local = threading.local()
        self._direct_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_using_socket(self) -> bool:
        """True if the adapter is currently talking to DBServer."""
        return self._using_socket

    # ------------------------------------------------------------------
    # Socket path
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> bool:
        """Try to connect the socket client.  Return True on success."""
        # Avoid circular import at module level.
        from .db_client import DBClient  # noqa: F811

        async with self._client_lock:
            if self._client is not None and self._using_socket:
                return True

            available = await DBClient.is_available(socket_path=self._socket_path)
            if not available:
                self._using_socket = False
                return False

            try:
                client = DBClient(socket_path=self._socket_path)
                await client.connect()
                self._client = client
                self._using_socket = True
                log.info("DBAdapter: using socket at %s", self._socket_path)
                return True
            except (OSError, ConnectionRefusedError):
                self._using_socket = False
                return False

    # ------------------------------------------------------------------
    # Direct-fallback path (thread-safe via thread-local storage)
    # ------------------------------------------------------------------

    def _get_direct_connection(self, db_name: str) -> sqlite3.Connection:
        """Return a thread-local SQLite connection for *db_name*."""
        attr = f"_conn_{db_name}"
        conn: sqlite3.Connection | None = getattr(self._local, attr, None)
        if conn is not None:
            return conn

        if db_name not in self._db_registry:
            raise ValueError(f"Unknown database: {db_name!r}")

        db_path = self._db_registry[db_name]
        with self._direct_lock:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        setattr(self._local, attr, conn)
        log.info("DBAdapter: direct connection to %r → %s", db_name, db_path)
        return conn

    def _direct_execute(
        self, db_name: str, sql: str, params: list | None = None
    ) -> dict[str, Any]:
        conn = self._get_direct_connection(db_name)
        cur = conn.execute(sql, params or [])
        conn.commit()
        return {"rowcount": cur.rowcount, "lastrowid": cur.lastrowid}

    def _direct_query(
        self, db_name: str, sql: str, params: list | None = None
    ) -> list[list]:
        conn = self._get_direct_connection(db_name)
        cur = conn.execute(sql, params or [])
        return [list(row) for row in cur.fetchall()]

    def _direct_query_dicts(
        self, db_name: str, sql: str, params: list | None = None
    ) -> list[dict[str, Any]]:
        conn = self._get_direct_connection(db_name)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(sql, params or [])
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.row_factory = None  # type: ignore[assignment]

    def _direct_executescript(self, db_name: str, sql: str) -> dict[str, Any]:
        conn = self._get_direct_connection(db_name)
        conn.executescript(sql)
        return {"ok": True}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        db_name: str,
        sql: str,
        params: list | None = None,
    ) -> dict[str, Any]:
        """Execute a write statement. Returns ``{rowcount, lastrowid}``."""
        if await self._ensure_client():
            try:
                return await self._client.execute(db_name, sql, params)
            except Exception:
                log.warning("Socket execute failed, falling back to direct", exc_info=True)
                self._using_socket = False

        return await asyncio.get_running_loop().run_in_executor(
            None, self._direct_execute, db_name, sql, params
        )

    async def query(
        self,
        db_name: str,
        sql: str,
        params: list | None = None,
    ) -> list[list]:
        """Run a read query. Returns a list of row-tuples (as lists)."""
        if await self._ensure_client():
            try:
                return await self._client.query(db_name, sql, params)
            except Exception:
                log.warning("Socket query failed, falling back to direct", exc_info=True)
                self._using_socket = False

        return await asyncio.get_running_loop().run_in_executor(
            None, self._direct_query, db_name, sql, params
        )

    async def query_dicts(
        self,
        db_name: str,
        sql: str,
        params: list | None = None,
    ) -> list[dict[str, Any]]:
        """Run a read query. Returns a list of dicts keyed by column name."""
        if await self._ensure_client():
            try:
                return await self._client.query_dicts(db_name, sql, params)
            except Exception:
                log.warning("Socket query_dicts failed, falling back to direct", exc_info=True)
                self._using_socket = False

        return await asyncio.get_running_loop().run_in_executor(
            None, self._direct_query_dicts, db_name, sql, params
        )

    async def executescript(self, db_name: str, sql: str) -> dict[str, Any]:
        """Execute a multi-statement SQL script. Returns ``{ok: true}``."""
        if await self._ensure_client():
            try:
                return await self._client.executescript(db_name, sql)
            except Exception:
                log.warning("Socket executescript failed, falling back to direct", exc_info=True)
                self._using_socket = False

        return await asyncio.get_running_loop().run_in_executor(
            None, self._direct_executescript, db_name, sql
        )
