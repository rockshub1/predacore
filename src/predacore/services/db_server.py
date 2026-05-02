"""
PredaCore DB Server — Unix Domain Socket database service.

Provides a single-writer, concurrent-reader SQLite service over a Unix
domain socket using a length-prefixed JSON-RPC protocol.

Protocol framing:
    4 bytes big-endian uint32 (payload length) + UTF-8 JSON body.

Request format:
    {"method": "execute"|"query"|"query_dicts"|"executescript"|"ping",
     "params": {...}, "id": N}

Response format:
    {"result": {...}, "id": N}  or  {"error": "message", "id": N}

Databases are opened lazily with WAL mode and busy_timeout=5000.  All
mutating calls (execute / executescript) are serialized through a single
asyncio.Queue so that only one write is in-flight at a time while reads
proceed concurrently.

Usage:
    registry = {"main": "/path/to/main.db", "memory": "/path/to/mem.db"}
    server = DBServer(db_registry=registry)
    await server.start()
    ...
    await server.stop()
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sqlite3
import struct
from pathlib import Path
from typing import Any


class _BytesSafeEncoder(json.JSONEncoder):
    """JSON encoder that base64-encodes bytes values instead of crashing."""

    def default(self, o: Any) -> Any:
        if isinstance(o, (bytes, bytearray, memoryview)):
            return {"__bytes__": base64.b64encode(bytes(o)).decode("ascii")}
        return super().default(o)

log = logging.getLogger(__name__)


# Backpressure on the single-writer queue. If writes outpace SQLite throughput
# the queue grows unbounded and reads start timing out as connections pile up.
# Cap the queue and reject writes with a clear error after a short wait so the
# caller can retry / backoff instead of hanging forever.
_DEFAULT_WRITE_QUEUE_MAX = int(os.environ.get("PREDACORE_DB_WRITE_QUEUE_MAX", "50"))
_DEFAULT_WRITE_QUEUE_PUT_TIMEOUT = float(
    os.environ.get("PREDACORE_DB_WRITE_QUEUE_PUT_TIMEOUT", "10.0")
)

_DEFAULT_SOCKET = str(Path.home() / ".predacore" / "db.sock")
_HEADER = struct.Struct("!I")  # 4-byte big-endian unsigned int


class DBServer:
    """Single-writer / concurrent-reader SQLite service over a Unix socket."""

    def __init__(
        self,
        db_registry: dict[str, str],
        socket_path: str | None = None,
    ) -> None:
        self._socket_path: str = (
            os.environ.get("PREDACORE_DB_SOCKET")
            or socket_path
            or _DEFAULT_SOCKET
        )
        self._db_registry: dict[str, str] = dict(db_registry)
        self._connections: dict[str, sqlite3.Connection] = {}
        self._server: asyncio.AbstractServer | None = None
        # Bounded queue: writes time out via wait_for() rather than queueing
        # forever when SQLite is slow.
        self._write_queue: asyncio.Queue[tuple[asyncio.Future, str, str, list | None]] = (
            asyncio.Queue(maxsize=_DEFAULT_WRITE_QUEUE_MAX)
        )
        self._write_put_timeout: float = _DEFAULT_WRITE_QUEUE_PUT_TIMEOUT
        self._writer_task: asyncio.Task | None = None
        self._conn_count: int = 0
        self._query_count: int = 0
        # Cumulative count of writes rejected due to queue-full backpressure.
        # Useful for `predacore status` to surface "DB is overloaded" signals.
        self._write_rejections: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the Unix socket server and the background writer task."""
        sock_path = Path(self._socket_path)
        sock_path.parent.mkdir(parents=True, exist_ok=True)

        # Clean stale socket file
        if sock_path.exists():
            try:
                # Try connecting — if it fails the socket is stale.
                r, w = await asyncio.open_unix_connection(str(sock_path))
                w.close()
                await w.wait_closed()
                raise RuntimeError(
                    f"Another DBServer is already listening on {self._socket_path}"
                )
            except (ConnectionRefusedError, OSError):
                sock_path.unlink(missing_ok=True)
                log.info("Removed stale socket file: %s", self._socket_path)

        self._writer_task = asyncio.create_task(self._writer_loop(), name="db-writer")
        self._server = await asyncio.start_unix_server(
            self._handle_client, path=str(sock_path)
        )
        log.info("DBServer listening on %s", self._socket_path)

    async def stop(self) -> None:
        """Gracefully shut down the server, writer task, and connections."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        if self._writer_task is not None:
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass
            self._writer_task = None

        for name, conn in self._connections.items():
            try:
                conn.close()
            except Exception:
                log.warning("Error closing DB %s", name, exc_info=True)
        self._connections.clear()

        sock = Path(self._socket_path)
        sock.unlink(missing_ok=True)
        log.info(
            "DBServer stopped — served %d connections, %d queries",
            self._conn_count,
            self._query_count,
        )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_connection(self, db_name: str) -> sqlite3.Connection:
        """Return (or lazily open) a SQLite connection for *db_name*."""
        if db_name in self._connections:
            return self._connections[db_name]

        if db_name not in self._db_registry:
            raise ValueError(f"Unknown database: {db_name!r}")

        db_path = self._db_registry[db_name]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        self._connections[db_name] = conn
        log.info("Opened database %r → %s", db_name, db_path)
        return conn

    # ------------------------------------------------------------------
    # Write serialization
    # ------------------------------------------------------------------

    async def _writer_loop(self) -> None:
        """Process write requests one at a time from the queue."""
        while True:
            fut, method, db_name, payload = await self._write_queue.get()
            try:
                if method == "execute":
                    sql, params = payload  # type: ignore[misc]
                    result = await asyncio.get_running_loop().run_in_executor(
                        None, self._do_execute, db_name, sql, params
                    )
                elif method == "executescript":
                    sql = payload  # type: ignore[assignment]
                    result = await asyncio.get_running_loop().run_in_executor(
                        None, self._do_executescript, db_name, sql
                    )
                else:
                    result = {"error": f"Unknown write method: {method}"}
                if not fut.done():
                    fut.set_result(result)
            except Exception as exc:
                if not fut.done():
                    fut.set_exception(exc)
            finally:
                self._write_queue.task_done()

    def _do_execute(
        self, db_name: str, sql: str, params: list | None
    ) -> dict[str, Any]:
        conn = self._get_connection(db_name)
        cur = conn.execute(sql, params or [])
        conn.commit()
        return {"rowcount": cur.rowcount, "lastrowid": cur.lastrowid}

    def _do_executescript(self, db_name: str, sql: str) -> dict[str, Any]:
        conn = self._get_connection(db_name)
        conn.executescript(sql)
        return {"ok": True}

    # ------------------------------------------------------------------
    # Read path (concurrent)
    # ------------------------------------------------------------------

    def _do_query(
        self, db_name: str, sql: str, params: list | None
    ) -> dict[str, Any]:
        conn = self._get_connection(db_name)
        cur = conn.execute(sql, params or [])
        rows: list[list] = [list(row) for row in cur.fetchall()]
        return {"rows": rows}

    def _do_query_dicts(
        self, db_name: str, sql: str, params: list | None
    ) -> dict[str, Any]:
        conn = self._get_connection(db_name)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(sql, params or [])
            rows: list[dict[str, Any]] = [dict(row) for row in cur.fetchall()]
        finally:
            conn.row_factory = None  # type: ignore[assignment]
        return {"rows": rows}

    # ------------------------------------------------------------------
    # Client handler
    # ------------------------------------------------------------------

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._conn_count += 1
        peer = writer.get_extra_info("peername") or "unknown"
        log.info("Client connected (%s) — total connections: %d", peer, self._conn_count)
        try:
            while True:
                header = await reader.readexactly(_HEADER.size)
                (length,) = _HEADER.unpack(header)
                body = await reader.readexactly(length)
                request = json.loads(body.decode("utf-8"))
                response = await self._dispatch(request)
                self._send(writer, response)
                await writer.drain()
        except asyncio.IncompleteReadError:
            pass
        except Exception:
            log.warning("Client error (%s)", peer, exc_info=True)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        """Route a JSON-RPC request to the appropriate handler."""
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})
        self._query_count += 1

        try:
            if method == "ping":
                return {"result": {"ok": True}, "id": req_id}

            db_name = params.get("db_name", "")
            sql = params.get("sql", "")
            # Decode base64-encoded bytes params from client
            raw_params = params.get("params")
            if isinstance(raw_params, list):
                decoded = []
                for p in raw_params:
                    if isinstance(p, dict) and "__bytes__" in p and len(p) == 1:
                        decoded.append(base64.b64decode(p["__bytes__"]))
                    else:
                        decoded.append(p)
                params["params"] = decoded

            if method in ("execute", "executescript"):
                loop = asyncio.get_running_loop()
                fut: asyncio.Future = loop.create_future()
                if method == "execute":
                    payload = (sql, params.get("params"))
                else:
                    payload = sql  # type: ignore[assignment]
                # Backpressure: if the writer can't drain fast enough, return
                # a structured 503-equivalent rather than hanging forever.
                try:
                    await asyncio.wait_for(
                        self._write_queue.put((fut, method, db_name, payload)),
                        timeout=self._write_put_timeout,
                    )
                except asyncio.TimeoutError:
                    self._write_rejections += 1
                    return {
                        "error": (
                            f"DB write queue full (>{self._write_queue.maxsize} pending "
                            f"after {self._write_put_timeout:.0f}s) — server overloaded. "
                            "Retry with backoff."
                        ),
                        "id": req_id,
                    }
                result = await fut
            elif method == "query":
                result = await asyncio.get_running_loop().run_in_executor(
                    None, self._do_query, db_name, sql, params.get("params")
                )
            elif method == "query_dicts":
                result = await asyncio.get_running_loop().run_in_executor(
                    None, self._do_query_dicts, db_name, sql, params.get("params")
                )
            else:
                return {"error": f"Unknown method: {method!r}", "id": req_id}

            return {"result": result, "id": req_id}
        except Exception as exc:
            log.error("Dispatch error for method=%s: %s", method, exc, exc_info=True)
            return {"error": str(exc), "id": req_id}

    # ------------------------------------------------------------------
    # Framing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _send(writer: asyncio.StreamWriter, obj: dict[str, Any]) -> None:
        payload = json.dumps(obj, cls=_BytesSafeEncoder).encode("utf-8")
        writer.write(_HEADER.pack(len(payload)) + payload)
