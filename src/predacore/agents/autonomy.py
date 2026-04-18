"""
Autonomy runtime for OpenClaw delegation in PredaCore.

Provides:
  - OpenClaw bridge calls with retry/backoff
  - Async status polling for queued/running jobs
  - Deterministic idempotency cache (SQLite)
  - Append-only action ledger (JSONL)
  - Runtime kill switch guard
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin
from uuid import uuid4

from ..config import PredaCoreConfig

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
PENDING_STATUSES = {"queued", "pending", "running", "processing", "accepted"}
COMPLETED_STATUSES = {
    "complete",
    "completed",
    "done",
    "success",
    "succeeded",
    "failed",
    "error",
    "cancelled",
    "canceled",
}

_SENSITIVE_KEYS = (
    "authorization",
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
)


def _parse_bool(raw: Any) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def _coerce_int(raw: Any, default: int, minimum: int | None = None) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _coerce_float(raw: Any, default: float, minimum: float | None = None) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, nested in value.items():
            key_lower = str(key).lower()
            if any(token in key_lower for token in _SENSITIVE_KEYS):
                out[key] = "[REDACTED]"
            else:
                out[key] = _redact_sensitive(nested)
        return out
    if isinstance(value, list):
        return [_redact_sensitive(v) for v in value]
    return value


class KillSwitchActivatedError(RuntimeError):
    """Raised when the autonomy kill switch is active."""


class ActionLedger:
    """Append-only JSONL ledger for delegated actions."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "ts": time.time(),
            "event": event_type,
            "payload": _redact_sensitive(payload),
        }
        line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(line)


class IdempotencyStore:
    """SQLite store for deterministic idempotency reuse."""

    _DB_NAME = "openclaw_idempotency"
    MAX_ROWS = 10000

    def __init__(self, db_path: Path, db_adapter: Any = None):
        self._db_path = Path(db_path)
        self._db_adapter = db_adapter
        if db_adapter is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._lock = threading.Lock()
            self._conn = sqlite3.connect(
                str(self._db_path),
                timeout=30.0,
                check_same_thread=False,
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS idempotency_cache (
                    cache_key TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    response_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "DELETE FROM idempotency_cache WHERE created_at < ?",
                (time.time() - 7 * 86400,),
            )
            self._conn.execute(
                """
                DELETE FROM idempotency_cache
                WHERE rowid NOT IN (
                    SELECT rowid FROM idempotency_cache
                    ORDER BY created_at DESC LIMIT ?
                )
                """,
                (self.MAX_ROWS,),
            )
            self._conn.commit()
        else:
            self._lock = None  # type: ignore[assignment]
            self._conn = None  # type: ignore[assignment]

    @property
    def path(self) -> Path:
        return self._db_path

    async def init_schema(self) -> None:
        """Create the idempotency table via the adapter (call once at startup)."""
        if self._db_adapter is not None:
            await self._db_adapter.executescript(
                self._DB_NAME,
                """
                CREATE TABLE IF NOT EXISTS idempotency_cache (
                    cache_key TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    response_json TEXT NOT NULL
                )
                """,
            )
            await self._db_adapter.execute(
                self._DB_NAME,
                "DELETE FROM idempotency_cache WHERE created_at < ?",
                [time.time() - 7 * 86400],
            )
            await self._db_adapter.execute(
                self._DB_NAME,
                """
                DELETE FROM idempotency_cache
                WHERE rowid NOT IN (
                    SELECT rowid FROM idempotency_cache
                    ORDER BY created_at DESC LIMIT ?
                )
                """,
                [self.MAX_ROWS],
            )

    def get(self, cache_key: str) -> dict[str, Any] | None:
        if self._db_adapter is not None:
            raise RuntimeError("Use async get_async() when db_adapter is configured")
        with self._lock:
            row = self._conn.execute(
                "SELECT response_json FROM idempotency_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    async def get_async(self, cache_key: str) -> dict[str, Any] | None:
        """Async get — uses adapter when available, falls back to sync path."""
        if self._db_adapter is not None:
            rows = await self._db_adapter.query(
                self._DB_NAME,
                "SELECT response_json FROM idempotency_cache WHERE cache_key = ?",
                [cache_key],
            )
            if not rows:
                return None
            try:
                return json.loads(rows[0][0])
            except (json.JSONDecodeError, IndexError):
                return None
        return self.get(cache_key)

    def put(self, cache_key: str, response_payload: dict[str, Any]) -> None:
        if self._db_adapter is not None:
            raise RuntimeError("Use async put_async() when db_adapter is configured")
        encoded = json.dumps(response_payload, default=str)
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO idempotency_cache
                (cache_key, created_at, response_json)
                VALUES (?, ?, ?)
                """,
                (cache_key, time.time(), encoded),
            )
            self._conn.commit()

    async def put_async(self, cache_key: str, response_payload: dict[str, Any]) -> None:
        """Async put — uses adapter when available, falls back to sync path."""
        if self._db_adapter is not None:
            encoded = json.dumps(response_payload, default=str)
            await self._db_adapter.execute(
                self._DB_NAME,
                """
                INSERT OR REPLACE INTO idempotency_cache
                (cache_key, created_at, response_json)
                VALUES (?, ?, ?)
                """,
                [cache_key, time.time(), encoded],
            )
            return
        self.put(cache_key, response_payload)


@dataclass
class OpenClawRuntimeSettings:
    status_path: str = "/v1/tasks/{task_id}"
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    poll_interval_seconds: float = 1.5
    max_poll_seconds: int = 180
    ledger_path: str = ""
    idempotency_db_path: str = ""

    @classmethod
    def from_config(cls, config: PredaCoreConfig) -> OpenClawRuntimeSettings:
        default_ledger = str(Path(config.logs_dir) / "openclaw_actions.jsonl")
        default_idempotency = str(
            Path(config.memory.persistence_dir) / "openclaw_idempotency.db"
        )
        return cls(
            status_path=(
                os.getenv("OPENCLAW_BRIDGE_STATUS_PATH") or config.openclaw.status_path
            ),
            max_retries=_coerce_int(
                os.getenv("OPENCLAW_BRIDGE_MAX_RETRIES"),
                config.openclaw.max_retries,
                minimum=0,
            ),
            retry_backoff_seconds=_coerce_float(
                os.getenv("OPENCLAW_BRIDGE_RETRY_BACKOFF"),
                config.openclaw.retry_backoff_seconds,
                minimum=0.1,
            ),
            poll_interval_seconds=_coerce_float(
                os.getenv("OPENCLAW_BRIDGE_POLL_INTERVAL"),
                config.openclaw.poll_interval_seconds,
                minimum=0.1,
            ),
            max_poll_seconds=_coerce_int(
                os.getenv("OPENCLAW_BRIDGE_MAX_POLL_SECONDS"),
                config.openclaw.max_poll_seconds,
                minimum=1,
            ),
            ledger_path=os.getenv("PREDACORE_ACTION_LEDGER_PATH", default_ledger),
            idempotency_db_path=os.getenv(
                "PREDACORE_IDEMPOTENCY_DB_PATH", default_idempotency
            ),
        )


class OpenClawBridgeRuntime:
    """Runtime client for OpenClaw delegation calls."""

    def __init__(self, config: PredaCoreConfig, db_adapter: Any = None):
        self._config = config
        self._db_adapter = db_adapter
        self._settings = OpenClawRuntimeSettings.from_config(config)
        self._ledger = ActionLedger(Path(self._settings.ledger_path))
        self._idempotency = IdempotencyStore(
            Path(self._settings.idempotency_db_path), db_adapter=db_adapter
        )
        self._http_client: Any = None  # lazily created httpx.AsyncClient

    async def close(self) -> None:
        """Close the HTTP client to prevent connection leaks."""
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except (OSError, RuntimeError):
                pass
            self._http_client = None

    @property
    def settings(self) -> OpenClawRuntimeSettings:
        return self._settings

    @property
    def ledger_path(self) -> Path:
        return self._ledger.path

    @property
    def idempotency_db_path(self) -> Path:
        return self._idempotency.path

    async def delegate(
        self,
        *,
        task: str,
        context: dict[str, Any],
        mode: str = "oneshot",
        timeout_seconds: int | None = None,
        await_completion: bool = True,
        idempotency_key: str = "",
        extra_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Delegate a task to OpenClaw with retries, polling, and audit trail."""
        if self.kill_switch_active():
            raise KillSwitchActivatedError(
                "PredaCore kill switch is active (set PREDACORE_KILL_SWITCH=0 to resume)"
            )

        base_url = (self._config.openclaw.base_url or "").strip()
        if not base_url:
            raise RuntimeError(
                "OpenClaw bridge is enabled but OPENCLAW_BRIDGE_URL is not configured"
            )

        task_value = str(task or "").strip()
        if not task_value:
            raise ValueError("Missing required field: task")

        mode_value = str(mode or "oneshot").strip() or "oneshot"
        request_timeout = _coerce_int(
            timeout_seconds or self._config.openclaw.timeout_seconds,
            self._config.openclaw.timeout_seconds,
            minimum=5,
        )
        request_timeout = min(request_timeout, 3600)

        endpoint = urljoin(
            base_url.rstrip("/") + "/", self._config.openclaw.task_path.lstrip("/")
        )

        context_value: dict[str, Any]
        if isinstance(context, dict):
            context_value = context
        else:
            context_value = {"raw_context": context}

        cache_key = idempotency_key or self._build_idempotency_key(
            task=task_value,
            mode=mode_value,
            context=context_value,
        )

        cached = await self._idempotency.get_async(cache_key)
        if cached:
            output = dict(cached)
            output["cache_hit"] = True
            self._ledger.append(
                "openclaw_delegate_cached",
                {
                    "idempotency_key": cache_key,
                    "task_preview": task_value[:240],
                    "endpoint": endpoint,
                },
            )
            return output

        request_id = str(uuid4())
        task_path = str(self._config.openclaw.task_path or "").strip()
        use_openresponses = self._is_openresponses_endpoint(task_path)
        payload = self._build_bridge_payload(
            request_id=request_id,
            task=task_value,
            mode=mode_value,
            context=context_value,
            extra_meta=extra_meta or {},
            use_openresponses=use_openresponses,
        )

        headers = {"Content-Type": "application/json", "X-Idempotency-Key": cache_key}
        api_key = (self._config.openclaw.api_key or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if use_openresponses:
            agent_id = str(
                os.getenv("OPENCLAW_BRIDGE_AGENT_ID") or self._config.openclaw.agent_id
            ).strip()
            if agent_id:
                headers["x-openclaw-agent-id"] = agent_id

        started = time.monotonic()
        try:
            body, http_status, attempts = await self._http_post_json(
                endpoint=endpoint,
                payload=payload,
                headers=headers,
                timeout_seconds=request_timeout,
            )
            polls = 0
            status_url = ""
            if await_completion and self._should_poll(
                body, allow_task_id_polling=not use_openresponses
            ):
                body, polls, status_url = await self._poll_until_complete(
                    initial_payload=body,
                    headers=headers,
                    timeout_seconds=request_timeout,
                    base_url=base_url,
                    allow_task_id_polling=not use_openresponses,
                )

            elapsed_ms = int((time.monotonic() - started) * 1000)
            result = {
                "cache_hit": False,
                "request_id": request_id,
                "idempotency_key": cache_key,
                "endpoint": endpoint,
                "http_status": http_status,
                "attempts": attempts,
                "polls": polls,
                "status_url": status_url,
                "status": self._extract_status(body),
                "elapsed_ms": elapsed_ms,
                "response": body,
            }
            await self._idempotency.put_async(cache_key, result)
            self._ledger.append(
                "openclaw_delegate",
                {
                    "request_id": request_id,
                    "idempotency_key": cache_key,
                    "endpoint": endpoint,
                    "attempts": attempts,
                    "polls": polls,
                    "http_status": http_status,
                    "elapsed_ms": elapsed_ms,
                    "status": self._extract_status(body),
                },
            )
            return result
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            self._ledger.append(
                "openclaw_delegate_error",
                {
                    "request_id": request_id,
                    "idempotency_key": cache_key,
                    "endpoint": endpoint,
                    "elapsed_ms": elapsed_ms,
                    "error": str(exc),
                },
            )
            raise

    def kill_switch_active(self) -> bool:
        """
        Kill switch is active when:
          - PREDACORE_KILL_SWITCH is truthy, or
          - PREDACORE_KILL_SWITCH points to a file with a truthy value.
        """
        raw = os.getenv("PREDACORE_KILL_SWITCH", "").strip()
        if not raw:
            return False
        if _parse_bool(raw):
            return True
        path = Path(raw).expanduser()
        if not path.exists() or not path.is_file():
            return False
        try:
            content = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return False
        return _parse_bool(content)

    def _build_idempotency_key(
        self, *, task: str, mode: str, context: dict[str, Any]
    ) -> str:
        canonical = json.dumps(
            {"mode": mode, "task": task, "context": context},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _get_http_client(self) -> Any:
        """Return a cached httpx.AsyncClient, creating one lazily on first use."""
        if self._http_client is None:
            import httpx
            self._http_client = httpx.AsyncClient(
                verify=self._config.openclaw.verify_tls,
            )
        return self._http_client

    async def _http_post_json(
        self,
        *,
        endpoint: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout_seconds: int,
    ) -> tuple[Any, int, int]:
        import httpx

        client = self._get_http_client()
        attempts_total = self._settings.max_retries + 1
        for attempt in range(1, attempts_total + 1):
            try:
                response = await client.post(
                    endpoint, json=payload, headers=headers, timeout=timeout_seconds
                )
                status_code = int(getattr(response, "status_code", 0))
                body = self._decode_response_body(response)

                if 200 <= status_code < 300:
                    return body, status_code, attempt

                if status_code in RETRYABLE_STATUS_CODES and attempt < attempts_total:
                    await self._retry_sleep(attempt)
                    continue

                raise RuntimeError(
                    f"OpenClaw bridge returned HTTP {status_code}: "
                    f"{self._short_preview(body)}"
                )
            except RuntimeError:
                raise
            except Exception as exc:
                import httpx
                is_transient = isinstance(exc, (httpx.TransportError, httpx.TimeoutException, OSError, ConnectionError))
                if is_transient and attempt < attempts_total:
                    logger.debug("Transient error on attempt %d: %s", attempt, exc)
                    await self._retry_sleep(attempt)
                    continue
                detail = str(exc).strip() or exc.__class__.__name__
                raise RuntimeError(
                    f"OpenClaw bridge call failed after {attempt} attempts: {detail}"
                ) from exc

        raise RuntimeError("OpenClaw bridge call failed unexpectedly")

    async def _http_get_json(
        self,
        *,
        endpoint: str,
        headers: dict[str, str],
        timeout_seconds: int,
    ) -> tuple[Any, int]:
        client = self._get_http_client()
        response = await client.get(endpoint, headers=headers, timeout=timeout_seconds)

        status_code = int(getattr(response, "status_code", 0))
        body = self._decode_response_body(response)
        return body, status_code

    async def _poll_until_complete(
        self,
        *,
        initial_payload: Any,
        headers: dict[str, str],
        timeout_seconds: int,
        base_url: str,
        allow_task_id_polling: bool,
    ) -> tuple[Any, int, str]:
        status_url = self._extract_status_url(initial_payload)
        if status_url:
            status_url = self._normalize_url(status_url, base_url)
        elif allow_task_id_polling and self._settings.status_path:
            task_id = self._extract_task_id(initial_payload)
            if not task_id:
                return initial_payload, 0, ""
            status_path = str(self._settings.status_path).format(task_id=task_id)
            status_url = self._normalize_url(status_path, base_url)
        else:
            return initial_payload, 0, ""

        current = initial_payload
        polls = 0
        started = time.monotonic()
        while (time.monotonic() - started) < self._settings.max_poll_seconds:
            status = self._extract_status(current)
            if status in COMPLETED_STATUSES:
                return current, polls, status_url
            await asyncio.sleep(self._settings.poll_interval_seconds)
            current, status_code = await self._http_get_json(
                endpoint=status_url,
                headers=headers,
                timeout_seconds=timeout_seconds,
            )
            polls += 1
            if status_code >= 400 and status_code not in RETRYABLE_STATUS_CODES:
                raise RuntimeError(
                    f"OpenClaw status endpoint returned HTTP {status_code}"
                )

        raise RuntimeError(
            f"OpenClaw polling timed out after {self._settings.max_poll_seconds}s"
        )

    async def _retry_sleep(self, attempt: int) -> None:
        delay = self._settings.retry_backoff_seconds * (2 ** (attempt - 1))
        await asyncio.sleep(delay)

    def _should_poll(self, payload: Any, *, allow_task_id_polling: bool) -> bool:
        status = self._extract_status(payload)
        if status in COMPLETED_STATUSES:
            return False
        if self._extract_status_url(payload):
            return True
        if not allow_task_id_polling:
            return False
        if status in PENDING_STATUSES:
            return True
        return bool(self._extract_task_id(payload))

    def _extract_status(self, payload: Any) -> str:
        value = self._find_nested_value(payload, ("status", "state", "phase"))
        return str(value).strip().lower() if value else ""

    def _extract_task_id(self, payload: Any) -> str:
        value = self._find_nested_value(
            payload,
            ("task_id", "taskId", "job_id", "jobId", "run_id", "runId", "id"),
        )
        return str(value).strip() if value else ""

    def _extract_status_url(self, payload: Any) -> str:
        value = self._find_nested_value(
            payload,
            ("status_url", "statusUrl", "poll_url", "pollUrl"),
        )
        return str(value).strip() if value else ""

    def _normalize_url(self, maybe_relative: str, base_url: str) -> str:
        value = str(maybe_relative).strip()
        if value.startswith("http://") or value.startswith("https://"):
            return value
        return urljoin(base_url.rstrip("/") + "/", value.lstrip("/"))

    def _find_nested_value(self, payload: Any, keys: tuple[str, ...]) -> Any | None:
        if isinstance(payload, dict):
            for key in keys:
                if key in payload and payload[key] not in (None, ""):
                    return payload[key]
            for nested in payload.values():
                found = self._find_nested_value(nested, keys)
                if found not in (None, ""):
                    return found
        elif isinstance(payload, list):
            for nested in payload:
                found = self._find_nested_value(nested, keys)
                if found not in (None, ""):
                    return found
        return None

    def _decode_response_body(self, response: Any) -> Any:
        try:
            return response.json()
        except (ValueError, AttributeError):
            text = getattr(response, "text", "")
            return {"raw_text": text}

    def _short_preview(self, payload: Any, max_len: int = 240) -> str:
        value = (
            json.dumps(payload, default=str)
            if not isinstance(payload, str)
            else payload
        )
        if len(value) <= max_len:
            return value
        return value[:max_len] + "...[truncated]"

    def _is_openresponses_endpoint(self, task_path: str) -> bool:
        path = str(task_path or "").strip().lower()
        return path.endswith("/v1/responses") or path.endswith("/responses")

    def _build_bridge_payload(
        self,
        *,
        request_id: str,
        task: str,
        mode: str,
        context: dict[str, Any],
        extra_meta: dict[str, Any],
        use_openresponses: bool,
    ) -> dict[str, Any]:
        if use_openresponses:
            return self._build_openresponses_payload(
                request_id=request_id,
                task=task,
                mode=mode,
                context=context,
                extra_meta=extra_meta,
            )
        return {
            "request_id": request_id,
            "mode": mode,
            "autonomy": "max",
            "task": task,
            "context": context,
            "meta": extra_meta,
        }

    def _build_openresponses_payload(
        self,
        *,
        request_id: str,
        task: str,
        mode: str,
        context: dict[str, Any],
        extra_meta: dict[str, Any],
    ) -> dict[str, Any]:
        model = str(
            os.getenv("OPENCLAW_BRIDGE_MODEL")
            or self._config.openclaw.model
            or "openclaw"
        ).strip()
        metadata = self._stringify_metadata(
            {
                "request_id": request_id,
                "mode": mode,
                **extra_meta,
            }
        )
        payload: dict[str, Any] = {
            "model": model,
            "input": task,
            "metadata": metadata,
        }
        if context:
            # Keep context explicit for OpenResponses-compatible gateways.
            payload["instructions"] = "Task context (JSON):\n" + json.dumps(
                context, ensure_ascii=False, default=str
            )
            payload["metadata"]["context_json"] = json.dumps(
                context, ensure_ascii=False, default=str
            )
        return payload

    def _stringify_metadata(self, metadata: dict[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, value in metadata.items():
            key_name = str(key).strip()
            if not key_name:
                continue
            if isinstance(value, dict | list):
                out[key_name] = json.dumps(value, ensure_ascii=False, default=str)
            else:
                out[key_name] = str(value)
        return out
