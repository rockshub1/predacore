"""
Minimal task/result store abstraction for DAF.
Supports in-memory and optional Redis-backed persistence (if redis is available).

Moved from src/daf/task_store.py into dynamic_agent_fabric/ as the canonical location.
"""
import asyncio
import os
from typing import Any

try:
    import redis.asyncio as redis
except ImportError:
    redis = None


class AbstractTaskStore:
    async def set_task(self, task_id: str, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    async def set_result(self, task_id: str, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    async def get_result(self, task_id: str) -> dict[str, Any] | None:
        raise NotImplementedError


class MemoryTaskStore(AbstractTaskStore):
    _MAX_ENTRIES = 10_000

    def __init__(self):
        self._tasks: dict[str, dict[str, Any]] = {}
        self._results: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    def _evict_oldest(self, store: dict, max_size: int | None = None) -> None:
        """Remove oldest entries when store exceeds max size."""
        limit = max_size or self._MAX_ENTRIES
        while len(store) > limit:
            oldest_key = next(iter(store))
            del store[oldest_key]

    async def set_task(self, task_id: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            self._tasks[task_id] = payload
            self._evict_oldest(self._tasks)

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        async with self._lock:
            return dict(self._tasks.get(task_id)) if task_id in self._tasks else None

    async def set_result(self, task_id: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            self._results[task_id] = payload
            self._evict_oldest(self._results)

    async def get_result(self, task_id: str) -> dict[str, Any] | None:
        async with self._lock:
            return (
                dict(self._results.get(task_id)) if task_id in self._results else None
            )


class RedisTaskStore(AbstractTaskStore):
    def __init__(self, url: str, namespace: str = "daf"):
        if not redis:
            raise RuntimeError("redis is not available")
        self._client = redis.from_url(url, max_connections=10)
        self._ns = namespace

    def _key(self, kind: str, task_id: str) -> str:
        return f"{self._ns}:{kind}:{task_id}"

    async def set_task(self, task_id: str, payload: dict[str, Any]) -> None:
        await self._client.set(self._key("task", task_id), json_dumps(payload))

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        raw = await self._client.get(self._key("task", task_id))
        return json_loads(raw) if raw else None

    async def set_result(self, task_id: str, payload: dict[str, Any]) -> None:
        await self._client.set(self._key("result", task_id), json_dumps(payload))

    async def get_result(self, task_id: str) -> dict[str, Any] | None:
        raw = await self._client.get(self._key("result", task_id))
        return json_loads(raw) if raw else None


def json_dumps(obj: dict[str, Any]) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


def json_loads(raw: Any) -> dict[str, Any]:
    import json

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}


def get_task_store() -> AbstractTaskStore:
    backend = os.getenv("DAF_TASK_STORE", "memory").lower()
    if backend == "redis" and redis:
        url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        try:
            return RedisTaskStore(url=url)
        except (OSError, ConnectionError, RuntimeError):
            pass
    return MemoryTaskStore()
