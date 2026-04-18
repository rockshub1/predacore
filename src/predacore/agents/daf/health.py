"""
Simple health/readiness checks for DAF components.

Moved from src/daf/health.py into dynamic_agent_fabric/ as the canonical location.
"""
from typing import Any

from .task_store import AbstractTaskStore


async def check_task_store(store: AbstractTaskStore) -> bool:
    try:
        await store.set_task("__health__", {"ok": True})
        res = await store.get_task("__health__")
        return bool(res)
    except Exception:
        return False


async def health_status(store: AbstractTaskStore) -> dict[str, Any]:
    ok_store = await check_task_store(store)
    return {"ok": ok_store, "store": store.__class__.__name__}
