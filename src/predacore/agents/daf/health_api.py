"""
Minimal HTTP health/readiness for DAF using FastAPI.

L51 (Phase 7): the health probe used to call `get_task_store()` per
request, which returned a FRESH `MemoryTaskStore` unrelated to the
live service. The probe wrote to a disconnected store and falsely
reported "ready" even when the real service was wedged.

Fix: `set_live_store(store)` lets the daemon inject the live store
at boot. The probe reads from that. Falls back to the legacy
`get_task_store()` only if no live store was injected (e.g. when the
FastAPI app is launched standalone for tests).
"""
from typing import Any

from fastapi import FastAPI

from .health import health_status
from .task_store import AbstractTaskStore, get_task_store

app = FastAPI(title="DAF Health")

# L51: module-level pointer to the live task store. Daemon calls
# set_live_store() at boot to wire the running instance in.
_LIVE_STORE: AbstractTaskStore | None = None


def set_live_store(store: AbstractTaskStore | None) -> None:
    """Inject the live DAF service's task store. Call from daemon boot."""
    global _LIVE_STORE
    _LIVE_STORE = store


def _resolve_store() -> AbstractTaskStore:
    """Prefer the injected live store; fall back to a fresh one only
    when nothing was wired (standalone test mode)."""
    return _LIVE_STORE if _LIVE_STORE is not None else get_task_store()


@app.get("/healthz")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, Any]:
    store = _resolve_store()
    status = await health_status(store)
    return {
        "ready": status.get("ok", False),
        "store": status,
        "live_store_wired": _LIVE_STORE is not None,
    }
