"""
Minimal HTTP health/readiness for DAF using FastAPI.
"""
from fastapi import FastAPI

from .health import health_status
from .task_store import get_task_store

app = FastAPI(title="DAF Health")


@app.get("/healthz")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    store = get_task_store()
    status = await health_status(store)
    return {"ready": status.get("ok", False), "store": status}
