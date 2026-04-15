"""
FastAPI health/readiness endpoints for Knowledge Nexus.
"""
from fastapi import FastAPI

app = FastAPI(title="Knowledge Nexus Health")


@app.get("/healthz")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    return {"ready": True}
