import pytest

from predacore.agents.daf.task_store import MemoryTaskStore

pytestmark = pytest.mark.asyncio

async def test_memory_task_store_roundtrip():
    store = MemoryTaskStore()
    await store.set_task("t1", {"foo": "bar"})
    got = await store.get_task("t1")
    assert got == {"foo": "bar"}
    await store.set_result("t1", {"ok": True})
    res = await store.get_result("t1")
    assert res == {"ok": True}

async def test_memory_task_store_missing():
    store = MemoryTaskStore()
    assert await store.get_task("missing") is None
    assert await store.get_result("missing") is None
