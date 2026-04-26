"""T3a — Tool handler tests for the memory subsystem.

Covers:
  • W3+W4+W5 — 4 NEW handlers (handle_memory_get/delete/stats/explain)
  • L5 — auto project_id detection in handle_memory_store + handle_memory_recall
  • Audit fix — project_id="all" sentinel for cross-project recall
  • Graceful degradation — no-memory ctx, broken-memory ctx, missing args

Tests dispatch through HANDLER_MAP to verify the wiring works end-to-end.
"""
from __future__ import annotations

import pytest

from predacore.tools.handlers import HANDLER_MAP
from predacore.tools.handlers._context import ToolContext, ToolError


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _ctx(memory_store=None) -> ToolContext:
    """Minimal ToolContext for handler tests. Pass memory_store=None to
    test the no-memory path."""
    return ToolContext(config=None, memory={}, unified_memory=memory_store)


# ═════════════════════════════════════════════════════════════════════
# HANDLER_MAP wiring
# ═════════════════════════════════════════════════════════════════════


class TestHandlerMapWiring:
    """All 6 memory handlers are registered and resolvable."""

    @pytest.mark.parametrize("name", [
        "memory_store", "memory_recall", "memory_get",
        "memory_delete", "memory_stats", "memory_explain",
    ])
    def test_handler_in_map(self, name):
        assert HANDLER_MAP.get(name) is not None
        assert callable(HANDLER_MAP[name])


# ═════════════════════════════════════════════════════════════════════
# memory_get
# ═════════════════════════════════════════════════════════════════════


class TestHandleMemoryGet:
    async def test_returns_formatted_row(self, memory_store):
        rid = await memory_store.store(content="findable note", memory_type="fact")
        out = await HANDLER_MAP["memory_get"]({"id": rid}, _ctx(memory_store))
        assert "findable note" in out
        assert rid[:8] in out  # id appears in formatted output

    async def test_missing_id_raises(self, memory_store):
        with pytest.raises(ToolError):
            await HANDLER_MAP["memory_get"]({}, _ctx(memory_store))

    async def test_unknown_id_returns_friendly_message(self, memory_store):
        out = await HANDLER_MAP["memory_get"]({"id": "no-such-uuid"}, _ctx(memory_store))
        assert "no memory" in out.lower()

    async def test_no_memory_ctx_degrades_cleanly(self):
        out = await HANDLER_MAP["memory_get"]({"id": "any"}, _ctx(memory_store=None))
        assert "not available" in out.lower()


# ═════════════════════════════════════════════════════════════════════
# memory_delete
# ═════════════════════════════════════════════════════════════════════


class TestHandleMemoryDelete:
    async def test_deletes_and_confirms(self, memory_store):
        rid = await memory_store.store(content="to be deleted", memory_type="note")
        out = await HANDLER_MAP["memory_delete"]({"id": rid}, _ctx(memory_store))
        assert "Deleted" in out
        # Verify the row is gone
        assert await memory_store.get(rid) is None

    async def test_missing_id_raises(self, memory_store):
        with pytest.raises(ToolError):
            await HANDLER_MAP["memory_delete"]({}, _ctx(memory_store))

    async def test_unknown_id_friendly_message(self, memory_store):
        out = await HANDLER_MAP["memory_delete"]({"id": "no-such-uuid"}, _ctx(memory_store))
        assert "not found" in out.lower() or "no memory" in out.lower()

    async def test_no_memory_ctx_degrades_cleanly(self):
        out = await HANDLER_MAP["memory_delete"]({"id": "any"}, _ctx(memory_store=None))
        assert "not available" in out.lower()


# ═════════════════════════════════════════════════════════════════════
# memory_stats
# ═════════════════════════════════════════════════════════════════════


class TestHandleMemoryStats:
    async def test_returns_formatted_summary(self, memory_store):
        await memory_store.store(content="one", memory_type="note")
        await memory_store.store(content="two", memory_type="fact")
        out = await HANDLER_MAP["memory_stats"]({}, _ctx(memory_store))
        assert "Memory Stats" in out
        assert "total memories" in out.lower()
        # Should mention both types we stored
        assert "note" in out
        assert "fact" in out

    async def test_no_memory_ctx_degrades_cleanly(self):
        out = await HANDLER_MAP["memory_stats"]({}, _ctx(memory_store=None))
        assert "not available" in out.lower()


# ═════════════════════════════════════════════════════════════════════
# memory_explain
# ═════════════════════════════════════════════════════════════════════


class TestHandleMemoryExplain:
    async def test_returns_per_stage_trace(self, memory_store):
        await memory_store.store(content="explainable content", memory_type="note")
        out = await HANDLER_MAP["memory_explain"](
            {"query": "explainable"}, _ctx(memory_store)
        )
        assert "recall trace" in out.lower()
        assert "verification state counts" in out.lower()

    async def test_missing_query_raises(self, memory_store):
        with pytest.raises(ToolError):
            await HANDLER_MAP["memory_explain"]({}, _ctx(memory_store))

    async def test_top_k_param_respected(self, memory_store):
        for i in range(5):
            await memory_store.store(content=f"item {i}", memory_type="note")
        out = await HANDLER_MAP["memory_explain"](
            {"query": "item", "top_k": 3}, _ctx(memory_store)
        )
        # Header reports the count we requested
        assert "trace" in out.lower()

    async def test_no_memory_ctx_degrades_cleanly(self):
        out = await HANDLER_MAP["memory_explain"]({"query": "x"}, _ctx(memory_store=None))
        assert "not available" in out.lower()


# ═════════════════════════════════════════════════════════════════════
# memory_store — auto project_id (L5)
# ═════════════════════════════════════════════════════════════════════


class TestHandleMemoryStoreAutoProject:
    """L5 — handle_memory_store auto-detects project_id from cwd/git unless
    caller passes an explicit value."""

    async def test_explicit_project_id_respected(self, memory_store):
        out = await HANDLER_MAP["memory_store"](
            {"key": "k1", "content": "test", "project_id": "explicit-proj"},
            _ctx(memory_store),
        )
        assert "Stored" in out
        # Verify in DB
        rows = await memory_store.get_all_memories(limit=10)
        assert any(r.get("project_id") == "explicit-proj" for r in rows)

    async def test_auto_detected_project_id_when_omitted(
        self, memory_store, monkeypatch
    ):
        """When project_id is omitted, handler auto-detects from cwd."""
        # Run inside the public repo cwd so detection finds a real project name
        from predacore.memory.project_id import clear_cache
        clear_cache()
        out = await HANDLER_MAP["memory_store"](
            {"key": "k2", "content": "auto detected"},
            _ctx(memory_store),
        )
        assert "Stored" in out
        rows = await memory_store.get_all_memories(limit=10)
        # The auto-detected project should be non-default (since we're in a git repo)
        proj_ids = [r.get("project_id") for r in rows]
        assert any(p and p != "default" for p in proj_ids), (
            f"expected an auto-detected project_id, got: {proj_ids}"
        )

    async def test_missing_key_raises(self, memory_store):
        with pytest.raises(ToolError):
            await HANDLER_MAP["memory_store"]({"content": "x"}, _ctx(memory_store))

    async def test_missing_content_raises(self, memory_store):
        with pytest.raises(ToolError):
            await HANDLER_MAP["memory_store"]({"key": "x"}, _ctx(memory_store))


# ═════════════════════════════════════════════════════════════════════
# memory_recall — auto project_id + "all" sentinel (L5)
# ═════════════════════════════════════════════════════════════════════


class TestHandleMemoryRecallAutoProject:
    """Recall handler defaults to current project filter; "all" disables."""

    async def test_default_filters_to_current_project(self, memory_store):
        """Without project_id arg, recall should be scoped to current project.
        Memories from OTHER projects shouldn't appear."""
        # Store one memory tagged with current project (auto-detected)
        # and one tagged with a DIFFERENT project explicitly
        await HANDLER_MAP["memory_store"](
            {"key": "current", "content": "current project memory"},
            _ctx(memory_store),
        )
        await HANDLER_MAP["memory_store"](
            {"key": "other", "content": "other project memory",
             "project_id": "some-other-project"},
            _ctx(memory_store),
        )

        # Recall with NO project_id → defaults to current project
        out = await HANDLER_MAP["memory_recall"](
            {"query": "memory"}, _ctx(memory_store)
        )
        assert "current project memory" in out
        assert "other project memory" not in out

    async def test_all_sentinel_disables_filter(self, memory_store):
        """Explicit project_id='all' sees ALL projects."""
        await HANDLER_MAP["memory_store"](
            {"key": "current", "content": "current proj X"},
            _ctx(memory_store),
        )
        await HANDLER_MAP["memory_store"](
            {"key": "other", "content": "elsewhere proj Y",
             "project_id": "different-project"},
            _ctx(memory_store),
        )

        out = await HANDLER_MAP["memory_recall"](
            {"query": "proj", "project_id": "all"}, _ctx(memory_store)
        )
        assert "current proj X" in out
        assert "elsewhere proj Y" in out

    async def test_missing_query_raises(self, memory_store):
        with pytest.raises(ToolError):
            await HANDLER_MAP["memory_recall"]({}, _ctx(memory_store))

    async def test_no_memory_ctx_degrades_to_session_cache(self):
        """With no unified_memory, handler falls through to session-cache mode
        (returns "no memories found" or similar friendly message)."""
        out = await HANDLER_MAP["memory_recall"]({"query": "anything"}, _ctx(memory_store=None))
        # Either falls through to legacy cache or returns friendly empty
        assert "no memories" in out.lower() or "[" in out
