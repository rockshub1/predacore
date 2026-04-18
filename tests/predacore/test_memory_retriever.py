from __future__ import annotations

import pytest

from src.predacore.memory.retriever import MemoryRetriever


class _FakeStore:
    async def recall(self, **kwargs):
        memory_types = kwargs.get("memory_types")
        if memory_types == ["preference"]:
            return []
        return [
            (
                {
                    "content": "PredaCore should remember long-term preferences.",
                    "memory_type": "fact",
                    "importance": 3,
                    "created_at": "2026-03-03T16:23:18.985501+00:00",
                },
                0.91,
            )
        ]

    async def get_all_memories(self, **kwargs):
        return []

    async def get_recent_episodes(self, limit: int = 3):
        return []

    async def list_entities(self, limit: int = 500):
        return []


@pytest.mark.asyncio
async def test_build_context_accepts_iso_created_at_strings():
    retriever = MemoryRetriever(_FakeStore())

    context = await retriever.build_context(
        query="What do you remember about my preferences?",
        user_id="u1",
    )

    assert "## Relevant Memories" in context
    assert "PredaCore should remember long-term preferences." in context
