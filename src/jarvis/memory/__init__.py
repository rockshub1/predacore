"""
JARVIS Unified Memory System.

Single-source-of-truth memory: SQLite storage + Rust compute kernel.
Inspired by Mem0, A-Mem (NeurIPS 2025), MAGMA (Jan 2026), EverMemOS (Jan 2026).

Architecture:
- store.py         UnifiedMemoryStore (SQLite + in-RAM vector cache)
- retriever.py     MemoryRetriever (5-section budgeted context builder)
- consolidator.py  MemoryConsolidator (decay, entity extract, auto-link, merge, prune)

Rust kernel (jarvis_core) is a HARD dependency — no Python fallbacks.

Usage:
    from jarvis.memory import UnifiedMemoryStore, MemoryRetriever, MemoryConsolidator

    store = UnifiedMemoryStore(db_path="~/.prometheus/memory/unified_memory.db", embedding_client=embed)
    retriever = MemoryRetriever(store, embed)
    consolidator = MemoryConsolidator(store, llm)

    await store.store("User prefers dark mode", memory_type="preference", importance=3)
    context = await retriever.build_context("how do I enable dark mode?", user_id="shubham")
    stats = await consolidator.consolidate()
"""
from .consolidator import MemoryConsolidator
from .retriever import MemoryRetriever
from .store import UnifiedMemoryStore

__all__ = [
    "UnifiedMemoryStore",
    "MemoryRetriever",
    "MemoryConsolidator",
]
