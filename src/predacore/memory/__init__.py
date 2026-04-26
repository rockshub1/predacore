"""
PredaCore Unified Memory System.

Single-source-of-truth memory: SQLite storage + Rust compute kernel.
Inspired by Mem0, A-Mem (NeurIPS 2025), MAGMA (Jan 2026), EverMemOS (Jan 2026).

Architecture:
- store.py         UnifiedMemoryStore (SQLite + in-RAM vector cache, HNSW)
- retriever.py     MemoryRetriever (5-section budgeted context builder)
- consolidator.py  MemoryConsolidator (decay, entity extract, auto-link, merge, prune)
- chunker.py       Semantic chunker (AST / markdown / brace / window strategies)
- safety.py        Ingress secret scanner + .memoryignore matcher
- healer.py        Self-healing daemon (audit, sweep, snapshot, integrity)

Rust kernel (predacore_core) is a HARD dependency — no Python fallbacks.

Usage:
    from predacore.memory import UnifiedMemoryStore, MemoryRetriever, MemoryConsolidator

    store = UnifiedMemoryStore(db_path="~/.predacore/memory/unified_memory.db", embedding_client=embed)
    retriever = MemoryRetriever(store, embed)
    consolidator = MemoryConsolidator(store, llm)

    await store.store("User prefers dark mode", memory_type="preference", importance=3)
    context = await retriever.build_context("how do I enable dark mode?", user_id="alice")
    stats = await consolidator.consolidate()

    # Auto-indexing of edited files + post-git-command sync:
    await store.reindex_file("/path/to/edited.py", project_id="proj")
    await store.sync_git_changes(repo_root, prior_head=old_sha)  # full coverage
    await store.warmup_embedder()  # one-shot at boot

    # Background drift / orphan / snapshot daemon:
    from predacore.memory import Healer
    healer = Healer(store, user="shubham")
    healer.start()  # mirrors mcp_server.py wiring; thread runs in background
"""
from .chunker import chunk_text, safe_read_text
from .consolidator import MemoryConsolidator
from .healer import Healer
from .retriever import MemoryRetriever
from .safety import MemoryIgnore, is_sensitive_path, scan_for_secrets
from .store import UnifiedMemoryStore

__all__ = [
    # Core (existing)
    "UnifiedMemoryStore",
    "MemoryRetriever",
    "MemoryConsolidator",
    # Chunker
    "chunk_text",
    "safe_read_text",
    # Safety
    "scan_for_secrets",
    "is_sensitive_path",
    "MemoryIgnore",
    # Healer
    "Healer",
]
