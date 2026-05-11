"""
Shared caching utilities for PredaCore.

Provides TTLCache and LRUCache to reduce per-message latency by
avoiding repeated computation (embeddings, vector searches, identity
resolution, file reads, etc.).
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    """
    Simple dict-based cache with per-key TTL expiry.

    Uses time.monotonic() so clock adjustments don't cause spurious
    invalidations.

    L63 (Wave 12) — guarded by an ``RLock`` so cross-thread ``set()``
    (which iterates the store, deletes expired keys, then inserts) can't
    race a concurrent ``get()``/``invalidate()`` mid-iteration and raise
    ``RuntimeError: dictionary changed size during iteration``. RLock so
    nested ``get``-during-``set`` paths don't self-deadlock.
    """

    _DEFAULT_MAX_SIZE = 10_000

    def __init__(self, max_size: int = _DEFAULT_MAX_SIZE) -> None:
        # key -> (value, expires_at_monotonic)
        self._store: dict[str, tuple[Any, float]] = {}
        self._max_size = max_size
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        """Return cached value or None if missing/expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() >= expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        """Store a value with the given TTL (seconds)."""
        with self._lock:
            # Evict expired entries when approaching max size
            if len(self._store) >= self._max_size:
                now = time.monotonic()
                expired = [k for k, (_, exp) in self._store.items() if now >= exp]
                for k in expired:
                    del self._store[k]
                # If still at capacity, evict oldest entries
                while len(self._store) >= self._max_size:
                    oldest_key = next(iter(self._store))
                    del self._store[oldest_key]
            self._store[key] = (value, time.monotonic() + ttl_seconds)

    def invalidate(self, key: str) -> None:
        """Remove a specific key from the cache."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


class LRUCache:
    """
    Bounded LRU cache with optional per-key TTL.

    Uses collections.OrderedDict for O(1) move-to-end on access.

    L63 (Wave 12) — guarded by an ``RLock`` for the same reason as
    ``TTLCache``. ``OrderedDict.move_to_end`` + ``popitem`` are not
    thread-safe under concurrent access.
    """

    def __init__(self, max_size: int = 128) -> None:
        self.max_size = max_size
        # key -> (value, expires_at_monotonic | None)
        self._store: OrderedDict[str, tuple[Any, float | None]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        """Return cached value or None if missing/expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at is not None and time.monotonic() >= expires_at:
                del self._store[key]
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl_seconds: float | None = None) -> None:
        """Store a value, optionally with TTL. Evicts LRU entry if at capacity."""
        with self._lock:
            expires_at = (time.monotonic() + ttl_seconds) if ttl_seconds is not None else None
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = (value, expires_at)
            else:
                if len(self._store) >= self.max_size:
                    self._store.popitem(last=False)  # Evict oldest
                self._store[key] = (value, expires_at)

    def invalidate(self, key: str) -> None:
        """Remove a specific key."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


def hash_key(*args: Any) -> str:
    """Create a stable cache key from arbitrary arguments using SHA-256.

    L62 fix: insert a record-separator byte (``\\x1e``, the ASCII RS
    control char which is illegal in JSON / printable text) between
    arguments so ``hash_key("ab", "c") != hash_key("a", "bc")``. Today's
    callers pass fixed-shape tuples so collisions weren't reachable, but
    the helper is exported and a future caller can't accidentally trip
    cache poisoning.
    """
    h = hashlib.sha256()
    sep = b"\x1e"
    for i, arg in enumerate(args):
        if i:
            h.update(sep)
        h.update(str(arg).encode("utf-8"))
    return h.hexdigest()
