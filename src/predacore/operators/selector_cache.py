"""Browser selector cache — the "faster than humans" path (T4b).

When the agent clicks the same element twice (login button, search bar,
"compose tweet"), the second click should not pay the LLM-grounding cost.
This module persists ``(domain, intent) → xpath`` mappings keyed by a
stable hash, with CDP-side verification at reuse time so a stale cache
entry never produces a phantom click.

Storage
-------
A dedicated SQLite table ``browser_selector_cache`` inside the existing
PredaCore memory DB. We piggyback on the memory DB so:

* Backups + integrity checks (the healer) apply to the cache too.
* No new on-disk file the user has to know about.
* But a dedicated *table* — not the ``memories`` table — so the cache
  doesn't pollute semantic-search ranking and key lookups are O(log n)
  instead of fan-out.

Schema:

    CREATE TABLE browser_selector_cache (
        domain         TEXT NOT NULL,
        intent_hash    TEXT NOT NULL,    -- sha256(f"{role}|{lower(text)}")
        xpath          TEXT NOT NULL,    -- CSS selector (we call it xpath
                                         -- historically; same shape as the
                                         -- existing ``_resolve`` output)
        intent_text    TEXT NOT NULL,
        role           TEXT,
        label          TEXT,
        success_count  INTEGER NOT NULL DEFAULT 1,
        last_used_at   REAL    NOT NULL,
        created_at     REAL    NOT NULL,
        PRIMARY KEY (domain, intent_hash)
    );

Verification (T7-aligned)
-------------------------
At lookup time we MUST validate the cached selector still resolves on
the current page before clicking, otherwise a UI redesign would silently
mis-click. ``BrowserBridge`` calls :meth:`SelectorCache.lookup` to get the
candidate, then runs a CDP ``document.querySelector(xpath)`` to confirm
the element exists. Only after that does it click. This is the same
"verify-with-code" pattern from T7 but for live DOM.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS browser_selector_cache (
    domain         TEXT NOT NULL,
    intent_hash    TEXT NOT NULL,
    xpath          TEXT NOT NULL,
    intent_text    TEXT NOT NULL,
    role           TEXT,
    label          TEXT,
    success_count  INTEGER NOT NULL DEFAULT 1,
    last_used_at   REAL    NOT NULL,
    created_at     REAL    NOT NULL,
    PRIMARY KEY (domain, intent_hash)
);
CREATE INDEX IF NOT EXISTS idx_browser_cache_domain
    ON browser_selector_cache (domain);
CREATE INDEX IF NOT EXISTS idx_browser_cache_last_used
    ON browser_selector_cache (last_used_at DESC);
"""


@dataclass
class SelectorEntry:
    """One cached ``(domain, intent) → selector`` row."""
    domain: str
    intent_hash: str
    xpath: str
    intent_text: str
    role: str
    label: str
    success_count: int
    last_used_at: float
    created_at: float


def _normalize_domain(url_or_host: str) -> str:
    """Extract the bare host (no scheme, no port, no trailing slash).

    Cache keys live per-host, not per-URL — clicking "Login" on
    ``twitter.com/home`` should reuse the entry stored from
    ``twitter.com/i/flow/login``.
    """
    s = (url_or_host or "").strip()
    if not s:
        return ""
    if "://" not in s:
        s = "http://" + s  # urlparse needs a scheme
    parsed = urlparse(s)
    host = (parsed.hostname or "").lower()
    return host


def intent_hash(text: str, role: str = "") -> str:
    """Stable key from a click intent. Lowercase, strip, role-prefixed.

    A future enhancement could BGE-embed the intent so "log in" and
    "sign in" map to the same selector — for now we keep it exact-match.
    Override at call site by canonicalizing before passing.
    """
    norm = f"{(role or '').strip().lower()}|{(text or '').strip().lower()}"
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


class SelectorCache:
    """Thread-safe key-value store for browser selectors.

    Constructed from a path (lives next to the memory DB by default).
    Methods are sync — the existing :class:`BrowserBridge` calls into them
    from async contexts via ``asyncio.to_thread``. Keeping the surface
    sync means we don't fight SQLite's blocking I/O model.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        # SQLite connections aren't safe across threads — make a fresh one
        # per call. The DB itself handles concurrency via WAL.
        c = sqlite3.connect(self._db_path, timeout=5.0)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _init_schema(self) -> None:
        with self._lock, self._conn() as c:
            c.executescript(_SCHEMA_SQL)
            c.commit()

    # ── Public API ────────────────────────────────────────────────────

    def lookup(
        self, domain: str, text: str, role: str = "",
    ) -> SelectorEntry | None:
        """Return the cached entry for ``(domain, text, role)``, or None.

        Does NOT verify the selector still resolves on the live page —
        that's the caller's job (CDP query). We just hand back the
        candidate and let the caller validate.
        """
        d = _normalize_domain(domain)
        if not d:
            return None
        h = intent_hash(text, role)
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT domain, intent_hash, xpath, intent_text, role, label, "
                "       success_count, last_used_at, created_at "
                "FROM browser_selector_cache "
                "WHERE domain = ? AND intent_hash = ?",
                (d, h),
            ).fetchone()
        if row is None:
            return None
        return SelectorEntry(*row)

    def record(
        self, *, domain: str, text: str, xpath: str,
        role: str = "", label: str = "",
    ) -> None:
        """Upsert a successful click into the cache.

        Idempotent: if the (domain, intent_hash) row already exists,
        increments ``success_count`` and bumps ``last_used_at`` instead
        of duplicating. The xpath wins on conflict — if the resolver
        produced a different selector this time, the new one is more
        likely to be correct (DOM may have changed; old selector might
        be the one we just invalidated).
        """
        d = _normalize_domain(domain)
        if not d or not xpath:
            return
        h = intent_hash(text, role)
        now = time.time()
        with self._lock, self._conn() as c:
            c.execute(
                """INSERT INTO browser_selector_cache
                       (domain, intent_hash, xpath, intent_text, role, label,
                        success_count, last_used_at, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                   ON CONFLICT(domain, intent_hash) DO UPDATE SET
                       xpath = excluded.xpath,
                       label = excluded.label,
                       success_count = success_count + 1,
                       last_used_at = excluded.last_used_at""",
                (d, h, xpath, text, role, label, now, now),
            )
            c.commit()

    def bump_use(
        self, *, domain: str, text: str, role: str = "",
    ) -> None:
        """Record a cache hit (increments success_count + last_used_at).
        Called after a cached selector verified + clicked successfully.
        """
        d = _normalize_domain(domain)
        if not d:
            return
        h = intent_hash(text, role)
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE browser_selector_cache "
                "SET success_count = success_count + 1, last_used_at = ? "
                "WHERE domain = ? AND intent_hash = ?",
                (time.time(), d, h),
            )
            c.commit()

    def invalidate(
        self, *, domain: str, text: str, role: str = "",
    ) -> bool:
        """Drop a cache entry whose verification failed. Returns True if
        a row was deleted."""
        d = _normalize_domain(domain)
        if not d:
            return False
        h = intent_hash(text, role)
        with self._lock, self._conn() as c:
            cur = c.execute(
                "DELETE FROM browser_selector_cache "
                "WHERE domain = ? AND intent_hash = ?",
                (d, h),
            )
            c.commit()
            return cur.rowcount > 0

    def stats(self) -> dict[str, Any]:
        """Cache health snapshot — total rows, distinct domains, top hits."""
        with self._lock, self._conn() as c:
            total = c.execute(
                "SELECT COUNT(*) FROM browser_selector_cache"
            ).fetchone()[0]
            domains = c.execute(
                "SELECT COUNT(DISTINCT domain) FROM browser_selector_cache"
            ).fetchone()[0]
            top = c.execute(
                "SELECT domain, intent_text, success_count "
                "FROM browser_selector_cache "
                "ORDER BY success_count DESC LIMIT 5"
            ).fetchall()
        return {
            "total_rows": total,
            "distinct_domains": domains,
            "top_intents": [
                {"domain": d, "intent": t, "uses": n} for d, t, n in top
            ],
        }
