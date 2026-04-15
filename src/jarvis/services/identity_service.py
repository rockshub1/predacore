"""
JARVIS Cross-Channel Identity Service.

Maps {channel, channel_user_id} tuples to a canonical user_id,
enabling session continuity across Telegram, WebChat, CLI, etc.

Storage: SQLite at ~/.prometheus/users.db

NOTE: DBAdapter wiring intentionally skipped for this file.
IdentityService is synchronous and only runs inside the daemon process,
so direct SQLite access is appropriate.  If async access is ever needed,
wire DBAdapter following the pattern in outcome_store.py.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

# Link codes expire after this many seconds (15 minutes).
_LINK_CODE_TTL = 15 * 60

logger = logging.getLogger(__name__)


class IdentityService:
    """
    Maps channel-specific user IDs to a canonical identity.

    Usage:
        ids = IdentityService(Path("~/.prometheus"))
        canonical = ids.resolve("telegram", "123456789")
        # First time: creates new canonical user, links telegram ID
        # Second time: returns same canonical user

        # Link another channel to same user
        ids.link("webchat", "some-token", canonical)

        # Now both channels resolve to same user
        assert ids.resolve("telegram", "123456789") == ids.resolve("webchat", "some-token")
    """

    def __init__(self, home_dir: Path | str):
        self._db_path = Path(home_dir) / "users.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self._db_path), timeout=10)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=30000")
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                canonical_id TEXT PRIMARY KEY,
                display_name TEXT DEFAULT '',
                created_at REAL NOT NULL,
                last_seen_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS channel_links (
                channel TEXT NOT NULL,
                channel_user_id TEXT NOT NULL,
                canonical_id TEXT NOT NULL REFERENCES users(canonical_id),
                linked_at REAL NOT NULL,
                metadata TEXT DEFAULT '{}',
                PRIMARY KEY (channel, channel_user_id)
            );

            CREATE INDEX IF NOT EXISTS idx_links_canonical
                ON channel_links(canonical_id);
        """)
        conn.commit()
        logger.debug("Identity DB initialized at %s", self._db_path)

    def resolve(
        self,
        channel: str,
        channel_user_id: str,
        display_name: str = "",
    ) -> str:
        """
        Resolve a channel-specific user ID to a canonical ID.

        If no mapping exists, creates a new canonical user and links it.
        Always returns a stable canonical_id.
        """
        conn = self._get_conn()
        now = time.time()

        # Check existing link
        row = conn.execute(
            "SELECT canonical_id FROM channel_links WHERE channel = ? AND channel_user_id = ?",
            (channel, channel_user_id),
        ).fetchone()

        if row:
            canonical_id = row["canonical_id"]
            # Update last_seen
            conn.execute(
                "UPDATE users SET last_seen_at = ? WHERE canonical_id = ?",
                (now, canonical_id),
            )
            conn.commit()
            return canonical_id

        # New user — create canonical identity
        canonical_id = f"u-{uuid.uuid4().hex[:12]}"
        try:
            conn.execute(
                "INSERT INTO users (canonical_id, display_name, created_at, last_seen_at) VALUES (?, ?, ?, ?)",
                (canonical_id, display_name, now, now),
            )
            conn.execute(
                "INSERT INTO channel_links (channel, channel_user_id, canonical_id, linked_at) VALUES (?, ?, ?, ?)",
                (channel, channel_user_id, canonical_id, now),
            )
            conn.commit()
            logger.info(
                "New user %s created for %s:%s", canonical_id, channel, channel_user_id[:20]
            )
            return canonical_id
        except sqlite3.IntegrityError:
            # Race: another thread inserted first — fetch their result
            conn.rollback()
            row = conn.execute(
                "SELECT canonical_id FROM channel_links WHERE channel = ? AND channel_user_id = ?",
                (channel, channel_user_id),
            ).fetchone()
            if row:
                return row["canonical_id"]
            raise RuntimeError(
                f"IntegrityError during resolve but no existing link for {channel}:{channel_user_id}"
            )

    def link(
        self,
        channel: str,
        channel_user_id: str,
        canonical_id: str,
    ) -> bool:
        """Link a channel identity to an existing canonical user."""
        conn = self._get_conn()

        # Verify canonical user exists
        exists = conn.execute(
            "SELECT 1 FROM users WHERE canonical_id = ?", (canonical_id,)
        ).fetchone()
        if not exists:
            logger.warning("Cannot link — canonical user %s not found", canonical_id)
            return False

        # Check if this channel identity is already linked elsewhere
        existing = conn.execute(
            "SELECT canonical_id FROM channel_links WHERE channel = ? AND channel_user_id = ?",
            (channel, channel_user_id),
        ).fetchone()

        if existing:
            if existing["canonical_id"] == canonical_id:
                return True  # Already linked to same user
            # Re-link to new canonical user
            conn.execute(
                "UPDATE channel_links SET canonical_id = ?, linked_at = ? WHERE channel = ? AND channel_user_id = ?",
                (canonical_id, time.time(), channel, channel_user_id),
            )
        else:
            conn.execute(
                "INSERT INTO channel_links (channel, channel_user_id, canonical_id, linked_at) VALUES (?, ?, ?, ?)",
                (channel, channel_user_id, canonical_id, time.time()),
            )

        conn.commit()
        logger.info("Linked %s:%s \u2192 %s", channel, channel_user_id[:20], canonical_id)
        return True

    def get_links(self, canonical_id: str) -> list[dict[str, str]]:
        """Get all channel identities linked to a canonical user."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT channel, channel_user_id, linked_at FROM channel_links WHERE canonical_id = ?",
            (canonical_id,),
        ).fetchall()
        return [
            {"channel": r["channel"], "channel_user_id": r["channel_user_id"]}
            for r in rows
        ]

    def get_user_info(self, canonical_id: str) -> dict[str, Any] | None:
        """Get canonical user info."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM users WHERE canonical_id = ?", (canonical_id,)
        ).fetchone()
        if not row:
            return None
        links = self.get_links(canonical_id)
        return {
            "canonical_id": row["canonical_id"],
            "display_name": row["display_name"],
            "created_at": row["created_at"],
            "last_seen_at": row["last_seen_at"],
            "channels": links,
        }

    def generate_link_code(self, canonical_id: str) -> str:
        """Generate a short code that another channel can use to link to this user.

        The code expires after ``_LINK_CODE_TTL`` seconds (default 15 min).
        """
        conn = self._get_conn()
        code = uuid.uuid4().hex[:8]
        now = time.time()
        metadata = json.dumps({"expires_at": now + _LINK_CODE_TTL})
        conn.execute(
            """INSERT OR REPLACE INTO channel_links
               (channel, channel_user_id, canonical_id, linked_at, metadata)
               VALUES ('_link_code', ?, ?, ?, ?)""",
            (code, canonical_id, now, metadata),
        )
        conn.commit()
        return code

    def redeem_link_code(
        self,
        code: str,
        channel: str,
        channel_user_id: str,
    ) -> str | None:
        """
        Redeem a link code to bind a channel identity to an existing user.

        Returns canonical_id if successful, None if code invalid or expired.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT canonical_id, metadata FROM channel_links WHERE channel = '_link_code' AND channel_user_id = ?",
            (code,),
        ).fetchone()

        if not row:
            return None

        # Check expiry
        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}
        expires_at = meta.get("expires_at")
        if expires_at is not None and time.time() > expires_at:
            # Code has expired — delete it and return None
            conn.execute(
                "DELETE FROM channel_links WHERE channel = '_link_code' AND channel_user_id = ?",
                (code,),
            )
            conn.commit()
            logger.info("Link code %s expired", code)
            return None

        canonical_id = row["canonical_id"]

        # Link the new channel identity
        self.link(channel, channel_user_id, canonical_id)

        # Delete the used code
        conn.execute(
            "DELETE FROM channel_links WHERE channel = '_link_code' AND channel_user_id = ?",
            (code,),
        )
        conn.commit()
        logger.info(
            "Link code %s redeemed: %s:%s \u2192 %s",
            code, channel, channel_user_id[:20], canonical_id,
        )
        return canonical_id

    def unlink(self, channel: str, channel_user_id: str) -> bool:
        """Remove a single channel link. Returns True if a row was deleted."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM channel_links WHERE channel = ? AND channel_user_id = ?",
            (channel, channel_user_id),
        )
        conn.commit()
        removed = cursor.rowcount > 0
        if removed:
            logger.info("Unlinked %s:%s", channel, channel_user_id[:20])
        return removed

    def delete_user(self, canonical_id: str) -> bool:
        """Delete a canonical user and all associated channel links.

        Returns True if the user existed and was deleted.
        """
        conn = self._get_conn()
        # Delete links first (foreign-key target)
        conn.execute(
            "DELETE FROM channel_links WHERE canonical_id = ?", (canonical_id,)
        )
        cursor = conn.execute(
            "DELETE FROM users WHERE canonical_id = ?", (canonical_id,)
        )
        conn.commit()
        removed = cursor.rowcount > 0
        if removed:
            logger.info("Deleted user %s and all channel links", canonical_id)
        return removed

    def close(self) -> None:
        """Close the thread-local database connection (if open)."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
