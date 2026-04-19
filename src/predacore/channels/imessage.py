"""
iMessage Channel Adapter — macOS only.

Apple doesn't expose a bot API for iMessage, so we piggy-back on the
native Messages.app:

- **Outbound** — AppleScript told to ``send "..." to buddy "..."``. The
  subprocess runs as the current user so messages ship from *your*
  account (no dedicated bot user).
- **Inbound** — poll the Messages SQLite database at
  ``~/Library/Messages/chat.db``. New rows in ``message`` become
  incoming events. Requires macOS **Full Disk Access** granted to the
  terminal / daemon (System Settings → Privacy & Security).

Graceful no-op on Linux/Windows (registers but refuses to start).

Privacy defaults
----------------
- Responds only to DMs. Group threads are intentionally skipped until
  we add an allow-list.
- Outbound messages route through the same ``handle`` that received
  the inbound, so replies land in the right conversation.

Config (optional — env vars)
----------------------------
``IMESSAGE_POLL_INTERVAL_S`` — seconds between DB polls (default 2.5)
``IMESSAGE_ALLOW_GROUPS``    — ``"true"`` to reply in group chats too
"""
from __future__ import annotations

import asyncio
import logging
import os
import platform
import sqlite3
import subprocess
from pathlib import Path

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

IMESSAGE_MAX_LENGTH = 20_000  # iMessage gets unhappy long before this

# Core-Data "nanosecond epoch" used by chat.db: nanoseconds since 2001-01-01.
_MAC_EPOCH_OFFSET = 978_307_200  # unix-seconds at 2001-01-01 UTC


def _chunk_message(text: str, max_len: int = IMESSAGE_MAX_LENGTH) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split = text.rfind("\n", 0, max_len)
        if split == -1:
            split = max_len
        chunks.append(text[:split])
        text = text[split:].lstrip("\n")
    return chunks


def _escape_applescript(text: str) -> str:
    """Escape a string for safe inclusion inside an AppleScript literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


class IMessageAdapter(ChannelAdapter):
    """iMessage via Messages.app + chat.db polling."""

    channel_name = "imessage"
    channel_capabilities = {
        "supports_media": False,
        "supports_buttons": False,
        "supports_embeds": False,
        "supports_markdown": False,
        "max_message_length": IMESSAGE_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        self.config = config
        self._message_handler = None
        self._poll_task: asyncio.Task | None = None
        self._running = False
        self._last_rowid: int = 0
        # Captured in start() so the polling thread can dispatch coroutines
        # back onto the event loop. ``asyncio.get_event_loop()`` from a
        # worker thread is unreliable on Python 3.12+ (returns a new loop),
        # so we grab the running loop upfront instead.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._chat_db = Path.home() / "Library" / "Messages" / "chat.db"
        self._poll_interval = float(os.getenv("IMESSAGE_POLL_INTERVAL_S", "2.5"))
        self._allow_groups = os.getenv("IMESSAGE_ALLOW_GROUPS", "").lower() in {
            "1", "true", "yes", "on",
        }

    async def start(self) -> None:
        if platform.system() != "Darwin":
            logger.warning(
                "iMessage adapter registered but platform is %s — skipping start",
                platform.system(),
            )
            return
        if not self._chat_db.exists():
            logger.error(
                "iMessage: chat.db not found at %s — Messages.app not set up?",
                self._chat_db,
            )
            return
        if not os.access(self._chat_db, os.R_OK):
            logger.error(
                "iMessage: chat.db not readable. Grant Full Disk Access to "
                "your terminal (System Settings → Privacy & Security → "
                "Full Disk Access)."
            )
            return

        # Seed _last_rowid to "now" so we don't replay the entire history.
        try:
            with sqlite3.connect(f"file:{self._chat_db}?mode=ro", uri=True) as conn:
                cur = conn.execute("SELECT COALESCE(MAX(ROWID), 0) FROM message")
                self._last_rowid = int(cur.fetchone()[0])
        except sqlite3.DatabaseError as exc:
            logger.error("iMessage: chat.db read failed: %s", exc)
            return

        self._running = True
        self._loop = asyncio.get_running_loop()
        self._poll_task = asyncio.create_task(self._poll_loop(), name="imessage-poll")
        logger.info(
            "iMessage adapter started (chat.db=%s, poll=%.1fs)",
            self._chat_db, self._poll_interval,
        )

    async def stop(self) -> None:
        self._running = False
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass
        logger.info("iMessage adapter stopped")

    async def send(self, message: OutgoingMessage) -> None:
        """Dispatch via AppleScript. ``user_id`` is the contact's phone/email."""
        if platform.system() != "Darwin":
            return
        for chunk in _chunk_message(message.text):
            script = _SEND_APPLESCRIPT.format(
                text=_escape_applescript(chunk),
                buddy=_escape_applescript(message.user_id),
            )
            try:
                proc = await asyncio.create_subprocess_exec(
                    "osascript", "-e", script,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
                if proc.returncode != 0:
                    logger.error(
                        "iMessage osascript failed (rc=%s): %s",
                        proc.returncode,
                        stderr.decode("utf-8", "replace")[:200],
                    )
                    return
            except asyncio.TimeoutError:
                logger.error("iMessage send timed out")
                return
            except FileNotFoundError:
                logger.error("iMessage: osascript not found (not on macOS?)")
                return

    # ── Polling loop ─────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await asyncio.to_thread(self._scan_new_messages_sync)
            except Exception as exc:
                logger.debug("iMessage poll error: %s", exc, exc_info=True)
            await asyncio.sleep(self._poll_interval)

    def _scan_new_messages_sync(self) -> None:
        """Read new chat.db rows > _last_rowid and dispatch them.

        Synchronous — we run it in a thread pool (``asyncio.to_thread``)
        because sqlite3 is blocking and the DB is small enough that
        this has negligible cost at a 2-3s cadence.
        """
        rows: list[tuple[int, str, str, int]] = []
        try:
            with sqlite3.connect(f"file:{self._chat_db}?mode=ro", uri=True) as conn:
                conn.row_factory = None
                cur = conn.execute(
                    """
                    SELECT
                        m.ROWID,
                        m.text,
                        h.id,
                        CASE WHEN c.chat_identifier IS NOT NULL AND c.style = 43
                             THEN 1 ELSE 0 END
                    FROM message m
                    LEFT JOIN handle h ON m.handle_id = h.ROWID
                    LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
                    LEFT JOIN chat c ON c.ROWID = cmj.chat_id
                    WHERE m.ROWID > ?
                      AND m.is_from_me = 0
                      AND m.text IS NOT NULL
                      AND m.text != ''
                    ORDER BY m.ROWID ASC
                    LIMIT 200
                    """,
                    (self._last_rowid,),
                )
                rows = cur.fetchall()
        except sqlite3.DatabaseError as exc:
            logger.debug("iMessage DB read failed: %s", exc)
            return

        if not rows:
            return
        for rowid, text, handle_id, is_group in rows:
            self._last_rowid = max(self._last_rowid, int(rowid))
            if not text or not handle_id:
                continue
            if is_group and not self._allow_groups:
                continue
            # Dispatch on the event loop (we're inside a thread here).
            # Use the loop captured in start() — asyncio.get_event_loop() is
            # unreliable from a worker thread on Python 3.12+.
            if self._loop is not None:
                asyncio.run_coroutine_threadsafe(
                    self._dispatch(text.strip(), handle_id, bool(is_group)),
                    self._loop,
                )

    async def _dispatch(self, text: str, handle_id: str, is_group: bool) -> None:
        logger.info(
            "iMessage %s from %s: %s",
            "group" if is_group else "DM", handle_id, text[:100],
        )
        if self._message_handler is None:
            return
        outgoing = await self._message_handler(
            IncomingMessage(
                channel=self.channel_name,
                user_id=handle_id,
                text=text,
                metadata={"is_group": is_group},
            )
        )
        if outgoing is not None:
            await self.send(outgoing)


# AppleScript that sends one message through Messages.app. We target the
# default iMessage service explicitly so SMS fallbacks don't surprise us
# (send via ``service 1st service whose service type is iMessage``).
_SEND_APPLESCRIPT = '''tell application "Messages"
    set targetService to 1st service whose service type = iMessage
    set targetBuddy to buddy "{buddy}" of targetService
    send "{text}" to targetBuddy
end tell'''
