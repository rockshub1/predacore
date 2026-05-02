"""Mastodon Channel Adapter — connects PredaCore to the Fediverse.

Mastodon is the most popular ActivityPub-based federated social network.
This adapter handles **direct messages** (Mastodon calls them "direct
visibility statuses") — public posts and replies are out of scope.

Setup
-----
1. Pick or create an account on any Mastodon instance.
2. Create an application: Settings → Development → New Application.
   Scopes needed: ``read:notifications read:statuses write:statuses``.
3. Save the access token.
4. Set environment variables (or ``~/.predacore/.env``)::

       MASTODON_API_BASE_URL=https://mastodon.social    # your instance
       MASTODON_ACCESS_TOKEN=...

5. Add ``mastodon`` to ``channels.enabled`` — daemon hot-attaches.

Implementation notes
--------------------
- Lazy-imports ``Mastodon.py``.
- Uses the streaming user API (websocket) for real-time DMs. Falls back
  to polling notifications every 30s if streaming isn't available on
  the instance.
- Replies use the same ``visibility=direct`` so the conversation thread
  stays private. Mentions the original sender so Mastodon threads it.
- 500-char per-status limit on most instances; we chunk before sending.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

# Default Mastodon instances cap statuses at 500 chars; some go higher
# (Pleroma, Akkoma). Stay conservative.
MASTODON_MAX_LENGTH = 500


def _strip_html(html: str) -> str:
    """Mastodon returns post content as HTML — convert to plain text.

    Cheap stripper good enough for status text. We don't need a full
    parser because we don't render anything; we just feed the plain
    text into the gateway.
    """
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"</p>\s*<p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _chunk_message(text: str, max_len: int = MASTODON_MAX_LENGTH) -> list[str]:
    """Split into status-safe chunks. Prefer paragraph boundaries."""
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split = text.rfind("\n\n", 0, max_len)
        if split == -1:
            split = text.rfind("\n", 0, max_len)
        if split == -1:
            split = max_len
        chunks.append(text[:split])
        text = text[split:].lstrip("\n")
    return chunks


class MastodonAdapter(ChannelAdapter):
    """Mastodon Direct Message adapter."""

    channel_name = "mastodon"
    channel_capabilities = {
        "supports_media": True,
        "supports_buttons": False,
        "supports_embeds": False,
        "supports_markdown": False,  # Mastodon strips most formatting
        "max_message_length": MASTODON_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        super().__init__()
        self.config = config
        cfg = getattr(config.channels, "__dict__", {}).get("mastodon", {}) or {}

        self._api_base_url = (
            cfg.get("api_base_url")
            or os.environ.get("MASTODON_API_BASE_URL", "")
        )
        self._access_token = (
            cfg.get("access_token")
            or os.environ.get("MASTODON_ACCESS_TOKEN", "")
        )
        self._poll_interval = float(
            os.environ.get("MASTODON_POLL_INTERVAL_S", "30")
        )

        self._client = None              # mastodon.Mastodon
        self._poll_task: asyncio.Task | None = None
        self._running = False
        self._self_account_id: int | None = None
        # Track the last notification id we processed so we don't replay.
        self._last_notification_id: int | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        if not (self._api_base_url and self._access_token):
            logger.error(
                "Mastodon: missing MASTODON_API_BASE_URL / "
                "MASTODON_ACCESS_TOKEN — set them in ~/.predacore/.env"
            )
            return
        try:
            from mastodon import Mastodon  # type: ignore[import-not-found]
        except ImportError:
            logger.error(
                "Mastodon: 'Mastodon.py' package not installed. "
                "pip install Mastodon.py  (or: pip install predacore[mastodon])"
            )
            return

        # Mastodon.py is sync; we'll call its methods inside asyncio.to_thread.
        self._client = Mastodon(
            access_token=self._access_token,
            api_base_url=self._api_base_url,
        )
        try:
            me = await asyncio.to_thread(self._client.account_verify_credentials)
            self._self_account_id = me.get("id")
        except Exception as exc:  # noqa: BLE001
            logger.error("Mastodon: credential verification failed: %s", exc)
            return

        self._running = True
        self._poll_task = asyncio.create_task(
            self._poll_notifications(), name="mastodon-poll",
        )
        logger.info(
            "Mastodon adapter started (instance=%s, account_id=%s)",
            self._api_base_url, self._self_account_id,
        )

    async def stop(self) -> None:
        self._running = False
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        logger.info("Mastodon adapter stopped")

    # ── Outgoing ──────────────────────────────────────────────────────

    async def send(self, message: OutgoingMessage) -> None:
        """Post a direct-visibility status as a reply.

        ``user_id`` is expected to be the recipient's @-handle (e.g.
        ``@alice@example.com``). The metadata dict carries the original
        status id we're threading on (``in_reply_to_id``).
        """
        if self._client is None:
            return
        in_reply_to_id = (message.metadata or {}).get("status_id")
        chunks = _chunk_message(message.text or "")
        # Mention the recipient so Mastodon delivers the DM properly.
        mention = message.user_id if message.user_id.startswith("@") else f"@{message.user_id}"
        for chunk in chunks:
            body = f"{mention} {chunk}".strip()
            try:
                await asyncio.to_thread(
                    self._client.status_post,
                    status=body[:MASTODON_MAX_LENGTH],
                    in_reply_to_id=in_reply_to_id,
                    visibility="direct",
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Mastodon send failed: %s", exc)
                return

    # ── Inbound ───────────────────────────────────────────────────────

    async def _poll_notifications(self) -> None:
        """Poll for ``mention`` notifications. Cheap on most instances."""
        while self._running:
            try:
                kwargs = {"types": ["mention"], "limit": 30}
                if self._last_notification_id:
                    kwargs["since_id"] = self._last_notification_id
                notes = await asyncio.to_thread(
                    self._client.notifications, **kwargs,
                )
                # Mastodon returns newest-first. Iterate oldest→newest.
                for note in reversed(notes or []):
                    self._last_notification_id = max(
                        self._last_notification_id or 0, int(note["id"]),
                    )
                    await self._handle_notification(note)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug("Mastodon poll error: %s", exc)
            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                return

    async def _handle_notification(self, note: dict[str, Any]) -> None:
        """Process a single notification."""
        status = note.get("status") or {}
        if not status:
            return
        if status.get("visibility") != "direct":
            return  # only handle DMs for now
        sender = note.get("account") or {}
        sender_acct = sender.get("acct", "")  # e.g. "alice@example.com"
        if not sender_acct:
            return
        # Strip our own mention from the text so the agent doesn't see it
        body = _strip_html(status.get("content", "") or "")
        body = re.sub(r"@\S+\s*", "", body, count=1).strip()
        if not body:
            return

        logger.info("Mastodon DM from @%s: %s", sender_acct, body[:100])
        if self._message_handler is None:
            return
        outgoing = await self._message_handler(
            IncomingMessage(
                channel=self.channel_name,
                user_id=f"@{sender_acct}",
                text=body,
                metadata={
                    "status_id": status.get("id"),
                    "sender_account_id": sender.get("id"),
                    "sender_url": sender.get("url"),
                },
            )
        )
        if outgoing is not None:
            await self.send(outgoing)
