"""Bluesky DM Channel Adapter — connects PredaCore to the Bluesky network.

Bluesky uses the AT Protocol (atproto). DMs are a recent addition; this
adapter wires PredaCore into the chat conversation API so users can
talk to the bot from any Bluesky client.

Setup
-----
1. Create a Bluesky account at bsky.app.
2. Generate an **app password** at Settings → Privacy & Security →
   App passwords. Don't use your main password.
3. Set environment variables (or ``~/.predacore/.env``)::

       BLUESKY_HANDLE=yourbot.bsky.social
       BLUESKY_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx

4. Add ``bluesky`` to ``channels.enabled`` — daemon hot-attaches.

Implementation notes
--------------------
- Lazy-imports ``atproto``.
- DM-only; public posts are out of scope for this adapter.
- Polls the conversation list every 15s. Bluesky does have a firehose
  but DMs aren't on it — polling is the canonical pattern.
- 1000-char limit per DM message; we chunk above that.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

BLUESKY_DM_MAX_LENGTH = 1000


def _chunk_message(text: str, max_len: int = BLUESKY_DM_MAX_LENGTH) -> list[str]:
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


class BlueskyAdapter(ChannelAdapter):
    """Bluesky DM adapter via atproto."""

    channel_name = "bluesky"
    channel_capabilities = {
        "supports_media": False,    # DM media support is limited; skip for now
        "supports_buttons": False,
        "supports_embeds": False,
        "supports_markdown": False, # Bluesky DMs render plain text
        "max_message_length": BLUESKY_DM_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        super().__init__()
        self.config = config
        cfg = getattr(config.channels, "__dict__", {}).get("bluesky", {}) or {}

        self._handle = (
            cfg.get("handle")
            or os.environ.get("BLUESKY_HANDLE", "")
        )
        self._app_password = (
            cfg.get("app_password")
            or os.environ.get("BLUESKY_APP_PASSWORD", "")
        )
        self._poll_interval = float(
            os.environ.get("BLUESKY_POLL_INTERVAL_S", "15")
        )

        self._client = None              # atproto.Client
        self._chat_proxy = None           # client.with_bsky_chat_proxy()
        self._poll_task: asyncio.Task | None = None
        self._running = False
        self._self_did: str | None = None
        # Track last seen message rkey per conversation to avoid replays.
        self._last_seen_rkey: dict[str, str] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        if not (self._handle and self._app_password):
            logger.error(
                "Bluesky: missing BLUESKY_HANDLE / BLUESKY_APP_PASSWORD — "
                "set them in ~/.predacore/.env"
            )
            return
        try:
            from atproto import Client  # type: ignore[import-not-found]
        except ImportError:
            logger.error(
                "Bluesky: 'atproto' package not installed. "
                "pip install atproto  (or: pip install predacore[bluesky])"
            )
            return

        self._client = Client()
        try:
            profile = await asyncio.to_thread(
                self._client.login, self._handle, self._app_password,
            )
            self._self_did = profile.did
        except Exception as exc:  # noqa: BLE001
            logger.error("Bluesky: login failed: %s", exc)
            return

        # The chat API lives on a sub-service proxy.
        try:
            self._chat_proxy = self._client.with_bsky_chat()
        except AttributeError:
            # Older atproto versions used a different name.
            try:
                self._chat_proxy = self._client.with_bsky_chat_proxy()
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Bluesky: chat proxy unavailable on this atproto version — "
                    "upgrade atproto: %s", exc,
                )
                return

        self._running = True
        self._poll_task = asyncio.create_task(
            self._poll_conversations(), name="bluesky-poll",
        )
        logger.info(
            "Bluesky adapter started (handle=%s, did=%s)",
            self._handle, self._self_did,
        )

    async def stop(self) -> None:
        self._running = False
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        logger.info("Bluesky adapter stopped")

    # ── Outgoing ──────────────────────────────────────────────────────

    async def send(self, message: OutgoingMessage) -> None:
        """Send a DM. ``user_id`` is the conversation id."""
        if self._chat_proxy is None:
            return
        convo_id = message.user_id
        for chunk in _chunk_message(message.text or ""):
            try:
                # atproto chat API: chat.bsky.convo.sendMessage
                await asyncio.to_thread(
                    self._chat_proxy.chat.bsky.convo.send_message,
                    {"convo_id": convo_id, "message": {"text": chunk}},
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Bluesky send failed (convo=%s): %s", convo_id, exc)
                return

    # ── Inbound ───────────────────────────────────────────────────────

    async def _poll_conversations(self) -> None:
        """Poll list_convos for unread messages, dispatch each new one."""
        while self._running:
            try:
                resp = await asyncio.to_thread(
                    self._chat_proxy.chat.bsky.convo.list_convos,
                )
                convos = getattr(resp, "convos", []) or []
                for convo in convos:
                    convo_id = getattr(convo, "id", None)
                    unread = getattr(convo, "unread_count", 0) or 0
                    if not convo_id or unread <= 0:
                        continue
                    await self._handle_convo(convo_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug("Bluesky poll error: %s", exc)
            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                return

    async def _handle_convo(self, convo_id: str) -> None:
        """Fetch unread messages in a conversation; dispatch each."""
        try:
            resp = await asyncio.to_thread(
                self._chat_proxy.chat.bsky.convo.get_messages,
                {"convo_id": convo_id, "limit": 20},
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Bluesky get_messages failed: %s", exc)
            return

        messages = getattr(resp, "messages", []) or []
        last_seen = self._last_seen_rkey.get(convo_id, "")
        # Bluesky returns newest-first; reverse so we process in order.
        for msg in reversed(messages):
            rkey = getattr(msg, "rev", "") or getattr(msg, "id", "")
            if rkey and rkey <= last_seen:
                continue
            sender_did = getattr(getattr(msg, "sender", None), "did", "")
            text = getattr(msg, "text", "") or ""
            if not text or sender_did == self._self_did:
                continue
            await self._dispatch(convo_id, sender_did, text)
            if rkey:
                self._last_seen_rkey[convo_id] = rkey

    async def _dispatch(
        self, convo_id: str, sender_did: str, text: str,
    ) -> None:
        logger.info("Bluesky DM from %s in %s: %s", sender_did, convo_id, text[:100])
        if self._message_handler is None:
            return
        outgoing = await self._message_handler(
            IncomingMessage(
                channel=self.channel_name,
                user_id=convo_id,
                text=text,
                metadata={
                    "bluesky_sender_did": sender_did,
                    "convo_id": convo_id,
                },
            )
        )
        if outgoing is not None:
            await self.send(outgoing)
