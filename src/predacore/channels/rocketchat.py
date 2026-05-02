"""Rocket.Chat Channel Adapter — uses ``rocketchat-async`` (Realtime API).

Setup
-----
1. Create a bot user on your Rocket.Chat server (or use an existing one).
2. Set env (or ``~/.predacore/.env``)::

       ROCKETCHAT_URL=https://chat.example.com
       ROCKETCHAT_USERNAME=botuser
       ROCKETCHAT_PASSWORD=...

3. Add ``rocketchat`` to ``channels.enabled`` — daemon hot-attaches.

Implementation: ``rocketchat-async`` exposes a Realtime API client where
``RocketChat()`` connects, ``login()`` authenticates, then
``subscribe_to_channel_messages`` registers a callback for new messages.
"""
from __future__ import annotations

import asyncio
import logging
import os

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)
RC_MAX_LENGTH = 5000


class RocketchatAdapter(ChannelAdapter):
    channel_name = "rocketchat"
    channel_capabilities = {
        "supports_media": True, "supports_buttons": False,
        "supports_embeds": False, "supports_markdown": True,
        "max_message_length": RC_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        super().__init__()
        self.config = config
        cfg = getattr(config.channels, "__dict__", {}).get("rocketchat", {}) or {}
        self._url = cfg.get("url") or os.environ.get("ROCKETCHAT_URL", "")
        self._username = cfg.get("username") or os.environ.get("ROCKETCHAT_USERNAME", "")
        self._password = cfg.get("password") or os.environ.get("ROCKETCHAT_PASSWORD", "")
        self._client = None
        self._self_user_id: str | None = None
        self._main_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if not (self._url and self._username and self._password):
            logger.error("Rocket.Chat: missing ROCKETCHAT_URL/USERNAME/PASSWORD")
            return
        try:
            from rocketchat_async import RocketChat  # type: ignore[import-not-found]
        except ImportError:
            logger.error("Rocket.Chat: pip install rocketchat-async")
            return
        self._client = RocketChat()
        try:
            await self._client.start(self._url, self._username, self._password)
            self._self_user_id = getattr(self._client, "user_id", None)
        except Exception as exc:  # noqa: BLE001
            logger.error("Rocket.Chat login failed: %s", exc)
            return
        self._running = True
        # Subscribe to all channels the bot is a member of.
        try:
            channels = await self._client.get_channels()
            for ch in channels:
                await self._client.subscribe_to_channel_messages(ch, self._on_message)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Rocket.Chat channel subscribe partial: %s", exc)
        # Keep the connection alive — rocketchat-async runs an event loop
        # internally; we just don't return from start until stop is called.
        self._main_task = asyncio.create_task(self._client.run_forever(), name="rocketchat-loop")
        logger.info("Rocket.Chat adapter started (user=%s)", self._self_user_id)

    async def stop(self) -> None:
        self._running = False
        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
            try:
                await self._main_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:  # noqa: BLE001
                pass
        logger.info("Rocket.Chat adapter stopped")

    async def send(self, message: OutgoingMessage) -> None:
        if self._client is None:
            return
        try:
            # user_id carries the channel/room id we received from
            await self._client.send_message(
                (message.text or "")[:RC_MAX_LENGTH], message.user_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Rocket.Chat send failed: %s", exc)

    async def _on_message(self, channel_id, sender_id, text, _msg_id, _qualifier) -> None:
        """rocketchat-async passes (channel_id, sender_id, text, msg_id, qualifier)."""
        if sender_id == self._self_user_id or not text:
            return
        if self._message_handler is None:
            return
        outgoing = await self._message_handler(
            IncomingMessage(
                channel=self.channel_name, user_id=channel_id, text=text,
                metadata={"sender_id": sender_id, "channel_id": channel_id},
            )
        )
        if outgoing is not None:
            await self.send(outgoing)
