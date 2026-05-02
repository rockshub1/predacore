"""Mattermost Channel Adapter — open-source Slack alternative.

Self-hosted team chat. The adapter uses ``mattermostdriver`` (REST + WS)
to login, listen for posts in DMs / mentions, and post replies. Tested
against Mattermost server v9+; older versions may need driver pin.

Setup
-----
1. Create a bot account: System Console → Integrations → Bot Accounts.
2. Copy its access token.
3. Set env (or ``~/.predacore/.env``)::

       MATTERMOST_URL=https://your.mattermost.example.com
       MATTERMOST_ACCESS_TOKEN=...

4. Add ``mattermost`` to ``channels.enabled`` — daemon hot-attaches.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any
from urllib.parse import urlparse

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)
MATTERMOST_MAX_LENGTH = 16_383


class MattermostAdapter(ChannelAdapter):
    channel_name = "mattermost"
    channel_capabilities = {
        "supports_media": True, "supports_buttons": False,
        "supports_embeds": True, "supports_markdown": True,
        "max_message_length": MATTERMOST_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        super().__init__()
        self.config = config
        cfg = getattr(config.channels, "__dict__", {}).get("mattermost", {}) or {}
        self._url = cfg.get("url") or os.environ.get("MATTERMOST_URL", "")
        self._token = cfg.get("access_token") or os.environ.get("MATTERMOST_ACCESS_TOKEN", "")
        self._driver = None
        self._ws_task: asyncio.Task | None = None
        self._self_user_id: str | None = None
        self._running = False

    async def start(self) -> None:
        if not (self._url and self._token):
            logger.error("Mattermost: missing MATTERMOST_URL / MATTERMOST_ACCESS_TOKEN")
            return
        try:
            from mattermostdriver import Driver  # type: ignore[import-not-found]
        except ImportError:
            logger.error("Mattermost: pip install mattermostdriver")
            return
        parsed = urlparse(self._url)
        self._driver = Driver({
            "url": parsed.hostname or self._url,
            "scheme": parsed.scheme or "https",
            "port": parsed.port or (443 if parsed.scheme == "https" else 80),
            "token": self._token,
        })
        try:
            await asyncio.to_thread(self._driver.login)
            me = await asyncio.to_thread(self._driver.users.get_user, "me")
            self._self_user_id = me.get("id")
        except Exception as exc:  # noqa: BLE001
            logger.error("Mattermost login failed: %s", exc)
            return
        self._running = True
        self._ws_task = asyncio.create_task(self._websocket_loop(), name="mattermost-ws")
        logger.info("Mattermost adapter started (user=%s)", self._self_user_id)

    async def stop(self) -> None:
        self._running = False
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        logger.info("Mattermost adapter stopped")

    async def send(self, message: OutgoingMessage) -> None:
        if self._driver is None:
            return
        try:
            # user_id is the channel_id we're posting into
            await asyncio.to_thread(
                self._driver.posts.create_post,
                {"channel_id": message.user_id,
                 "message": (message.text or "")[:MATTERMOST_MAX_LENGTH]},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Mattermost send failed: %s", exc)

    async def _websocket_loop(self) -> None:
        """Run the driver's async websocket listener.

        ``init_websocket`` is an awaitable on mattermostdriver 7.x — it
        opens the WS connection and dispatches events through our handler
        until cancelled. No thread bridging needed.
        """
        try:
            await self._driver.init_websocket(self._on_event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Mattermost websocket loop failed: %s", exc)

    def _on_event(self, event: str | dict) -> None:
        """Driver fires this callback inside its websocket thread; bridge to asyncio."""
        try:
            data = json.loads(event) if isinstance(event, str) else event
        except (json.JSONDecodeError, TypeError):
            return
        if data.get("event") != "posted":
            return
        post = json.loads(data.get("data", {}).get("post", "{}"))
        if not post or post.get("user_id") == self._self_user_id:
            return
        text = post.get("message", "") or ""
        channel_id = post.get("channel_id", "")
        if not text or not channel_id:
            return
        # Schedule the dispatch on the running loop.
        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(
            self._dispatch(channel_id, post.get("user_id", ""), text), loop,
        )

    async def _dispatch(self, channel_id: str, sender: str, text: str) -> None:
        if self._message_handler is None:
            return
        outgoing = await self._message_handler(
            IncomingMessage(
                channel=self.channel_name, user_id=channel_id, text=text,
                metadata={"sender_id": sender, "channel_id": channel_id},
            )
        )
        if outgoing is not None:
            await self.send(outgoing)
