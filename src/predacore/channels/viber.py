"""Viber Channel Adapter — official ``viberbot`` SDK.

Setup
-----
1. Create a Public Account at https://partners.viber.com.
2. Get the auth token from the account dashboard.
3. Set env (or ``~/.predacore/.env``)::

       VIBER_AUTH_TOKEN=...
       VIBER_BOT_NAME=YourBotName
       VIBER_PUBLIC_URL=https://your-host:PORT/viber/incoming   # for set_webhook

4. Add ``viber`` to ``channels.enabled``.
"""
from __future__ import annotations

import asyncio
import logging
import os

from aiohttp import web

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)
VIBER_MAX_LENGTH = 7000


class ViberAdapter(ChannelAdapter):
    channel_name = "viber"
    channel_capabilities = {
        "supports_media": True, "supports_buttons": True,
        "supports_embeds": False, "supports_markdown": False,
        "max_message_length": VIBER_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        super().__init__()
        self.config = config
        cfg = getattr(config.channels, "__dict__", {}).get("viber", {}) or {}
        self._auth_token = cfg.get("auth_token") or os.environ.get("VIBER_AUTH_TOKEN", "")
        self._bot_name = cfg.get("bot_name") or os.environ.get("VIBER_BOT_NAME", "PredaCore")
        self._public_url = cfg.get("public_url") or os.environ.get("VIBER_PUBLIC_URL", "")
        self._port = int(os.environ.get("PREDACORE_VIBER_PORT", "") or cfg.get("webhook_port", 8771))
        self._viber = None
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._bg_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        if not self._auth_token:
            logger.error("Viber: missing VIBER_AUTH_TOKEN")
            return
        try:
            from viberbot import Api    # type: ignore[import-not-found]
            from viberbot.api.bot_configuration import BotConfiguration  # type: ignore[import-not-found]
        except ImportError:
            logger.error("Viber: pip install viberbot")
            return
        self._viber = Api(BotConfiguration(name=self._bot_name, avatar="", auth_token=self._auth_token))
        self._app = web.Application()
        self._app.router.add_post("/viber/incoming", self._handle_incoming)
        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        bind_host = os.environ.get("PREDACORE_VIBER_BIND_HOST", "127.0.0.1")
        await web.TCPSite(self._runner, bind_host, self._port).start()
        # Register webhook with Viber
        if self._public_url:
            try:
                await asyncio.to_thread(self._viber.set_webhook, self._public_url)
                logger.info("Viber webhook registered: %s", self._public_url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Viber set_webhook failed: %s", exc)
        else:
            logger.warning("Viber: VIBER_PUBLIC_URL unset — registering the webhook is up to you")
        logger.info("Viber adapter started — port %d", self._port)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
            self._bg_tasks.clear()
        logger.info("Viber adapter stopped")

    async def send(self, message: OutgoingMessage) -> None:
        if self._viber is None:
            return
        try:
            from viberbot.api.messages.text_message import TextMessage  # type: ignore[import-not-found]
        except ImportError:
            return
        body = (message.text or "")[:VIBER_MAX_LENGTH]
        try:
            await asyncio.to_thread(
                self._viber.send_messages, message.user_id, [TextMessage(text=body)],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Viber send failed: %s", exc)

    async def _handle_incoming(self, request: web.Request) -> web.Response:
        body_bytes = await request.read()
        signature = request.headers.get("X-Viber-Content-Signature", "")
        try:
            valid = await asyncio.to_thread(self._viber.verify_signature, body_bytes, signature)
        except Exception:  # noqa: BLE001
            valid = False
        if not valid:
            return web.Response(status=403, text="forbidden")
        try:
            request_obj = await asyncio.to_thread(self._viber.parse_request, body_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Viber parse_request failed: %s", exc)
            return web.Response(status=400, text="bad request")
        # Only handle MessageRequest events
        if type(request_obj).__name__ != "ViberMessageRequest":
            return web.Response(status=200, text="ok")
        msg = getattr(request_obj, "message", None)
        sender = getattr(request_obj, "sender", None)
        if msg is None or sender is None:
            return web.Response(status=200, text="ok")
        text = getattr(msg, "text", "") or ""
        sender_id = getattr(sender, "id", "")
        if not text or not sender_id:
            return web.Response(status=200, text="ok")
        if self._message_handler is not None:
            task = asyncio.create_task(self._dispatch(sender_id, text))
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
        return web.Response(status=200, text="ok")

    async def _dispatch(self, sender_id: str, text: str) -> None:
        if self._message_handler is None:
            return
        outgoing = await self._message_handler(
            IncomingMessage(
                channel=self.channel_name, user_id=sender_id, text=text,
                metadata={"viber_sender_id": sender_id},
            )
        )
        if outgoing is not None:
            await self.send(outgoing)
