"""MessageBird (Bird) Channel Adapter — SMS via MessageBird REST API.

Same shape as the Twilio / Vonage adapters: webhook in, REST out.
MessageBird POSTs **form-encoded** data for inbound SMS (not JSON).

Setup
-----
1. Sign up at https://messagebird.com.
2. Buy a number; configure inbound webhook URL.
3. Set env (or ``~/.predacore/.env``)::

       MESSAGEBIRD_ACCESS_KEY=...
       MESSAGEBIRD_FROM=YourBrand   # alphanumeric sender ID, or E.164

4. Add ``messagebird`` to ``channels.enabled``.
"""
from __future__ import annotations

import asyncio
import logging
import os

from aiohttp import web

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)
MB_MAX_LENGTH = 1500


class MessagebirdAdapter(ChannelAdapter):
    channel_name = "messagebird"
    channel_capabilities = {
        "supports_media": False, "supports_buttons": False,
        "supports_embeds": False, "supports_markdown": False,
        "max_message_length": MB_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        super().__init__()
        self.config = config
        cfg = getattr(config.channels, "__dict__", {}).get("messagebird", {}) or {}
        self._access_key = cfg.get("access_key") or os.environ.get("MESSAGEBIRD_ACCESS_KEY", "")
        self._from = cfg.get("from") or os.environ.get("MESSAGEBIRD_FROM", "")
        self._port = int(os.environ.get("PREDACORE_MESSAGEBIRD_PORT", "") or cfg.get("webhook_port", 8769))
        self._client = None
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._bg_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        if not (self._access_key and self._from):
            logger.error("MessageBird: missing MESSAGEBIRD_ACCESS_KEY / MESSAGEBIRD_FROM")
            return
        try:
            import messagebird  # type: ignore[import-not-found]
        except ImportError:
            logger.error("MessageBird: pip install messagebird")
            return
        self._client = messagebird.Client(self._access_key)
        self._app = web.Application()
        self._app.router.add_post("/messagebird/inbound", self._handle_inbound)
        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        bind_host = os.environ.get("PREDACORE_MESSAGEBIRD_BIND_HOST", "127.0.0.1")
        site = web.TCPSite(self._runner, bind_host, self._port)
        await site.start()
        logger.info("MessageBird adapter started — port %d", self._port)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
            self._bg_tasks.clear()
        logger.info("MessageBird adapter stopped")

    async def send(self, message: OutgoingMessage) -> None:
        if self._client is None:
            return
        body = (message.text or "")[:MB_MAX_LENGTH]
        try:
            await asyncio.to_thread(
                self._client.message_create, self._from, [message.user_id], body,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("MessageBird send failed: %s", exc)

    async def _handle_inbound(self, request: web.Request) -> web.Response:
        """Inbound SMS webhook — MessageBird sends form-encoded POSTs."""
        try:
            form = await request.post()
        except Exception:  # noqa: BLE001
            return web.Response(status=400)
        sender = str(form.get("originator") or "")
        text = str(form.get("body") or "").strip()
        if not sender or not text:
            return web.Response(status=200, text="ok")
        if self._message_handler is not None:
            task = asyncio.create_task(self._dispatch(sender, text))
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
        return web.Response(status=200, text="ok")

    async def _dispatch(self, sender: str, text: str) -> None:
        if self._message_handler is None:
            return
        outgoing = await self._message_handler(
            IncomingMessage(
                channel=self.channel_name, user_id=sender, text=text,
                metadata={"phone": sender},
            )
        )
        if outgoing is not None:
            await self.send(outgoing)
