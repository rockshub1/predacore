"""Vonage SMS Channel Adapter — Vonage Messages API (Python SDK v3+).

Vonage (formerly Nexmo) is a Twilio-style telco gateway: their
Messages API covers SMS, MMS, WhatsApp, Viber, Facebook Messenger
in one unified surface. This adapter focuses on **SMS** — same pattern
as the Twilio adapter (webhook in, REST out).

Setup
-----
1. Sign up at https://dashboard.vonage.com (free trial credit).
2. Create an Application + buy a number.
3. Set env (or ``~/.predacore/.env``)::

       VONAGE_APPLICATION_ID=xxxxxx
       VONAGE_PRIVATE_KEY_PATH=/path/to/private.key   # OR raw PEM in VONAGE_PRIVATE_KEY
       VONAGE_FROM_NUMBER=12055551234   # E.164 without +

4. In Vonage dashboard, point the inbound URL at
   ``https://your-host:PORT/vonage/inbound``.

5. Add ``vonage`` to ``channels.enabled``.
"""
from __future__ import annotations

import asyncio
import logging
import os

from aiohttp import web

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)
VONAGE_MAX_LENGTH = 1500


class VonageAdapter(ChannelAdapter):
    channel_name = "vonage"
    channel_capabilities = {
        "supports_media": False, "supports_buttons": False,
        "supports_embeds": False, "supports_markdown": False,
        "max_message_length": VONAGE_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        super().__init__()
        self.config = config
        cfg = getattr(config.channels, "__dict__", {}).get("vonage", {}) or {}
        self._app_id = cfg.get("application_id") or os.environ.get("VONAGE_APPLICATION_ID", "")
        self._private_key_path = cfg.get("private_key_path") or os.environ.get("VONAGE_PRIVATE_KEY_PATH", "")
        self._private_key_inline = os.environ.get("VONAGE_PRIVATE_KEY", "")
        self._from_number = cfg.get("from_number") or os.environ.get("VONAGE_FROM_NUMBER", "")
        self._port = int(os.environ.get("PREDACORE_VONAGE_PORT", "") or cfg.get("webhook_port", 8768))
        self._client = None
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._bg_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        if not (self._app_id and self._from_number and (self._private_key_path or self._private_key_inline)):
            logger.error("Vonage: missing VONAGE_APPLICATION_ID / VONAGE_PRIVATE_KEY[_PATH] / VONAGE_FROM_NUMBER")
            return
        try:
            from vonage import Auth, Vonage  # type: ignore[import-not-found]
        except ImportError:
            logger.error("Vonage: pip install vonage")
            return

        # Vonage Auth accepts either a path or inline PEM.
        auth_kwargs = {"application_id": self._app_id}
        if self._private_key_path:
            auth_kwargs["private_key"] = self._private_key_path
        else:
            auth_kwargs["private_key"] = self._private_key_inline
        self._client = Vonage(Auth(**auth_kwargs))

        self._app = web.Application()
        self._app.router.add_post("/vonage/inbound", self._handle_inbound)
        self._app.router.add_post("/vonage/status", self._handle_status)
        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        bind_host = os.environ.get("PREDACORE_VONAGE_BIND_HOST", "127.0.0.1")
        site = web.TCPSite(self._runner, bind_host, self._port)
        await site.start()
        logger.info("Vonage adapter started — listening on %s:%d/vonage/inbound", bind_host, self._port)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
            self._bg_tasks.clear()
        logger.info("Vonage adapter stopped")

    async def send(self, message: OutgoingMessage) -> None:
        if self._client is None:
            return
        try:
            from vonage_messages import Sms  # type: ignore[import-not-found]
        except ImportError:
            logger.error("Vonage: vonage_messages submodule missing")
            return
        text = (message.text or "")[:VONAGE_MAX_LENGTH]
        try:
            # The Messages SDK is sync; bounce to a thread.
            await asyncio.to_thread(
                self._client.messages.send,
                Sms(to=message.user_id, from_=self._from_number, text=text),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Vonage send failed: %s", exc)

    async def _handle_status(self, _request: web.Request) -> web.Response:
        return web.Response(status=200, text="ok")

    async def _handle_inbound(self, request: web.Request) -> web.Response:
        """Inbound SMS webhook — Vonage POSTs JSON for Messages API."""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return web.Response(status=400, text="bad request")
        if body.get("channel") != "sms":
            return web.Response(status=200, text="ignored")
        sender = (body.get("from") or "").lstrip("+")
        text = ((body.get("message") or {}).get("content") or {}).get("text", "")
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
