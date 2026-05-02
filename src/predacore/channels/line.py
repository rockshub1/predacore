"""LINE Channel Adapter — Messaging API (line-bot-sdk-python v3).

LINE is the dominant chat app in Japan, Taiwan, Thailand. We use the
official ``line-bot-sdk`` v3 (the v1/v2 modules are still around but
v3 is the maintained surface as of 2026).

Setup
-----
1. Create a Messaging API channel at https://developers.line.biz.
2. Get the channel access token + channel secret from the LINE Developers
   console.
3. Set env (or ``~/.predacore/.env``)::

       LINE_CHANNEL_ACCESS_TOKEN=...
       LINE_CHANNEL_SECRET=...

4. In the LINE Developers console, set the webhook URL to
   ``https://your-host:PORT/line/callback`` (HTTPS required by LINE).
5. Add ``line`` to ``channels.enabled``.
"""
from __future__ import annotations

import asyncio
import logging
import os

from aiohttp import web

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)
LINE_MAX_LENGTH = 5000


class LineAdapter(ChannelAdapter):
    channel_name = "line"
    channel_capabilities = {
        "supports_media": True, "supports_buttons": True,
        "supports_embeds": False, "supports_markdown": False,
        "max_message_length": LINE_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        super().__init__()
        self.config = config
        cfg = getattr(config.channels, "__dict__", {}).get("line", {}) or {}
        self._access_token = (
            cfg.get("channel_access_token") or os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
        )
        self._secret = cfg.get("channel_secret") or os.environ.get("LINE_CHANNEL_SECRET", "")
        self._port = int(os.environ.get("PREDACORE_LINE_PORT", "") or cfg.get("webhook_port", 8770))
        self._messaging_api = None
        self._parser = None
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._bg_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        if not (self._access_token and self._secret):
            logger.error("LINE: missing LINE_CHANNEL_ACCESS_TOKEN / LINE_CHANNEL_SECRET")
            return
        try:
            from linebot.v3.messaging import Configuration, ApiClient, MessagingApi  # type: ignore[import-not-found]
            from linebot.v3.webhook import WebhookParser   # type: ignore[import-not-found]
        except ImportError:
            logger.error("LINE: pip install line-bot-sdk")
            return
        configuration = Configuration(access_token=self._access_token)
        api_client = ApiClient(configuration)
        self._messaging_api = MessagingApi(api_client)
        self._parser = WebhookParser(self._secret)
        self._app = web.Application()
        self._app.router.add_post("/line/callback", self._handle_callback)
        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        bind_host = os.environ.get("PREDACORE_LINE_BIND_HOST", "127.0.0.1")
        await web.TCPSite(self._runner, bind_host, self._port).start()
        logger.info("LINE adapter started — port %d", self._port)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
            self._bg_tasks.clear()
        logger.info("LINE adapter stopped")

    async def send(self, message: OutgoingMessage) -> None:
        """Push a text message via LINE's pushMessage. ``user_id`` is the LINE user id."""
        if self._messaging_api is None:
            return
        try:
            from linebot.v3.messaging import PushMessageRequest, TextMessage  # type: ignore[import-not-found]
        except ImportError:
            return
        body = (message.text or "")[:LINE_MAX_LENGTH]
        try:
            await asyncio.to_thread(
                self._messaging_api.push_message,
                PushMessageRequest(to=message.user_id, messages=[TextMessage(text=body)]),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("LINE send failed: %s", exc)

    async def _handle_callback(self, request: web.Request) -> web.Response:
        body = await request.text()
        signature = request.headers.get("X-Line-Signature", "")
        try:
            events = self._parser.parse(body, signature)
        except Exception as exc:  # noqa: BLE001 — InvalidSignatureError + others
            logger.warning("LINE webhook signature rejected: %s", exc)
            return web.Response(status=400)
        for event in events or []:
            etype = type(event).__name__
            if etype != "MessageEvent":
                continue
            msg = getattr(event, "message", None)
            if msg is None or type(msg).__name__ != "TextMessageContent":
                continue
            user_id = getattr(getattr(event, "source", None), "user_id", "")
            text = getattr(msg, "text", "")
            if not user_id or not text:
                continue
            if self._message_handler is not None:
                task = asyncio.create_task(self._dispatch(user_id, text))
                self._bg_tasks.add(task)
                task.add_done_callback(self._bg_tasks.discard)
        return web.Response(status=200, text="ok")

    async def _dispatch(self, user_id: str, text: str) -> None:
        if self._message_handler is None:
            return
        outgoing = await self._message_handler(
            IncomingMessage(
                channel=self.channel_name, user_id=user_id, text=text,
                metadata={"line_user_id": user_id},
            )
        )
        if outgoing is not None:
            await self.send(outgoing)
