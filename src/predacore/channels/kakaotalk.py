"""KakaoTalk Channel Adapter — Kakao Channel REST API + webhook.

Kakao Channel (formerly Plus Friend / 카카오톡 채널) is the Korean
business-channel API. Bots can receive messages via webhook callbacks
and reply via the Channel Message REST endpoint.

Setup
-----
1. Register your business at https://developers.kakao.com.
2. Create a Kakao Channel; verify the business owner.
3. Get the channel access token + channel public id (UUID).
4. Set env (or ``~/.predacore/.env``)::

       KAKAO_CHANNEL_ACCESS_TOKEN=...
       KAKAO_CHANNEL_PUBLIC_ID=...

5. In Kakao Developers console, configure the webhook URL to
   ``https://your-host:PORT/kakao/webhook``.

6. Add ``kakaotalk`` to ``channels.enabled``.

Notes
-----
Kakao gates this API behind business verification + Korean-language
docs. Treat this as a *starter* adapter — real-world fields may need
tweaks based on your specific channel's webhook contract.
"""
from __future__ import annotations

import asyncio
import logging
import os

import httpx
from aiohttp import web

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)
KAKAO_MAX_LENGTH = 1000
KAKAO_API_BASE = "https://kapi.kakao.com/v1/api/talk/channels"


class KakaotalkAdapter(ChannelAdapter):
    channel_name = "kakaotalk"
    channel_capabilities = {
        "supports_media": False, "supports_buttons": True,
        "supports_embeds": False, "supports_markdown": False,
        "max_message_length": KAKAO_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        super().__init__()
        self.config = config
        cfg = getattr(config.channels, "__dict__", {}).get("kakaotalk", {}) or {}
        self._access_token = cfg.get("access_token") or os.environ.get("KAKAO_CHANNEL_ACCESS_TOKEN", "")
        self._channel_id = cfg.get("public_id") or os.environ.get("KAKAO_CHANNEL_PUBLIC_ID", "")
        self._port = int(os.environ.get("PREDACORE_KAKAO_PORT", "") or cfg.get("webhook_port", 8774))
        self._http: httpx.AsyncClient | None = None
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._bg_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        if not (self._access_token and self._channel_id):
            logger.error("KakaoTalk: missing KAKAO_CHANNEL_ACCESS_TOKEN / KAKAO_CHANNEL_PUBLIC_ID")
            return
        self._http = httpx.AsyncClient(timeout=30)
        self._app = web.Application()
        self._app.router.add_post("/kakao/webhook", self._handle_webhook)
        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        bind_host = os.environ.get("PREDACORE_KAKAO_BIND_HOST", "127.0.0.1")
        await web.TCPSite(self._runner, bind_host, self._port).start()
        logger.info("KakaoTalk adapter started — port %d", self._port)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        if self._http is not None:
            await self._http.aclose()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
            self._bg_tasks.clear()
        logger.info("KakaoTalk adapter stopped")

    async def send(self, message: OutgoingMessage) -> None:
        if self._http is None:
            return
        body = (message.text or "")[:KAKAO_MAX_LENGTH]
        url = f"{KAKAO_API_BASE}/{self._channel_id}/messages/text"
        try:
            resp = await self._http.post(
                url,
                headers={"Authorization": f"Bearer {self._access_token}"},
                data={"recipient": message.user_id, "text": body},
            )
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.error("KakaoTalk send failed: %s", exc)

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return web.Response(status=400)
        # Kakao webhook event shape varies by channel type. Most common:
        #   {"event": "send", "user_id": "...", "content": {"text": "..."}}
        if body.get("event") not in ("send", "friend_added"):
            return web.Response(status=200, text="ok")
        user_id = body.get("user_id") or (body.get("user") or {}).get("id", "")
        text = ((body.get("content") or {}).get("text")
                or body.get("text") or "").strip()
        if not user_id or not text:
            return web.Response(status=200, text="ok")
        if self._message_handler is not None:
            task = asyncio.create_task(self._dispatch(str(user_id), text))
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
        return web.Response(status=200, text="ok")

    async def _dispatch(self, user_id: str, text: str) -> None:
        if self._message_handler is None:
            return
        outgoing = await self._message_handler(
            IncomingMessage(
                channel=self.channel_name, user_id=user_id, text=text,
                metadata={"kakao_user_id": user_id},
            )
        )
        if outgoing is not None:
            await self.send(outgoing)
