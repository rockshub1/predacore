"""Zalo Official Account Channel Adapter — Vietnam's dominant chat app.

Zalo has both an Official Account (OA) API (legitimate, requires Zalo
Business approval + 1-year access tokens) and an unofficial personal-
account approach (zlapi). This adapter targets the OA API: webhook
in, REST out.

Setup
-----
1. Register an Official Account at https://oa.zalo.me (requires
   Vietnamese phone or business documents).
2. Get the OA Access Token (1-year expiry; refresh via OAuth flow).
3. Set env (or ``~/.predacore/.env``)::

       ZALO_OA_ACCESS_TOKEN=...

4. In Zalo OA dashboard, set the webhook to
   ``https://your-host:PORT/zalo/webhook``.

5. Add ``zalo`` to ``channels.enabled``.
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
ZALO_MAX_LENGTH = 2000
ZALO_API_BASE = "https://openapi.zalo.me/v3.0/oa"


class ZaloAdapter(ChannelAdapter):
    channel_name = "zalo"
    channel_capabilities = {
        "supports_media": True, "supports_buttons": False,
        "supports_embeds": False, "supports_markdown": False,
        "max_message_length": ZALO_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        super().__init__()
        self.config = config
        cfg = getattr(config.channels, "__dict__", {}).get("zalo", {}) or {}
        self._access_token = cfg.get("access_token") or os.environ.get("ZALO_OA_ACCESS_TOKEN", "")
        self._port = int(os.environ.get("PREDACORE_ZALO_PORT", "") or cfg.get("webhook_port", 8773))
        self._http: httpx.AsyncClient | None = None
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._bg_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        if not self._access_token:
            logger.error("Zalo: missing ZALO_OA_ACCESS_TOKEN")
            return
        self._http = httpx.AsyncClient(timeout=30)
        self._app = web.Application()
        self._app.router.add_post("/zalo/webhook", self._handle_webhook)
        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        bind_host = os.environ.get("PREDACORE_ZALO_BIND_HOST", "127.0.0.1")
        await web.TCPSite(self._runner, bind_host, self._port).start()
        logger.info("Zalo adapter started — port %d", self._port)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        if self._http is not None:
            await self._http.aclose()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
            self._bg_tasks.clear()
        logger.info("Zalo adapter stopped")

    async def send(self, message: OutgoingMessage) -> None:
        """``user_id`` is the Zalo user_id (numeric string)."""
        if self._http is None:
            return
        body = (message.text or "")[:ZALO_MAX_LENGTH]
        # Zalo OA send-message v3 endpoint
        url = f"{ZALO_API_BASE}/message"
        payload = {
            "recipient": {"user_id": message.user_id},
            "message": {"text": body},
        }
        try:
            resp = await self._http.post(
                url, json=payload,
                headers={"access_token": self._access_token},
            )
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.error("Zalo send failed: %s", exc)

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return web.Response(status=400)
        # Zalo events: ``event_name`` in {user_send_text, user_send_image, ...}
        event = body.get("event_name", "")
        if event != "user_send_text":
            return web.Response(status=200, text="ok")
        sender_id = (body.get("sender") or {}).get("id", "")
        text = (body.get("message") or {}).get("text", "")
        if not sender_id or not text:
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
                metadata={"zalo_user_id": sender_id},
            )
        )
        if outgoing is not None:
            await self.send(outgoing)
