"""Google Chat Channel Adapter — webhook-based, no SDK required.

Google Chat lets you build "outgoing webhook" bots: Chat POSTs JSON to
your URL on every event, you respond with JSON. The simplest path —
no Google API client library needed — and works for any space the bot
is added to.

Setup
-----
1. In Google Workspace admin: create a new Chat app
   (https://console.cloud.google.com → APIs & Services → Chat API).
2. Configure connection settings → "App URL" → set to your public
   webhook (HTTPS required by Google).
3. Set env (or ``~/.predacore/.env``)::

       GOOGLE_CHAT_VERIFICATION_TOKEN=...   # from the app's settings panel

4. Add ``google_chat`` to ``channels.enabled``.

Outbound replies come synchronously in the webhook response. For
async / multi-message replies you'd use the Chat REST API + a service
account; that's a future extension.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

from aiohttp import web

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)
GC_MAX_LENGTH = 32_000


class GoogleChatAdapter(ChannelAdapter):
    channel_name = "google_chat"
    channel_capabilities = {
        "supports_media": False, "supports_buttons": True,  # via cards
        "supports_embeds": True, "supports_markdown": True,
        "max_message_length": GC_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        super().__init__()
        self.config = config
        cfg = getattr(config.channels, "__dict__", {}).get("google_chat", {}) or {}
        self._verification_token = (
            cfg.get("verification_token")
            or os.environ.get("GOOGLE_CHAT_VERIFICATION_TOKEN", "")
        )
        self._port = int(os.environ.get("PREDACORE_GCHAT_PORT", "") or cfg.get("webhook_port", 8772))
        # Map (space.name) → pending response future. Google Chat expects
        # the reply in the *webhook response body*, so we need to bridge
        # async LLM calls back to the open HTTP request.
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        self._app = web.Application()
        self._app.router.add_post("/google_chat/webhook", self._handle_webhook)
        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        bind_host = os.environ.get("PREDACORE_GCHAT_BIND_HOST", "127.0.0.1")
        await web.TCPSite(self._runner, bind_host, self._port).start()
        logger.info("Google Chat adapter started — port %d", self._port)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        logger.info("Google Chat adapter stopped")

    async def send(self, message: OutgoingMessage) -> None:
        """Out-of-band send via the Chat REST API isn't wired in this adapter
        — replies happen inline in the webhook response. Logging only."""
        logger.debug("Google Chat: send() called out-of-band — not wired (use webhook reply)")

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return web.Response(status=400, text="bad request")

        # Verify the bearer-style token Google Chat optionally sends.
        if self._verification_token:
            tok = body.get("token", "")
            if tok != self._verification_token:
                logger.warning("Google Chat: token mismatch — rejecting")
                return web.Response(status=403, text="forbidden")

        evt_type = body.get("type", "")
        if evt_type != "MESSAGE":
            # ADDED_TO_SPACE / REMOVED_FROM_SPACE etc — friendly ack
            return web.json_response({"text": "Hello, I'm online."})

        msg = body.get("message", {}) or {}
        text = msg.get("text", "") or ""
        sender = (msg.get("sender") or {}).get("name", "")  # "users/123"
        space_name = (msg.get("space") or {}).get("name", "")  # "spaces/AAAA"

        # Strip leading @bot mention (Chat prepends it for spaces)
        annotations = msg.get("annotations") or []
        for ann in annotations:
            if ann.get("type") == "USER_MENTION":
                start = ann.get("startIndex", 0)
                length = ann.get("length", 0)
                text = (text[:start] + text[start + length:]).strip()
                break

        if not text or self._message_handler is None:
            return web.json_response({"text": "(no content)"})

        # Run the message through the gateway and reply inline.
        try:
            outgoing = await asyncio.wait_for(
                self._message_handler(
                    IncomingMessage(
                        channel=self.channel_name,
                        user_id=space_name or sender,
                        text=text,
                        metadata={
                            "gchat_sender": sender,
                            "gchat_space": space_name,
                        },
                    )
                ),
                timeout=25,  # Chat's webhook timeout is 30s; leave headroom
            )
        except asyncio.TimeoutError:
            return web.json_response({"text": "Still thinking — try again in a moment."})
        except Exception as exc:  # noqa: BLE001
            logger.error("Google Chat dispatch failed: %s", exc)
            return web.json_response({"text": "Internal error."})

        reply_text = (outgoing.text if outgoing else "")[:GC_MAX_LENGTH]
        return web.json_response({"text": reply_text or "(empty)"})
