"""
WhatsApp Channel Adapter — connects PredaCore via WhatsApp Business Cloud API.

Uses Meta's WhatsApp Business Platform (Cloud API) with webhook
receiver for incoming messages and REST API for outgoing messages.

Setup:
  1. Create a Meta Business App at https://developers.facebook.com
  2. Set up WhatsApp Business API
  3. Add credentials to config.yaml:
       channels:
         whatsapp:
           phone_number_id: "YOUR_PHONE_NUMBER_ID"
           access_token: "YOUR_ACCESS_TOKEN"
           verify_token: "YOUR_WEBHOOK_VERIFY_TOKEN"
  4. Enable whatsapp in channels.enabled list
  5. Configure webhook URL to point to your server

Features:
  - Webhook receiver for incoming messages
  - REST API for outgoing messages
  - Template message support for notifications
  - Media message support (images, documents)
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
from typing import Optional

import httpx
from aiohttp import web

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

WA_API_URL = "https://graph.facebook.com/v18.0"
WA_MAX_LENGTH = 4096


def _chunk_message(text: str, max_len: int = WA_MAX_LENGTH) -> list[str]:
    """Split message into WhatsApp-safe chunks."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


class WhatsAppAdapter(ChannelAdapter):
    """
    WhatsApp Business Cloud API adapter.

    Receives messages via webhook, sends responses via REST API.
    """

    channel_name = "whatsapp"
    channel_capabilities = {
        "supports_media": True,
        "supports_buttons": True,
        "supports_markdown": False,  # WhatsApp uses limited formatting
        "max_message_length": WA_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        self.config = config
        self._message_handler = None
        wa_config = config.channels.__dict__.get("whatsapp", {})
        self._phone_number_id = wa_config.get("phone_number_id", "")
        self._access_token = wa_config.get("access_token", "")
        self._verify_token = wa_config.get("verify_token", "predacore_webhook_verify")
        self._app_secret = wa_config.get(
            "app_secret", ""
        )  # For webhook signature verification
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._bg_tasks: set = set()  # Track fire-and-forget tasks
        self._port = wa_config.get("webhook_port", 8766)

    async def start(self) -> None:
        """Start the webhook server for incoming WhatsApp messages."""
        if not self._access_token or not self._phone_number_id:
            logger.error(
                "WhatsApp not configured. Set channels.whatsapp.phone_number_id "
                "and channels.whatsapp.access_token"
            )
            return

        self._app = web.Application()
        self._app.router.add_get("/webhook", self._verify_webhook)
        self._app.router.add_post("/webhook", self._handle_webhook)

        self._http_client = httpx.AsyncClient(timeout=30.0)

        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        bind_host = os.environ.get("PREDACORE_WHATSAPP_BIND_HOST", "127.0.0.1")
        site = web.TCPSite(self._runner, bind_host, self._port)
        await site.start()

        logger.info("✅ WhatsApp adapter started (webhook on port %d)", self._port)

    async def stop(self) -> None:
        """Stop the webhook server."""
        if self._http_client:
            await self._http_client.aclose()
        if self._runner:
            await self._runner.cleanup()
            logger.info("WhatsApp adapter stopped")

    async def send(self, message: OutgoingMessage) -> None:
        """Send a text message via WhatsApp Business API."""
        user_id = message.user_id
        chunks = _chunk_message(message.text)
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        client = self._http_client

        for chunk in chunks:
            payload = {
                "messaging_product": "whatsapp",
                "to": user_id,
                "type": "text",
                "text": {"body": chunk},
            }

            try:
                resp = await client.post(
                    f"{WA_API_URL}/{self._phone_number_id}/messages",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._access_token}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
            except (OSError, ConnectionError, ValueError) as e:
                logger.error("Failed to send WhatsApp message: %s", e)
            except Exception as e:
                # Catch httpx.HTTPStatusError and other unexpected errors
                logger.error("WhatsApp send failed (HTTP or unexpected): %s", e)

    async def _verify_webhook(self, request: web.Request) -> web.Response:
        """Handle Facebook webhook verification (GET request)."""
        mode = request.query.get("hub.mode")
        token = request.query.get("hub.verify_token")
        challenge = request.query.get("hub.challenge")

        if mode == "subscribe" and token == self._verify_token:
            logger.info("WhatsApp webhook verified")
            return web.Response(text=challenge)

        return web.Response(status=403, text="Forbidden")

    def _verify_signature(self, body_bytes: bytes, signature_header: str) -> bool:
        """Verify the webhook payload signature using app secret."""
        if not self._app_secret:
            logger.critical("WhatsApp app_secret not configured — rejecting webhook")
            return False
        if not signature_header or not signature_header.startswith("sha256="):
            return False
        expected = hmac.new(
            self._app_secret.encode("utf-8"),
            body_bytes,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(f"sha256={expected}", signature_header)

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        """Handle incoming WhatsApp messages (POST request)."""
        body_bytes = await request.read()

        # Verify webhook signature if app_secret is configured
        sig_header = request.headers.get("X-Hub-Signature-256", "")
        if not self._verify_signature(body_bytes, sig_header):
            logger.warning("WhatsApp webhook signature verification failed")
            return web.Response(status=403, text="Forbidden")

        try:
            body = json.loads(body_bytes)
        except (ValueError, json.JSONDecodeError):
            return web.Response(status=400, text="Bad Request")

        # Parse the webhook payload
        try:
            for entry in body.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    messages = value.get("messages", [])

                    for msg in messages:
                        if msg.get("type") == "text":
                            sender = msg["from"]
                            text = msg["text"]["body"]

                            logger.info(
                                "WhatsApp message from %s: %s", sender, text[:100]
                            )

                            # Route through gateway handler
                            if self._message_handler:
                                task = asyncio.create_task(
                                    self._process_and_reply(sender, text)
                                )
                                self._bg_tasks.add(task)
                                task.add_done_callback(self._bg_tasks.discard)

        except Exception as e:
            logger.error("Error processing WhatsApp webhook: %s", e, exc_info=True)

        return web.Response(status=200, text="OK")

    async def _process_and_reply(self, sender: str, text: str) -> None:
        """Process a message and send the response."""
        if self._message_handler:
            outgoing = await self._message_handler(
                IncomingMessage(
                    channel=self.channel_name,
                    user_id=sender,
                    text=text,
                    metadata={"phone": sender},
                )
            )

            if outgoing:
                await self.send(outgoing)
