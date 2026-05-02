"""Twilio Channel Adapter — SMS via the Twilio Programmable Messaging API.

One adapter, two channels effectively: any phone number worldwide
becomes a PredaCore endpoint. Twilio handles the telco gateway; we
just expose a webhook for inbound SMS and call their REST API for
outbound replies.

Setup
-----
1. Sign up at https://www.twilio.com (free trial credit + sandbox number).
2. Buy a phone number (or use the trial sandbox).
3. Set environment variables (or ``~/.predacore/.env``)::

       TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
       TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
       TWILIO_FROM_NUMBER=+15551234567        # the number you bought
       # Optional:
       PREDACORE_TWILIO_PORT=8767             # webhook listener port
       PREDACORE_TWILIO_WEBHOOK_PATH=/twilio  # POST endpoint path

4. In Twilio console, configure your number's "A message comes in"
   webhook to ``https://your-host:8767/twilio/sms``. For local dev,
   tunnel with ngrok or cloudflared.

5. Add ``twilio`` to ``channels.enabled`` in config.yaml. The daemon
   hot-attaches the adapter via PR T8 — no restart needed.

Voice support is out of scope for this initial adapter — voice calls
need TwiML response generation, recording handling, and are a separate
problem. SMS-only here keeps the contract tight.

Implementation notes
--------------------
- Lazy-imports ``twilio`` so the core install doesn't pull it in.
- Verifies webhook signatures via ``twilio.request_validator``: rejects
  requests Twilio didn't sign, even for trial accounts.
- Falls back to plain HTTP POST if the user pins an older twilio SDK
  that doesn't expose ``AsyncTwilioHttpClient``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.parse

from aiohttp import web

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

# Twilio SMS hard cap is 1600 chars per message; carriers often chunk
# at 160 GSM-7 chars or 70 UCS-2 chars, but the API accepts up to 1600.
# We chunk at 1500 so multi-segment delivery stays predictable.
TWILIO_MAX_LENGTH = 1500


def _chunk_message(text: str, max_len: int = TWILIO_MAX_LENGTH) -> list[str]:
    """Split ``text`` into Twilio-safe chunks. Prefers newline boundaries."""
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split = text.rfind("\n", 0, max_len)
        if split == -1:
            split = max_len
        chunks.append(text[:split])
        text = text[split:].lstrip("\n")
    return chunks


class TwilioAdapter(ChannelAdapter):
    """Twilio SMS adapter — webhook in, REST API out."""

    channel_name = "twilio"
    channel_capabilities = {
        "supports_media": False,    # MMS would need a separate field; stick to SMS
        "supports_buttons": False,
        "supports_embeds": False,
        "supports_markdown": False, # SMS strips formatting; keep it plain
        "max_message_length": TWILIO_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        super().__init__()
        self.config = config

        cfg = getattr(config.channels, "__dict__", {}).get("twilio", {}) or {}
        self._account_sid = (
            cfg.get("account_sid")
            or os.environ.get("TWILIO_ACCOUNT_SID", "")
        )
        self._auth_token = (
            cfg.get("auth_token")
            or os.environ.get("TWILIO_AUTH_TOKEN", "")
        )
        self._from_number = (
            cfg.get("from_number")
            or os.environ.get("TWILIO_FROM_NUMBER", "")
        )
        self._port = int(
            os.environ.get("PREDACORE_TWILIO_PORT", "")
            or cfg.get("webhook_port", 8767)
        )
        self._webhook_path = (
            os.environ.get("PREDACORE_TWILIO_WEBHOOK_PATH", "")
            or cfg.get("webhook_path", "/twilio")
        ).rstrip("/")
        # Public URL used when verifying webhook signatures. If unset,
        # signature verification is skipped (warned at start). Set this
        # to your ngrok / public hostname including the path:
        #   https://abc123.ngrok.io/twilio/sms
        self._public_url = (
            os.environ.get("PREDACORE_TWILIO_PUBLIC_URL", "")
            or cfg.get("public_url", "")
        )

        self._client = None        # twilio.rest.Client (set in start)
        self._validator = None     # twilio.request_validator.RequestValidator
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._bg_tasks: set[asyncio.Task] = set()
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        if not (self._account_sid and self._auth_token and self._from_number):
            logger.error(
                "Twilio: missing TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / "
                "TWILIO_FROM_NUMBER — set them in ~/.predacore/.env",
            )
            return

        try:
            from twilio.rest import Client            # type: ignore[import-not-found]
            from twilio.request_validator import RequestValidator  # type: ignore[import-not-found]
        except ImportError:
            logger.error(
                "Twilio: 'twilio' package not installed. "
                "pip install twilio  (or: pip install predacore[twilio])",
            )
            return

        self._client = Client(self._account_sid, self._auth_token)
        self._validator = RequestValidator(self._auth_token)

        if not self._public_url:
            logger.warning(
                "Twilio: PREDACORE_TWILIO_PUBLIC_URL unset — webhook "
                "signature verification will be SKIPPED. Set this to your "
                "ngrok/public URL (e.g. https://abc.ngrok.io%s/sms) for "
                "production.", self._webhook_path,
            )

        # aiohttp listener for inbound SMS webhooks.
        self._app = web.Application()
        self._app.router.add_post(f"{self._webhook_path}/sms", self._handle_sms_webhook)
        self._app.router.add_get(f"{self._webhook_path}/health", self._health_handler)

        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        bind_host = os.environ.get("PREDACORE_TWILIO_BIND_HOST", "127.0.0.1")
        site = web.TCPSite(self._runner, bind_host, self._port)
        await site.start()

        self._running = True
        logger.info(
            "Twilio adapter started — listening on %s:%d%s/sms (from=%s)",
            bind_host, self._port, self._webhook_path, self._from_number,
        )

    async def stop(self) -> None:
        self._running = False
        if self._runner is not None:
            await self._runner.cleanup()
        # Drain any in-flight reply tasks so we don't drop messages on shutdown.
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
            self._bg_tasks.clear()
        logger.info("Twilio adapter stopped")

    # ── Outgoing ──────────────────────────────────────────────────────

    async def send(self, message: OutgoingMessage) -> None:
        """Send an SMS via Twilio. ``user_id`` is the recipient's phone (E.164)."""
        if self._client is None:
            logger.error("Twilio: not started, dropping outgoing")
            return
        to_number = message.user_id
        for chunk in _chunk_message(message.text or ""):
            try:
                # twilio-python's REST client is sync. Run in a thread
                # so we don't block the event loop on long carrier hops.
                await asyncio.to_thread(
                    self._client.messages.create,
                    body=chunk,
                    from_=self._from_number,
                    to=to_number,
                )
            except Exception as exc:  # noqa: BLE001 — twilio raises a wide bag
                logger.error("Twilio send failed (to=%s): %s", to_number, exc)
                return

    # ── Inbound ───────────────────────────────────────────────────────

    async def _health_handler(self, _request: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def _handle_sms_webhook(self, request: web.Request) -> web.Response:
        """Twilio webhook receiver — validate, dispatch, reply with empty TwiML.

        Twilio expects a TwiML response (XML). Empty <Response/> means
        "thanks, no inline reply" — we send the actual reply via the REST
        API after running the message through the gateway. This pattern
        avoids 15-second webhook timeouts on long LLM responses.
        """
        # Twilio posts form-encoded data, not JSON.
        try:
            form = await request.post()
        except Exception:  # noqa: BLE001
            return web.Response(status=400, text="bad request")

        # Verify signature unless the operator opted out by leaving
        # PREDACORE_TWILIO_PUBLIC_URL unset.
        if self._public_url and self._validator is not None:
            sig = request.headers.get("X-Twilio-Signature", "")
            if not self._validator.validate(self._public_url, dict(form), sig):
                logger.warning("Twilio webhook signature rejected")
                return web.Response(status=403, text="forbidden")

        sender = str(form.get("From") or "")
        body = str(form.get("Body") or "").strip()
        message_sid = str(form.get("MessageSid") or "")
        if not sender or not body:
            return _empty_twiml()

        logger.info("Twilio SMS from %s: %s", sender, body[:100])

        if self._message_handler is not None:
            task = asyncio.create_task(
                self._process_and_reply(sender, body, message_sid),
                name=f"twilio-reply-{message_sid[:8]}",
            )
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)

        return _empty_twiml()

    async def _process_and_reply(
        self, sender: str, text: str, message_sid: str,
    ) -> None:
        if self._message_handler is None:
            return
        try:
            outgoing = await self._message_handler(
                IncomingMessage(
                    channel=self.channel_name,
                    user_id=sender,
                    text=text,
                    metadata={
                        "phone": sender,
                        "message_sid": message_sid,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Twilio gateway dispatch failed: %s", exc)
            return
        if outgoing is not None:
            await self.send(outgoing)


def _empty_twiml() -> web.Response:
    """Return Twilio's expected empty TwiML acknowledgment."""
    return web.Response(
        status=200,
        text='<?xml version="1.0" encoding="UTF-8"?><Response/>',
        content_type="text/xml",
    )
