"""Vonage SMS Channel Adapter — Vonage Messages API (Python SDK v3+).

Vonage (formerly Nexmo) is a Twilio-style telco gateway: their
Messages API covers SMS, MMS, WhatsApp, Viber, Facebook Messenger
in one unified surface. This adapter focuses on **SMS** — same pattern
as the Twilio adapter (webhook in, REST out).

Setup
-----
1. Sign up at https://dashboard.vonage.com (free trial credit).
2. Create an Application + buy a number.
3. Copy the **signature secret** from your Vonage account settings (it's
   a separate token from the application private key).
4. Set env (or ``~/.predacore/.env``)::

       VONAGE_APPLICATION_ID=xxxxxx
       VONAGE_PRIVATE_KEY_PATH=/path/to/private.key   # OR raw PEM in VONAGE_PRIVATE_KEY
       VONAGE_FROM_NUMBER=12055551234   # E.164 without +
       VONAGE_SIGNATURE_SECRET=...      # HS256 secret for inbound JWT verification

5. In Vonage dashboard, point the inbound URL at
   ``https://your-host:PORT/vonage/inbound``.

6. Add ``vonage`` to ``channels.enabled``.

Inbound webhook signature verification
--------------------------------------
Vonage Messages API signs every inbound webhook with HS256 JWT in the
``Authorization: Bearer <jwt>`` header. We verify:

  * HS256 signature against ``VONAGE_SIGNATURE_SECRET``
  * ``nbf`` / ``exp`` claims (5 min clock-skew tolerance)
  * ``payload_hash`` claim equals ``sha256(body)`` hex digest

If ``VONAGE_SIGNATURE_SECRET`` is unset, the adapter refuses to start
unless ``PREDACORE_VONAGE_INSECURE=1`` is set explicitly. This mirrors
the Twilio default-deny posture.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time

from aiohttp import web

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)
VONAGE_MAX_LENGTH = 1500
_JWT_CLOCK_SKEW_SECONDS = 300


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


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
        self._signature_secret = cfg.get("signature_secret") or os.environ.get("VONAGE_SIGNATURE_SECRET", "")
        self._port = int(os.environ.get("PREDACORE_VONAGE_PORT", "") or cfg.get("webhook_port", 8768))
        self._client = None
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._bg_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        if not (self._app_id and self._from_number and (self._private_key_path or self._private_key_inline)):
            logger.error("Vonage: missing VONAGE_APPLICATION_ID / VONAGE_PRIVATE_KEY[_PATH] / VONAGE_FROM_NUMBER")
            return
        insecure = os.environ.get("PREDACORE_VONAGE_INSECURE", "").strip().lower() in {"1", "true", "yes", "on"}
        if not self._signature_secret and not insecure:
            logger.critical(
                "Vonage: refusing to start — VONAGE_SIGNATURE_SECRET is unset. "
                "Inbound webhooks would be unauthenticated and any caller able to "
                "POST to the listener could spoof user messages and trigger LLM "
                "tool calls. Copy the signature secret from your Vonage account "
                "settings and set VONAGE_SIGNATURE_SECRET, or set "
                "PREDACORE_VONAGE_INSECURE=1 to opt out of verification."
            )
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
        if self._signature_secret:
            logger.info(
                "Vonage adapter started — listening on %s:%d/vonage/inbound (HS256 JWT verification active)",
                bind_host, self._port,
            )
        else:
            logger.warning(
                "Vonage adapter started — listening on %s:%d/vonage/inbound (PREDACORE_VONAGE_INSECURE=1; "
                "signature verification disabled, accepting all inbound POSTs)",
                bind_host, self._port,
            )

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

    def _verify_signature(self, body_bytes: bytes, auth_header: str) -> bool:
        """Verify Vonage Messages API ``Authorization: Bearer <jwt>`` HS256 JWT.

        Returns True iff:
        - Header has ``Bearer`` prefix and a 3-part JWT
        - HS256 signature matches signature_secret (timing-safe compare)
        - ``nbf`` ≤ now ≤ ``exp`` (5 min clock-skew tolerance)
        - ``payload_hash`` (hex) equals ``sha256(body_bytes)``
        """
        if not self._signature_secret:
            return False
        if not auth_header:
            return False
        token = auth_header.strip()
        if not token.lower().startswith("bearer "):
            return False
        token = token[7:].strip()
        parts = token.split(".")
        if len(parts) != 3:
            return False
        header_b64, payload_b64, sig_b64 = parts
        try:
            header = json.loads(_b64url_decode(header_b64))
            payload = json.loads(_b64url_decode(payload_b64))
            received_sig = _b64url_decode(sig_b64)
        except (ValueError, json.JSONDecodeError, base64.binascii.Error):
            return False
        if header.get("alg") != "HS256" or header.get("typ", "JWT") != "JWT":
            return False
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        expected_sig = hmac.new(
            self._signature_secret.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        if not hmac.compare_digest(expected_sig, received_sig):
            return False
        now = int(time.time())
        nbf = payload.get("nbf")
        exp = payload.get("exp")
        if isinstance(nbf, (int, float)) and now + _JWT_CLOCK_SKEW_SECONDS < int(nbf):
            return False
        if isinstance(exp, (int, float)) and now - _JWT_CLOCK_SKEW_SECONDS > int(exp):
            return False
        body_hash = hashlib.sha256(body_bytes).hexdigest()
        claim_hash = payload.get("payload_hash")
        if not isinstance(claim_hash, str):
            return False
        if not hmac.compare_digest(body_hash.lower(), claim_hash.lower()):
            return False
        return True

    async def _handle_status(self, _request: web.Request) -> web.Response:
        return web.Response(status=200, text="ok")

    async def _handle_inbound(self, request: web.Request) -> web.Response:
        """Inbound SMS webhook — Vonage POSTs JSON for Messages API."""
        body_bytes = await request.read()
        if self._signature_secret:
            auth_header = request.headers.get("Authorization", "")
            if not self._verify_signature(body_bytes, auth_header):
                logger.warning("Vonage webhook signature verification failed")
                return web.Response(status=401, text="invalid signature")
        try:
            body = json.loads(body_bytes.decode("utf-8", errors="replace"))
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
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
