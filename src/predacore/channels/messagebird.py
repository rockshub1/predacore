"""MessageBird (Bird) Channel Adapter — SMS via MessageBird REST API.

Same shape as the Twilio / Vonage adapters: webhook in, REST out.
MessageBird POSTs **form-encoded** data for inbound SMS (not JSON).

Setup
-----
1. Sign up at https://messagebird.com.
2. Buy a number; configure inbound webhook URL.
3. Generate a signing key in the MessageBird dashboard
   (Developers → Webhooks → Signing key).
4. Set env (or ``~/.predacore/.env``)::

       MESSAGEBIRD_ACCESS_KEY=...
       MESSAGEBIRD_FROM=YourBrand     # alphanumeric sender ID, or E.164
       MESSAGEBIRD_SIGNING_KEY=...    # base64 signing key from dashboard

5. Add ``messagebird`` to ``channels.enabled``.

Inbound webhook signature verification
--------------------------------------
MessageBird signs every inbound webhook with HS256 JWT in the
``MessageBird-Signature-JWT`` header. We verify:

  * HS256 signature against ``MESSAGEBIRD_SIGNING_KEY``
  * ``nbf`` / ``exp`` claims (with 5 min clock-skew tolerance)
  * ``payload_hash`` claim equals ``sha256(body)`` hex digest

The legacy ``MessageBird-Signature`` HMAC scheme (v1) is deprecated by
MessageBird and not supported here.

If ``MESSAGEBIRD_SIGNING_KEY`` is unset, the adapter refuses to start
unless ``PREDACORE_MESSAGEBIRD_INSECURE=1`` is set explicitly. This
mirrors the Twilio default-deny posture.
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
MB_MAX_LENGTH = 1500
_JWT_CLOCK_SKEW_SECONDS = 300


def _b64url_decode(s: str) -> bytes:
    """Base64-url-decode with padding fix."""
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


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
        self._signing_key = cfg.get("signing_key") or os.environ.get("MESSAGEBIRD_SIGNING_KEY", "")
        self._port = int(os.environ.get("PREDACORE_MESSAGEBIRD_PORT", "") or cfg.get("webhook_port", 8769))
        self._client = None
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._bg_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        if not (self._access_key and self._from):
            logger.error("MessageBird: missing MESSAGEBIRD_ACCESS_KEY / MESSAGEBIRD_FROM")
            return
        insecure = os.environ.get("PREDACORE_MESSAGEBIRD_INSECURE", "").strip().lower() in {"1", "true", "yes", "on"}
        if not self._signing_key and not insecure:
            logger.critical(
                "MessageBird: refusing to start — MESSAGEBIRD_SIGNING_KEY is unset. "
                "Inbound webhooks would be unauthenticated and any caller able to "
                "POST to the listener could spoof user messages and trigger LLM "
                "tool calls. Set MESSAGEBIRD_SIGNING_KEY (from the MessageBird "
                "dashboard) to enable JWT signature verification, or set "
                "PREDACORE_MESSAGEBIRD_INSECURE=1 to opt out of verification."
            )
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
        if self._signing_key:
            logger.info(
                "MessageBird adapter started — port %d (HS256 JWT verification active)",
                self._port,
            )
        else:
            logger.warning(
                "MessageBird adapter started — port %d (PREDACORE_MESSAGEBIRD_INSECURE=1; "
                "signature verification disabled, accepting all inbound POSTs)",
                self._port,
            )

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

    def _verify_signature(self, body_bytes: bytes, jwt_header: str) -> bool:
        """Verify ``MessageBird-Signature-JWT`` against signing key + payload hash.

        Returns True iff:
        - JWT is well-formed and uses HS256
        - HMAC signature matches (timing-safe compare)
        - ``nbf`` ≤ now ≤ ``exp`` (5 min clock-skew tolerance)
        - ``payload_hash`` (hex) equals ``sha256(body_bytes)``
        """
        if not self._signing_key:
            return False
        if not jwt_header:
            return False
        # Strip optional "Bearer " prefix some intermediaries add.
        token = jwt_header.strip()
        if token.lower().startswith("bearer "):
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
            self._signing_key.encode("utf-8"), signing_input, hashlib.sha256
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

    async def _handle_inbound(self, request: web.Request) -> web.Response:
        """Inbound SMS webhook — MessageBird sends form-encoded POSTs."""
        body_bytes = await request.read()
        if self._signing_key:
            sig_header = request.headers.get("MessageBird-Signature-JWT", "")
            if not self._verify_signature(body_bytes, sig_header):
                logger.warning("MessageBird webhook signature verification failed")
                return web.Response(status=401, text="invalid signature")
        try:
            # Re-parse form from already-read bytes; aiohttp's request.post()
            # would re-read from the wire (which is now drained), so we parse
            # the captured bytes manually.
            from urllib.parse import parse_qs
            decoded = body_bytes.decode("utf-8", errors="replace")
            form = {k: v[0] if v else "" for k, v in parse_qs(decoded).items()}
        except (ValueError, UnicodeDecodeError):
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
