"""Google Chat Channel Adapter — webhook-based, no SDK required.

Google Chat lets you build "outgoing webhook" bots: Chat POSTs JSON to
your URL on every event, you respond with JSON. The simplest path —
no Google API client library needed — and works for any space the bot
is added to.

Setup
-----
1. In Google Cloud Console: create a Chat app
   (https://console.cloud.google.com → APIs & Services → Chat API).
2. Configure connection settings → "App URL" → set to your public
   webhook (HTTPS required by Google).
3. Set env (or ``~/.predacore/.env``)::

       # Modern (preferred): OIDC bearer JWT verification
       GOOGLE_CHAT_PROJECT_NUMBER=123456789012   # numeric project number — JWT audience

       # Legacy (deprecated by Google): body-supplied verification token
       GOOGLE_CHAT_VERIFICATION_TOKEN=...        # from the app's settings panel

4. Add ``google_chat`` to ``channels.enabled``.

Outbound replies come synchronously in the webhook response. For
async / multi-message replies you'd use the Chat REST API + a service
account; that's a future extension.

Inbound webhook authentication
------------------------------
Google Chat signs every inbound webhook with an OIDC bearer JWT in the
``Authorization: Bearer <jwt>`` header. We verify:

  * RS256 signature against the Google service-account public certs at
    ``https://www.googleapis.com/service_accounts/v1/metadata/x509/chat@system.gserviceaccount.com``
    (cached 1h, re-fetched on key rotation).
  * ``iss == "chat@system.gserviceaccount.com"``
  * ``aud == GOOGLE_CHAT_PROJECT_NUMBER``
  * ``nbf`` / ``exp`` (5 min clock-skew tolerance)

Legacy body ``token`` field is still accepted when
``GOOGLE_CHAT_VERIFICATION_TOKEN`` is set, but only as a fallback for
operators who haven't migrated. Prefer the JWT path.

If neither ``GOOGLE_CHAT_PROJECT_NUMBER`` nor
``GOOGLE_CHAT_VERIFICATION_TOKEN`` is set, the adapter refuses to start
unless ``PREDACORE_GCHAT_INSECURE=1`` is set explicitly.
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import os
import time

from aiohttp import web

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)
GC_MAX_LENGTH = 32_000
_GOOGLE_CHAT_ISSUER = "chat@system.gserviceaccount.com"
_GOOGLE_CERTS_URL = (
    "https://www.googleapis.com/service_accounts/v1/metadata/x509/"
    "chat@system.gserviceaccount.com"
)
_JWT_CLOCK_SKEW_SECONDS = 300
_JWKS_CACHE_TTL_SECONDS = 3600


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


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
        self._project_number = (
            cfg.get("project_number")
            or os.environ.get("GOOGLE_CHAT_PROJECT_NUMBER", "")
        )
        self._port = int(os.environ.get("PREDACORE_GCHAT_PORT", "") or cfg.get("webhook_port", 8772))
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        # JWKS cache: dict[kid, public_key] + expiry timestamp
        self._certs: dict[str, object] = {}
        self._certs_expires_at: float = 0.0
        self._certs_lock = asyncio.Lock()
        self._http = None  # lazy httpx.AsyncClient

    async def start(self) -> None:
        insecure = os.environ.get("PREDACORE_GCHAT_INSECURE", "").strip().lower() in {"1", "true", "yes", "on"}
        if not (self._project_number or self._verification_token) and not insecure:
            logger.critical(
                "Google Chat: refusing to start — neither GOOGLE_CHAT_PROJECT_NUMBER "
                "(modern OIDC bearer JWT) nor GOOGLE_CHAT_VERIFICATION_TOKEN (legacy "
                "body token) is set. Inbound webhooks would be unauthenticated. Set "
                "GOOGLE_CHAT_PROJECT_NUMBER (preferred) or set "
                "PREDACORE_GCHAT_INSECURE=1 to opt out."
            )
            return
        if not self._project_number and self._verification_token:
            logger.warning(
                "Google Chat: only the legacy body-token check is configured; "
                "set GOOGLE_CHAT_PROJECT_NUMBER for OIDC bearer JWT verification."
            )
        self._app = web.Application()
        self._app.router.add_post("/google_chat/webhook", self._handle_webhook)
        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        bind_host = os.environ.get("PREDACORE_GCHAT_BIND_HOST", "127.0.0.1")
        await web.TCPSite(self._runner, bind_host, self._port).start()
        if self._project_number:
            logger.info(
                "Google Chat adapter started — port %d (OIDC bearer JWT verification active for project %s)",
                self._port, self._project_number,
            )
        elif self._verification_token:
            logger.info(
                "Google Chat adapter started — port %d (legacy body-token verification only)",
                self._port,
            )
        else:
            logger.warning(
                "Google Chat adapter started — port %d (PREDACORE_GCHAT_INSECURE=1; "
                "auth disabled, accepting all inbound POSTs)",
                self._port,
            )

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception:  # noqa: BLE001
                pass
        logger.info("Google Chat adapter stopped")

    async def send(self, message: OutgoingMessage) -> None:
        """Out-of-band send via the Chat REST API isn't wired in this adapter
        — replies happen inline in the webhook response. Logging only."""
        logger.debug("Google Chat: send() called out-of-band — not wired (use webhook reply)")

    async def _refresh_certs(self) -> None:
        """Fetch Google Chat service-account public certs (PEM X.509)."""
        try:
            import httpx
            from cryptography import x509
            from cryptography.hazmat.primitives import serialization
        except ImportError as exc:
            logger.error("Google Chat: missing cryptography/httpx for JWT verification: %s", exc)
            return
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=10.0)
        try:
            resp = await self._http.get(_GOOGLE_CERTS_URL)
            resp.raise_for_status()
            certs_dict = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Google Chat: certs fetch failed: %s", exc)
            return
        new_certs: dict[str, object] = {}
        for kid, pem in certs_dict.items():
            if not isinstance(pem, str):
                continue
            try:
                cert = x509.load_pem_x509_certificate(pem.encode("utf-8"))
                new_certs[kid] = cert.public_key()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Google Chat: ignoring unparseable cert kid=%s: %s", kid, exc)
        if new_certs:
            self._certs = new_certs
            self._certs_expires_at = time.time() + _JWKS_CACHE_TTL_SECONDS
            logger.debug("Google Chat: refreshed %d certs", len(new_certs))

    async def _ensure_certs(self) -> None:
        async with self._certs_lock:
            if not self._certs or time.time() >= self._certs_expires_at:
                await self._refresh_certs()

    async def _verify_jwt(self, auth_header: str) -> bool:
        """Verify Google Chat OIDC bearer JWT against project number audience."""
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
        except ImportError:
            return False
        if not self._project_number:
            return False
        if not auth_header or not auth_header.lower().startswith("bearer "):
            return False
        token = auth_header[7:].strip()
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
        if header.get("alg") != "RS256":
            return False
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            return False
        await self._ensure_certs()
        public_key = self._certs.get(kid)
        if public_key is None:
            # Maybe key rotated since last fetch — force refresh once.
            self._certs_expires_at = 0.0
            await self._ensure_certs()
            public_key = self._certs.get(kid)
        if public_key is None:
            return False
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        try:
            public_key.verify(
                received_sig, signing_input, padding.PKCS1v15(), hashes.SHA256(),
            )
        except Exception:  # noqa: BLE001 — InvalidSignature is part of cryptography.exceptions
            return False
        # Validate claims.
        iss = payload.get("iss")
        if iss != _GOOGLE_CHAT_ISSUER:
            return False
        aud = payload.get("aud")
        # aud may be a string or list
        if isinstance(aud, list):
            if self._project_number not in [str(a) for a in aud]:
                return False
        elif str(aud) != self._project_number:
            return False
        now = int(time.time())
        nbf = payload.get("nbf")
        exp = payload.get("exp")
        if isinstance(nbf, (int, float)) and now + _JWT_CLOCK_SKEW_SECONDS < int(nbf):
            return False
        if isinstance(exp, (int, float)) and now - _JWT_CLOCK_SKEW_SECONDS > int(exp):
            return False
        return True

    def _verify_legacy_token(self, body_token: str) -> bool:
        if not self._verification_token:
            return False
        return hmac.compare_digest(str(body_token), self._verification_token)

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return web.Response(status=400, text="bad request")

        # Modern path: OIDC bearer JWT
        if self._project_number:
            auth_header = request.headers.get("Authorization", "")
            if not await self._verify_jwt(auth_header):
                logger.warning("Google Chat: JWT verification failed")
                return web.Response(status=401, text="invalid token")
        # Legacy path: body-supplied verification_token (only if JWT not configured)
        elif self._verification_token:
            if not self._verify_legacy_token(body.get("token", "")):
                logger.warning("Google Chat: legacy token mismatch — rejecting")
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
