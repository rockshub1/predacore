"""KakaoTalk Channel Adapter — Kakao Channel REST API + webhook.

Kakao Channel (formerly Plus Friend / 카카오톡 채널) is the Korean
business-channel API. Bots can receive messages via webhook callbacks
and reply via the Channel Message REST endpoint.

Setup
-----
1. Register your business at https://developers.kakao.com.
2. Create a Kakao Channel; verify the business owner.
3. Get the channel access token + channel public id (UUID).
4. In the Kakao Developers console, configure a **verification token**
   to be sent in the ``Authorization`` header of inbound webhooks.
5. Set env (or ``~/.predacore/.env``)::

       KAKAO_CHANNEL_ACCESS_TOKEN=...
       KAKAO_CHANNEL_PUBLIC_ID=...
       KAKAO_VERIFICATION_TOKEN=...           # bearer token Kakao sends inbound
       KAKAO_IP_ALLOWLIST=1.2.3.4,5.6.7.0/24  # optional CIDR/IP allowlist

6. In Kakao Developers console, configure the webhook URL to
   ``https://your-host:PORT/kakao/webhook``.

7. Add ``kakaotalk`` to ``channels.enabled``.

Inbound webhook authentication
------------------------------
Each request must carry ``Authorization: Bearer <verification_token>``.
We compare with ``hmac.compare_digest`` (timing-safe). When
``KAKAO_IP_ALLOWLIST`` is set, the request's remote address (last hop in
``X-Forwarded-For``, falling back to ``request.remote``) must match one
of the listed IPs or CIDR blocks.

If ``KAKAO_VERIFICATION_TOKEN`` is unset, the adapter refuses to start
unless ``PREDACORE_KAKAO_INSECURE=1`` is set explicitly.

Notes
-----
Kakao gates this API behind business verification + Korean-language
docs. Treat this as a *starter* adapter — real-world fields may need
tweaks based on your specific channel's webhook contract.
"""
from __future__ import annotations

import asyncio
import hmac
import ipaddress
import logging
import os

import httpx
from aiohttp import web

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)
KAKAO_MAX_LENGTH = 1000
KAKAO_API_BASE = "https://kapi.kakao.com/v1/api/talk/channels"


def _parse_ip_allowlist(raw: str) -> list[ipaddress._BaseNetwork]:
    nets: list[ipaddress._BaseNetwork] = []
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            nets.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            logger.warning("KakaoTalk: ignoring invalid IP/CIDR in allowlist: %r", entry)
    return nets


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
        self._verification_token = cfg.get("verification_token") or os.environ.get("KAKAO_VERIFICATION_TOKEN", "")
        ip_raw = cfg.get("ip_allowlist") or os.environ.get("KAKAO_IP_ALLOWLIST", "")
        self._ip_allowlist = _parse_ip_allowlist(ip_raw)
        self._port = int(os.environ.get("PREDACORE_KAKAO_PORT", "") or cfg.get("webhook_port", 8774))
        self._http: httpx.AsyncClient | None = None
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._bg_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        if not (self._access_token and self._channel_id):
            logger.error("KakaoTalk: missing KAKAO_CHANNEL_ACCESS_TOKEN / KAKAO_CHANNEL_PUBLIC_ID")
            return
        insecure = os.environ.get("PREDACORE_KAKAO_INSECURE", "").strip().lower() in {"1", "true", "yes", "on"}
        if not self._verification_token and not insecure:
            logger.critical(
                "KakaoTalk: refusing to start — KAKAO_VERIFICATION_TOKEN is unset. "
                "Inbound webhooks would be unauthenticated and any caller able to "
                "POST could spoof events and trigger LLM tool calls. Set the "
                "verification token from the Kakao Developers console, or set "
                "PREDACORE_KAKAO_INSECURE=1 to opt out."
            )
            return
        self._http = httpx.AsyncClient(timeout=30)
        self._app = web.Application()
        self._app.router.add_post("/kakao/webhook", self._handle_webhook)
        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        bind_host = os.environ.get("PREDACORE_KAKAO_BIND_HOST", "127.0.0.1")
        await web.TCPSite(self._runner, bind_host, self._port).start()
        if self._verification_token:
            ip_note = (
                f", IP allowlist ({len(self._ip_allowlist)} entries) active"
                if self._ip_allowlist
                else ", no IP allowlist"
            )
            logger.info(
                "KakaoTalk adapter started — port %d (Bearer token verification%s)",
                self._port, ip_note,
            )
        else:
            logger.warning(
                "KakaoTalk adapter started — port %d (PREDACORE_KAKAO_INSECURE=1; "
                "auth disabled, accepting all inbound POSTs)",
                self._port,
            )

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

    def _verify_bearer(self, auth_header: str) -> bool:
        """Verify ``Authorization: Bearer <token>`` against ``KAKAO_VERIFICATION_TOKEN``.

        Timing-safe comparison via ``hmac.compare_digest``.
        """
        if not self._verification_token:
            return False
        if not auth_header:
            return False
        token = auth_header.strip()
        if not token.lower().startswith("bearer "):
            return False
        token = token[7:].strip()
        return hmac.compare_digest(token, self._verification_token)

    def _verify_remote_ip(self, remote_ip: str) -> bool:
        """Check ``remote_ip`` against ``KAKAO_IP_ALLOWLIST`` (CIDR-aware).

        Returns True when no allowlist is configured (skip).
        """
        if not self._ip_allowlist:
            return True
        if not remote_ip:
            return False
        try:
            ip = ipaddress.ip_address(remote_ip)
        except ValueError:
            return False
        return any(ip in net for net in self._ip_allowlist)

    @staticmethod
    def _resolve_remote_ip(request: web.Request) -> str:
        """Last hop of X-Forwarded-For, else request.remote."""
        xff = request.headers.get("X-Forwarded-For", "")
        if xff:
            # Last hop is closest trusted proxy (mirrors AuthMiddleware M54 fix).
            parts = [p.strip() for p in xff.split(",") if p.strip()]
            if parts:
                return parts[-1]
        return request.remote or ""

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        if self._verification_token:
            if not self._verify_bearer(request.headers.get("Authorization", "")):
                logger.warning("KakaoTalk webhook bearer-token verification failed")
                return web.Response(status=401, text="invalid token")
            if not self._verify_remote_ip(self._resolve_remote_ip(request)):
                logger.warning("KakaoTalk webhook IP allowlist check failed")
                return web.Response(status=403, text="ip not allowed")
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
