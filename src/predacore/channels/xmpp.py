"""XMPP Channel Adapter — slixmpp-based, async-native.

XMPP is the open-standard chat protocol behind Jabber, Conversations,
many self-hosted servers (ejabberd, Prosody, MongooseIM). This adapter
uses ``slixmpp`` 1.14+, which is asyncio-native.

Setup
-----
1. Have a JID (e.g. ``bot@example.com``) and password on any XMPP server.
2. Set env (or ``~/.predacore/.env``)::

       XMPP_JID=bot@example.com
       XMPP_PASSWORD=...
       XMPP_HOST=example.com   # optional — overrides JID's domain

3. Add ``xmpp`` to ``channels.enabled``.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)
XMPP_MAX_LENGTH = 65_000


class XmppAdapter(ChannelAdapter):
    channel_name = "xmpp"
    channel_capabilities = {
        "supports_media": False, "supports_buttons": False,
        "supports_embeds": False, "supports_markdown": False,
        "max_message_length": XMPP_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        super().__init__()
        self.config = config
        cfg = getattr(config.channels, "__dict__", {}).get("xmpp", {}) or {}
        self._jid = cfg.get("jid") or os.environ.get("XMPP_JID", "")
        self._password = cfg.get("password") or os.environ.get("XMPP_PASSWORD", "")
        self._host = cfg.get("host") or os.environ.get("XMPP_HOST", "")
        self._client = None
        self._running = False

    async def start(self) -> None:
        if not (self._jid and self._password):
            logger.error("XMPP: missing XMPP_JID / XMPP_PASSWORD")
            return
        try:
            from slixmpp import ClientXMPP  # type: ignore[import-not-found]
        except ImportError:
            logger.error("XMPP: pip install slixmpp")
            return

        class _Client(ClientXMPP):
            pass

        self._client = _Client(self._jid, self._password)
        self._client.add_event_handler("session_start", self._on_session_start)
        self._client.add_event_handler("message", self._on_message)
        if self._host:
            self._client.connect((self._host, 5222), use_tls=True)
        else:
            self._client.connect()
        self._running = True
        logger.info("XMPP adapter started (jid=%s)", self._jid)

    async def stop(self) -> None:
        self._running = False
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:  # noqa: BLE001
                pass
        logger.info("XMPP adapter stopped")

    async def send(self, message: OutgoingMessage) -> None:
        if self._client is None:
            return
        body = (message.text or "")[:XMPP_MAX_LENGTH]
        try:
            self._client.send_message(mto=message.user_id, mbody=body, mtype="chat")
        except Exception as exc:  # noqa: BLE001
            logger.error("XMPP send failed: %s", exc)

    async def _on_session_start(self, _event: Any) -> None:
        if self._client is None:
            return
        self._client.send_presence()
        try:
            await self._client.get_roster()
        except Exception:  # noqa: BLE001
            pass

    async def _on_message(self, msg: Any) -> None:
        # slixmpp dispatches every <message/>; filter to chat type only.
        if msg.get("type") not in ("chat", "normal"):
            return
        sender = str(msg.get("from").bare) if msg.get("from") else ""
        body = msg.get("body") or ""
        if not sender or not body:
            return
        if sender.startswith(self._jid.split("/")[0]):
            return  # echo of our own JID
        if self._message_handler is None:
            return
        outgoing = await self._message_handler(
            IncomingMessage(
                channel=self.channel_name, user_id=sender, text=body,
                metadata={"xmpp_sender": sender},
            )
        )
        if outgoing is not None:
            await self.send(outgoing)
