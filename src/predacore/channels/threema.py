"""Threema Gateway Channel Adapter — paid Threema bot service.

The official Threema *Gateway* is a paid service (per-message billing).
You buy a Gateway ID + secret + API identity at https://gateway.threema.ch,
then the ``threema.gateway`` Python SDK lets you send + receive messages.

This adapter is **send-only** (REST out). Inbound requires running an
HTTPS callback URL that Threema POSTs to with end-to-end encrypted
payloads — wiring the inbound path needs a public TLS endpoint and a
keypair, both of which are user-specific. Skipping inbound here keeps
the adapter useful for notifications today; full bidirectional support
lands when we have a pattern for the keypair lifecycle.

Setup
-----
1. Buy Gateway credentials at https://gateway.threema.ch.
2. Set env (or ``~/.predacore/.env``)::

       THREEMA_GATEWAY_ID=*MYBOT01     # 8 chars, starts with *
       THREEMA_GATEWAY_SECRET=...
       THREEMA_PRIVATE_KEY=...         # 64-char hex, optional (E2E mode)

3. Add ``threema`` to ``channels.enabled``.
"""
from __future__ import annotations

import asyncio
import logging
import os

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, OutgoingMessage

logger = logging.getLogger(__name__)
THREEMA_MAX_LENGTH = 3500


class ThreemaAdapter(ChannelAdapter):
    channel_name = "threema"
    channel_capabilities = {
        "supports_media": True, "supports_buttons": False,
        "supports_embeds": False, "supports_markdown": False,
        "max_message_length": THREEMA_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        super().__init__()
        self.config = config
        cfg = getattr(config.channels, "__dict__", {}).get("threema", {}) or {}
        self._gateway_id = cfg.get("gateway_id") or os.environ.get("THREEMA_GATEWAY_ID", "")
        self._secret = cfg.get("secret") or os.environ.get("THREEMA_GATEWAY_SECRET", "")
        self._private_key = cfg.get("private_key") or os.environ.get("THREEMA_PRIVATE_KEY", "")
        self._connection = None

    async def start(self) -> None:
        if not (self._gateway_id and self._secret):
            logger.error("Threema: missing THREEMA_GATEWAY_ID / THREEMA_GATEWAY_SECRET")
            return
        try:
            from threema.gateway import Connection  # type: ignore[import-not-found]
        except ImportError:
            logger.error("Threema: pip install threema.gateway")
            return
        kwargs = {"identity": self._gateway_id, "secret": self._secret}
        if self._private_key:
            kwargs["key"] = self._private_key
        try:
            self._connection = Connection(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.error("Threema: Connection init failed: %s", exc)
            return
        logger.info("Threema adapter started (gateway_id=%s, e2e=%s)",
                    self._gateway_id, bool(self._private_key))

    async def stop(self) -> None:
        if self._connection is not None:
            try:
                await self._connection.close()
            except Exception:  # noqa: BLE001
                pass
        logger.info("Threema adapter stopped")

    async def send(self, message: OutgoingMessage) -> None:
        if self._connection is None:
            return
        text = (message.text or "")[:THREEMA_MAX_LENGTH]
        try:
            from threema.gateway.simple import TextMessage  # type: ignore[import-not-found]
            msg = TextMessage(connection=self._connection, to_id=message.user_id, text=text)
            await msg.send()
        except ImportError:
            # Fall back to e2e mode if simple submodule unavailable
            try:
                from threema.gateway.e2e import TextMessage as E2EText  # type: ignore[import-not-found]
                msg = E2EText(connection=self._connection, to_id=message.user_id, text=text)
                await msg.send()
            except Exception as exc:  # noqa: BLE001
                logger.error("Threema send failed (e2e path): %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("Threema send failed: %s", exc)
