"""IRC Channel Adapter — pydle (asyncio-native IRCv3 client).

Setup
-----
1. Pick a server + nickname (e.g. ``irc.libera.chat``).
2. Set env (or ``~/.predacore/.env``)::

       IRC_SERVER=irc.libera.chat
       IRC_NICKNAME=predacore_bot
       IRC_PORT=6697               # default 6697 (TLS)
       IRC_CHANNELS=#mychannel,#another   # comma-separated
       IRC_PASSWORD=...            # optional, NickServ identify

3. Add ``irc`` to ``channels.enabled``.
"""
from __future__ import annotations

import asyncio
import logging
import os

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)
IRC_MAX_LENGTH = 400  # IRC line limit is 510 bytes; leave room for prefix


class IrcAdapter(ChannelAdapter):
    channel_name = "irc"
    channel_capabilities = {
        "supports_media": False, "supports_buttons": False,
        "supports_embeds": False, "supports_markdown": False,
        "max_message_length": IRC_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        super().__init__()
        self.config = config
        cfg = getattr(config.channels, "__dict__", {}).get("irc", {}) or {}
        self._server = cfg.get("server") or os.environ.get("IRC_SERVER", "irc.libera.chat")
        self._nickname = cfg.get("nickname") or os.environ.get("IRC_NICKNAME", "predacore_bot")
        self._port = int(os.environ.get("IRC_PORT", "") or cfg.get("port", 6697))
        self._channels = [
            c.strip() for c in (
                os.environ.get("IRC_CHANNELS", "")
                or ",".join(cfg.get("channels", []) or [])
            ).split(",") if c.strip()
        ]
        self._password = cfg.get("password") or os.environ.get("IRC_PASSWORD", "")
        self._client = None
        self._connect_task: asyncio.Task | None = None

    async def start(self) -> None:
        try:
            import pydle  # type: ignore[import-not-found]
        except ImportError:
            logger.error("IRC: pip install pydle")
            return

        adapter_self = self

        class _Bot(pydle.Client):
            async def on_connect(self):  # type: ignore[override]
                for ch in adapter_self._channels:
                    await self.join(ch)

            async def on_message(self, target, source, message):  # type: ignore[override]
                if source == self.nickname:
                    return
                # Direct messages (target == our nick) → user_id is source;
                # channel messages → user_id is the channel name.
                user_id = source if target == self.nickname else target
                if adapter_self._message_handler is None:
                    return
                outgoing = await adapter_self._message_handler(
                    IncomingMessage(
                        channel=adapter_self.channel_name,
                        user_id=user_id, text=message,
                        metadata={"irc_source": source, "irc_target": target},
                    )
                )
                if outgoing is not None:
                    await adapter_self.send(outgoing)

        self._client = _Bot(self._nickname)
        self._connect_task = asyncio.create_task(
            self._client.connect(
                hostname=self._server, port=self._port, tls=True,
                password=self._password or None,
            ),
            name="irc-connect",
        )
        logger.info("IRC adapter started (%s as %s)", self._server, self._nickname)

    async def stop(self) -> None:
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:  # noqa: BLE001
                pass
        if self._connect_task and not self._connect_task.done():
            self._connect_task.cancel()
            try:
                await self._connect_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        logger.info("IRC adapter stopped")

    async def send(self, message: OutgoingMessage) -> None:
        if self._client is None:
            return
        text = message.text or ""
        # IRC has hard line-length limits; split on newlines too.
        for line in text.splitlines() or [text]:
            chunks = [line[i:i+IRC_MAX_LENGTH] for i in range(0, len(line), IRC_MAX_LENGTH)] or [""]
            for chunk in chunks:
                try:
                    await self._client.message(message.user_id, chunk)
                except Exception as exc:  # noqa: BLE001
                    logger.error("IRC send failed: %s", exc)
                    return
