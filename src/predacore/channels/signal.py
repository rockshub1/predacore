"""
Signal Channel Adapter — via ``signal-cli-rest-api``.

Signal doesn't publish an official bot API. The community-standard path
is `signal-cli <https://github.com/AsamK/signal-cli>`_ wrapped by
`signal-cli-rest-api <https://github.com/bbernhard/signal-cli-rest-api>`_
which exposes a small HTTP surface the user runs locally (usually
via Docker).

Setup
-----
1. Run signal-cli-rest-api in Docker on ``localhost:8080``::

       docker run -d --name signal-api -p 8080:8080 \\
         -v $HOME/.local/share/signal-cli:/home/.local/share/signal-cli \\
         -e MODE=json-rpc \\
         bbernhard/signal-cli-rest-api

2. Register your phone number with Signal::

       curl -X POST \\
         "http://localhost:8080/v1/register/+15551234567"
       # Then verify the SMS code:
       curl -X POST \\
         "http://localhost:8080/v1/register/+15551234567/verify/123456"

3. Store config in ``~/.predacore/.env`` (or via ``secret_set``)::

       SIGNAL_API_URL=http://localhost:8080
       SIGNAL_NUMBER=+15551234567

4. ``channels.enabled`` list → add ``"signal"``; restart the daemon.

Features
--------
- 1-on-1 DMs (group messages can be added later — see ``_handle_message``)
- Text only (media routing exists in the REST API; we don't surface it yet)
- Chunking at 4 000 chars (Signal's soft limit)
- Long-polls the receive endpoint with ``timeout=30`` — gentle on resources
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

SIGNAL_MAX_LENGTH = 4_000
_RECEIVE_TIMEOUT_S = 30  # server-side long-poll window
_IDLE_SLEEP_S = 0.5      # breathing room between receive polls


def _chunk_message(text: str, max_len: int = SIGNAL_MAX_LENGTH) -> list[str]:
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


class SignalAdapter(ChannelAdapter):
    """Signal bridge via local signal-cli-rest-api."""

    channel_name = "signal"
    channel_capabilities = {
        "supports_media": False,   # TODO: pass attachments from /v2/send
        "supports_buttons": False,
        "supports_embeds": False,
        "supports_markdown": False,
        "max_message_length": SIGNAL_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        self.config = config
        self._message_handler = None
        cfg = getattr(config.channels, "__dict__", {}).get("signal", {}) or {}
        self._api_url = (
            os.getenv("SIGNAL_API_URL", "")
            or cfg.get("api_url", "")
            or "http://localhost:8080"
        ).rstrip("/")
        self._number = (
            os.getenv("SIGNAL_NUMBER", "")
            or cfg.get("number", "")
        )
        self._client: httpx.AsyncClient | None = None
        self._recv_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if not self._number:
            logger.error(
                "Signal: missing SIGNAL_NUMBER — add it to ~/.predacore/.env "
                "(format: +15551234567)"
            )
            return
        self._client = httpx.AsyncClient(
            base_url=self._api_url,
            timeout=_RECEIVE_TIMEOUT_S + 10,
        )

        # Optional liveness check — log + bail if the REST API isn't up,
        # but don't crash the gateway.
        try:
            r = await self._client.get("/v1/about", timeout=5)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error(
                "Signal REST API unreachable at %s (%s). "
                "Is `signal-cli-rest-api` running?",
                self._api_url, exc,
            )
            await self.stop()
            return

        self._running = True
        self._recv_task = asyncio.create_task(self._receive_loop(), name="signal-recv")
        logger.info("Signal adapter started (number=%s, api=%s)", self._number, self._api_url)

    async def stop(self) -> None:
        self._running = False
        if self._recv_task is not None and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None
        logger.info("Signal adapter stopped")

    async def send(self, message: OutgoingMessage) -> None:
        if self._client is None:
            logger.error("Signal: not connected, dropping outgoing message")
            return
        for chunk in _chunk_message(message.text):
            payload = {
                "message": chunk,
                "number": self._number,
                "recipients": [message.user_id],  # user_id is the recipient's phone number
            }
            try:
                r = await self._client.post("/v2/send", json=payload, timeout=15)
                r.raise_for_status()
            except httpx.HTTPError as exc:
                logger.error("Signal send failed: %s", exc)
                return

    # ── Receive loop ─────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        assert self._client is not None
        while self._running:
            try:
                r = await self._client.get(
                    f"/v1/receive/{self._number}",
                    params={"timeout": _RECEIVE_TIMEOUT_S},
                )
                r.raise_for_status()
            except httpx.HTTPError as exc:
                logger.debug("Signal receive error (will retry): %s", exc)
                await asyncio.sleep(2.0)
                continue

            envelopes = r.json() if r.content else []
            if not isinstance(envelopes, list):
                continue
            for env in envelopes:
                await self._handle_envelope(env)
            await asyncio.sleep(_IDLE_SLEEP_S)

    async def _handle_envelope(self, env: dict[str, Any]) -> None:
        try:
            envelope = env.get("envelope", env) or {}
            data_msg = envelope.get("dataMessage") or {}
            text = (data_msg.get("message") or "").strip()
            source = envelope.get("source") or envelope.get("sourceNumber") or ""
        except AttributeError:
            return
        if not text or not source:
            return
        if source == self._number:
            return  # our own echo

        logger.info("Signal DM from %s: %s", source, text[:100])
        if self._message_handler is None:
            return
        outgoing = await self._message_handler(
            IncomingMessage(
                channel=self.channel_name,
                user_id=source,
                text=text,
                metadata={
                    "signal_source": source,
                    "signal_timestamp": data_msg.get("timestamp"),
                },
            )
        )
        if outgoing is not None:
            await self.send(outgoing)
