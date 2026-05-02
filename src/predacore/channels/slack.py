"""
Slack Channel Adapter — Socket Mode (no public URL required).

Slack's Socket Mode gives us a long-polling-equivalent: the bot opens a
WebSocket to Slack's infra and receives events over it. That's perfect
for personal-agent use — no reverse tunnel, no public HTTPS endpoint.

Setup
-----
1. Create an app at https://api.slack.com/apps → "From scratch"
2. Basic Information → "App-Level Tokens" → Generate (scope:
   ``connections:write``) → copy the ``xapp-...`` token.
3. Socket Mode → enable it.
4. OAuth & Permissions → add bot scopes:
       - ``app_mentions:read``
       - ``chat:write``
       - ``im:history``
       - ``im:read``
       - ``im:write``
5. Event Subscriptions → enable; subscribe to bot events:
       - ``app_mention``
       - ``message.im``
6. Install the app to your workspace → copy the Bot User OAuth token
   (``xoxb-...``).
7. Write both tokens to ``~/.predacore/.env`` (or use ``secret_set`` in
   chat):
       - ``SLACK_APP_TOKEN=xapp-...``
       - ``SLACK_BOT_TOKEN=xoxb-...``
8. Add ``slack`` to ``channels.enabled`` in ``config.yaml`` and restart
   the daemon (or ``channel_configure action=add channel=slack``).

Features
--------
- DMs + ``@mention`` mentions in channels (nothing else — privacy first)
- Markdown pass-through (Slack renders ``*bold*`` / ``_italic_`` natively)
- Message chunking at 40 000 chars (Slack's hard limit)
- Graceful degradation when the ``slack-sdk`` library is missing
"""
from __future__ import annotations

import logging
import os
from typing import Any

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

# Slack's hard per-message limit is 40 000 chars; be a little conservative.
SLACK_MAX_LENGTH = 39_000


def _chunk_message(text: str, max_len: int = SLACK_MAX_LENGTH) -> list[str]:
    """Split a long response into Slack-safe chunks, preferring newline boundaries."""
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


class SlackAdapter(ChannelAdapter):
    """Slack Socket Mode adapter — one WebSocket, both directions."""

    channel_name = "slack"
    channel_capabilities = {
        "supports_media": True,
        "supports_buttons": True,
        "supports_embeds": True,
        "supports_markdown": True,  # Slack flavor, but close enough
        "max_message_length": SLACK_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        self.config = config
        self._message_handler = None

        # Tokens are read from env (the setup wizard and secret_set both
        # write there). Fallback to the ChannelConfig dict-style fields
        # for parity with the other adapters.
        slack_cfg = getattr(config.channels, "__dict__", {}).get("slack", {}) or {}
        self._app_token = os.getenv("SLACK_APP_TOKEN", "") or slack_cfg.get("app_token", "")
        self._bot_token = os.getenv("SLACK_BOT_TOKEN", "") or slack_cfg.get("bot_token", "")

        self._web_client = None
        self._socket_handler = None
        self._bot_user_id: str | None = None

    async def start(self) -> None:
        """Open the Socket Mode WebSocket + cache our bot user id."""
        if not self._app_token or not self._bot_token:
            logger.error(
                "Slack: missing tokens — set SLACK_APP_TOKEN + SLACK_BOT_TOKEN "
                "in ~/.predacore/.env (or via secret_set)"
            )
            return

        try:
            from slack_sdk.socket_mode.aiohttp import (
                SocketModeClient,  # type: ignore[import-not-found]
            )
            from slack_sdk.socket_mode.request import (
                SocketModeRequest,  # type: ignore[import-not-found]
            )
            from slack_sdk.socket_mode.response import (
                SocketModeResponse,  # type: ignore[import-not-found]
            )
            from slack_sdk.web.async_client import (
                AsyncWebClient,  # type: ignore[import-not-found]
            )
        except ImportError:
            logger.error(
                "Slack: slack-sdk not installed. "
                "pip install slack-sdk>=3.27 (or: pip install predacore[slack])"
            )
            return

        self._web_client = AsyncWebClient(token=self._bot_token)

        # Cache our own user id so we don't reply to ourselves on echoes.
        try:
            auth = await self._web_client.auth_test()
            self._bot_user_id = auth.get("user_id", "")
            logger.info("Slack: authenticated as %s (%s)", auth.get("user"), self._bot_user_id)
        except Exception as exc:  # slack_sdk raises a rich hierarchy; log and continue
            logger.error("Slack auth_test failed: %s", exc)
            return

        self._socket_handler = SocketModeClient(
            app_token=self._app_token,
            web_client=self._web_client,
        )

        async def _on_event(client: SocketModeClient, req: SocketModeRequest) -> None:
            try:
                # Ack every request first — Slack retries if we take too long.
                await client.send_socket_mode_response(
                    SocketModeResponse(envelope_id=req.envelope_id)
                )
            except Exception as exc:
                logger.debug("Slack ack failed: %s", exc)

            if req.type != "events_api":
                return
            event = (req.payload or {}).get("event", {}) or {}
            await self._handle_event(event)

        self._socket_handler.socket_mode_request_listeners.append(_on_event)

        await self._socket_handler.connect()
        logger.info("Slack adapter connected via Socket Mode")

    async def stop(self) -> None:
        if self._socket_handler is not None:
            try:
                await self._socket_handler.disconnect()
            except Exception:
                pass
        if self._web_client is not None:
            try:
                await self._web_client.close()
            except Exception:
                pass
        logger.info("Slack adapter stopped")

    async def send(self, message: OutgoingMessage) -> None:
        """Post a message to the channel ID carried on the outgoing envelope.

        We reuse ``OutgoingMessage.user_id`` as the Slack conversation
        (channel/DM) ID because that's how Slack routes — you post to
        ``C...`` or ``D...`` even for 1:1 DMs.
        """
        if self._web_client is None:
            logger.error("Slack: not connected, dropping outgoing message")
            return
        for chunk in _chunk_message(message.text):
            try:
                await self._web_client.chat_postMessage(
                    channel=message.user_id,
                    text=chunk,
                    mrkdwn=True,
                )
            except Exception as exc:
                logger.error("Slack send failed: %s", exc)
                return

    # ── Internal ────────────────────────────────────────────────────

    async def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type", "")
        subtype = event.get("subtype", "")
        user = event.get("user", "")
        text = event.get("text", "") or ""
        channel = event.get("channel", "")
        thread_ts = event.get("thread_ts") or event.get("ts")

        # Skip our own messages + bot echoes + edits/deletes.
        if not user or user == self._bot_user_id:
            return
        if subtype in {"bot_message", "message_changed", "message_deleted"}:
            return

        is_dm = event.get("channel_type") == "im"
        is_mention = event_type == "app_mention" or (
            self._bot_user_id and f"<@{self._bot_user_id}>" in text
        )

        # Privacy default: respond to DMs and explicit mentions only.
        if not (is_dm or is_mention):
            return

        # Strip the bot mention from the text we forward to the engine.
        if self._bot_user_id:
            text = text.replace(f"<@{self._bot_user_id}>", "").strip()
        if not text:
            return

        logger.info("Slack %s from %s: %s", "DM" if is_dm else "mention", user, text[:100])

        if self._message_handler is None:
            return

        # T9 — Slack allows ~1 edit/sec/msg via chat.update. Buffer streams
        # tokens into a single message; on flush we leave the final content
        # in place so threading / metadata is preserved.
        buffer = self._make_stream_buffer(channel, thread_ts)

        outgoing = await self._message_handler(
            IncomingMessage(
                channel=self.channel_name,
                user_id=channel,  # used as the target channel ID on reply
                text=text,
                metadata={
                    "slack_user": user,
                    "slack_channel": channel,
                    "slack_thread_ts": thread_ts,
                    "is_dm": is_dm,
                },
            ),
            stream_fn=buffer.feed,
        )
        if outgoing is not None:
            handle = await buffer.flush(outgoing.text)
            if handle is None:
                await self.send(outgoing)

    def _make_stream_buffer(
        self, channel: str, thread_ts: str | None,
    ) -> "StreamingMessageBuffer":
        """Per-event Slack streaming buffer.

        Uses chat.postMessage for the placeholder and chat.update for
        edits. Slack's per-message limit is 4000 chars (text field) —
        we cap at 3500 to leave room for the cursor + multibyte runes.
        """
        from .streaming import StreamingMessageBuffer

        async def _send_initial(text: str):
            if self._web_client is None:
                return None
            kwargs = {"channel": channel, "text": text, "mrkdwn": True}
            if thread_ts:
                kwargs["thread_ts"] = thread_ts
            resp = await self._web_client.chat_postMessage(**kwargs)
            # Slack returns a `ts` (timestamp) that uniquely identifies the
            # message in the channel — that's what chat.update needs.
            return resp.get("ts") if hasattr(resp, "get") else resp["ts"]

        async def _edit(ts: str, text: str):
            if self._web_client is None or ts is None:
                return
            await self._web_client.chat_update(
                channel=channel, ts=ts, text=text,
            )

        return StreamingMessageBuffer(
            send_initial=_send_initial,
            edit=_edit,
            edit_min_interval_seconds=1.0,
            max_chars=3500,
        )
