"""Matrix Channel Adapter — connects PredaCore to any Matrix homeserver.

Matrix is the open federated chat protocol behind Element, Beeper, and
many self-hosted servers. One bot account works across the entire
federation — talk to it from element.io, your homeserver, or any
Matrix client.

Setup
-----
1. Create a bot account on your homeserver (matrix.org or self-hosted).
2. Get an access token. Easiest way::

       curl -X POST -H 'Content-Type: application/json' \\
            -d '{"type":"m.login.password","user":"bot","password":"..."}' \\
            https://matrix.org/_matrix/client/v3/login

   The ``access_token`` field in the response is what you need.

3. Set environment variables (or ``~/.predacore/.env``)::

       MATRIX_HOMESERVER=https://matrix.org
       MATRIX_USER_ID=@yourbot:matrix.org
       MATRIX_ACCESS_TOKEN=syt_...

4. Add ``matrix`` to ``channels.enabled`` in config.yaml — daemon
   hot-attaches via PR T8, no restart.

5. From any Matrix client, invite the bot to a room. It auto-joins on
   the next sync tick.

Implementation notes
--------------------
- Lazy-imports ``matrix-nio`` so the core install doesn't pull it in.
- DM-only by default. Group rooms are joined but only direct mentions
  + replies trigger a response (matches our Telegram/Discord defaults).
- Streams replies via the StreamingMessageBuffer (T9 pattern). Matrix's
  edit primitive is ``m.replace`` with a relates-to header.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage
from .streaming import StreamingMessageBuffer

logger = logging.getLogger(__name__)

MATRIX_MAX_LENGTH = 65_000  # Matrix events allow ~64KB of body — generous


class MatrixAdapter(ChannelAdapter):
    """Matrix protocol adapter via matrix-nio."""

    channel_name = "matrix"
    channel_capabilities = {
        "supports_media": True,
        "supports_buttons": False,
        "supports_embeds": False,
        "supports_markdown": True,
        "max_message_length": MATRIX_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        super().__init__()
        self.config = config
        cfg = getattr(config.channels, "__dict__", {}).get("matrix", {}) or {}

        self._homeserver = (
            cfg.get("homeserver")
            or os.environ.get("MATRIX_HOMESERVER", "")
        )
        self._user_id = (
            cfg.get("user_id")
            or os.environ.get("MATRIX_USER_ID", "")
        )
        self._access_token = (
            cfg.get("access_token")
            or os.environ.get("MATRIX_ACCESS_TOKEN", "")
        )

        self._client = None              # nio.AsyncClient
        self._sync_task: asyncio.Task | None = None
        self._running = False
        # Track our own user id so we don't reply to our own messages
        self._self_event_ids: set[str] = set()

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        if not (self._homeserver and self._user_id and self._access_token):
            logger.error(
                "Matrix: missing MATRIX_HOMESERVER / MATRIX_USER_ID / "
                "MATRIX_ACCESS_TOKEN — set them in ~/.predacore/.env"
            )
            return
        try:
            import nio  # type: ignore[import-not-found]
        except ImportError:
            logger.error(
                "Matrix: 'matrix-nio' package not installed. "
                "pip install matrix-nio  (or: pip install predacore[matrix])"
            )
            return

        self._client = nio.AsyncClient(self._homeserver, self._user_id)
        self._client.access_token = self._access_token
        # Tell nio our device id is whatever the homeserver assigned —
        # without one, end-to-end encryption support breaks. We don't ship
        # E2EE here yet (would need olm), so unencrypted rooms only.
        self._client.device_id = "PREDACORE"
        self._client.add_event_callback(self._on_message_event, nio.RoomMessageText)
        self._client.add_event_callback(self._on_invite, nio.InviteMemberEvent)

        self._running = True
        self._sync_task = asyncio.create_task(
            self._client.sync_forever(timeout=30000),
            name="matrix-sync",
        )
        logger.info("Matrix adapter started (user=%s, homeserver=%s)",
                    self._user_id, self._homeserver)

    async def stop(self) -> None:
        self._running = False
        if self._sync_task is not None and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:  # noqa: BLE001
                pass
        logger.info("Matrix adapter stopped")

    # ── Outgoing ──────────────────────────────────────────────────────

    async def send(self, message: OutgoingMessage) -> None:
        """Send a text message into a Matrix room. ``user_id`` is the room id."""
        if self._client is None:
            logger.error("Matrix: not connected, dropping outgoing")
            return
        try:
            import nio  # type: ignore[import-not-found]
            await self._client.room_send(
                room_id=message.user_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": message.text or "",
                    # Use markdown-rendered HTML when possible — Element + most
                    # clients will render it; plain ``body`` is the fallback.
                    "format": "org.matrix.custom.html",
                    "formatted_body": _render_markdown(message.text or ""),
                },
            )
            del nio  # silence unused import warning
        except Exception as exc:  # noqa: BLE001
            logger.error("Matrix send failed (room=%s): %s", message.user_id, exc)

    # ── Inbound ───────────────────────────────────────────────────────

    async def _on_invite(self, room: Any, event: Any) -> None:
        """Auto-accept room invites for the bot user."""
        if event.state_key != self._user_id:
            return
        try:
            await self._client.join(room.room_id)
            logger.info("Matrix: auto-joined room %s", room.room_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Matrix join failed for %s: %s", room.room_id, exc)

    async def _on_message_event(self, room: Any, event: Any) -> None:
        """Route incoming text messages through the gateway."""
        sender = getattr(event, "sender", "")
        body = getattr(event, "body", "") or ""
        event_id = getattr(event, "event_id", "")

        if sender == self._user_id:
            return  # our own echo
        if event_id in self._self_event_ids:
            return
        if not body.strip():
            return

        # In group rooms, only respond when explicitly mentioned.
        # The Matrix convention is the bot's display name or user id
        # appearing in the body.
        is_dm = self._is_dm(room)
        is_mention = (
            self._user_id in body
            or self._user_id.split(":")[0] in body
        )
        if not (is_dm or is_mention):
            return

        logger.info("Matrix message from %s in %s: %s",
                    sender, room.room_id, body[:100])

        if self._message_handler is None:
            return

        buffer = self._make_stream_buffer(room.room_id)
        outgoing = await self._message_handler(
            IncomingMessage(
                channel=self.channel_name,
                user_id=room.room_id,
                text=body,
                metadata={
                    "matrix_sender": sender,
                    "room_id": room.room_id,
                    "event_id": event_id,
                    "is_dm": is_dm,
                },
            ),
            stream_fn=buffer.feed,
        )
        if outgoing is not None:
            handle = await buffer.flush(outgoing.text)
            if handle is None:
                await self.send(outgoing)

    @staticmethod
    def _is_dm(room: Any) -> bool:
        """Heuristic: a 2-member room is a DM."""
        try:
            return len(room.users) <= 2
        except Exception:  # noqa: BLE001
            return False

    def _make_stream_buffer(self, room_id: str) -> StreamingMessageBuffer:
        """Per-message streaming buffer wired to this Matrix room.

        Matrix's edit primitive: send a new event with ``m.relates_to``
        of type ``m.replace`` referencing the original event id. The
        buffer's ``edit`` callback constructs that edit event.
        """
        async def _send_initial(text: str):
            if self._client is None:
                return None
            resp = await self._client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content={"msgtype": "m.text", "body": text},
            )
            event_id = getattr(resp, "event_id", None)
            if event_id:
                self._self_event_ids.add(event_id)
            return event_id

        async def _edit(event_id: str, text: str):
            if self._client is None or event_id is None:
                return
            await self._client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": "* " + text,
                    "m.new_content": {"msgtype": "m.text", "body": text},
                    "m.relates_to": {
                        "rel_type": "m.replace",
                        "event_id": event_id,
                    },
                },
            )

        return StreamingMessageBuffer(
            send_initial=_send_initial,
            edit=_edit,
            edit_min_interval_seconds=1.0,
            max_chars=8000,  # well under Matrix's 64K body limit
        )


def _render_markdown(text: str) -> str:
    """Minimal markdown → HTML for Matrix's ``formatted_body`` field.

    We don't pull in a markdown lib — Matrix clients render the plain
    ``body`` field as fallback, so HTML is best-effort. Just escape the
    brokenest cases: angle brackets and ampersands.
    """
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
    )
