"""Shared streaming primitive for chat-style channels (T9).

Most chat platforms (Telegram, Discord, Slack, Signal) let bots edit
their own messages. That gives us "watch the assistant type" UX over a
chat connection without paying for a websocket: send a placeholder,
then edit it as LLM tokens arrive.

The challenge is rate limiting. Each platform throttles edits:
  Telegram   ~1 edit/sec/message
  Discord    ~5 edits/5sec  (effectively 1/sec)
  Slack      ~1 edit/sec
  Signal     conservative 1.5/sec  (limits less documented)

This module provides ``StreamingMessageBuffer`` — a tiny state machine
that handles:
  - placeholder send on first chunk
  - debounced edits (last-write-wins under a min interval)
  - final flush guaranteed to land

Each adapter only needs to implement ``send_initial`` and ``edit``;
the buffer takes care of *when* to edit. No threads, no queues —
``feed()`` is awaitable and uses ``asyncio.sleep`` for backoff.

The buffer is **per-incoming-message**: instantiated by the adapter
when it receives a user message, then ``buffer.feed`` is passed as
``stream_fn`` to the gateway, and ``buffer.flush(final_text)`` runs
once the LLM is done. Don't share buffers across messages.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Cursor glyph appended to the streaming text so the user can see
# something is still in flight. Removed on flush.
_CURSOR = "█"  # full block — renders cleanly in Telegram, Discord, Slack


@dataclass
class StreamingMessageBuffer:
    """Accumulate streamed chunks and edit a single chat message.

    Lifecycle:
      1. Adapter constructs with ``send_initial`` + ``edit`` callbacks.
      2. Adapter passes ``buffer.feed`` to the gateway as ``stream_fn``.
      3. LLM tokens arrive; the buffer batches them and edits when the
         throttle window allows.
      4. After the LLM finishes, adapter calls ``buffer.flush(final_text)``
         which guarantees one last edit even if the throttle window is
         still open. ``final_text`` overrides accumulated text — useful
         because the gateway may post-process (truncate, format).

    The ``edit_min_interval_seconds`` knob is set per-adapter. Telegram
    sets ~1.0; Discord sets ~1.0 to fit its 5/5s burst; Slack ~1.0;
    Signal ~1.5 (conservative).

    If ``send_initial`` returns None (e.g. the underlying API rejected
    the placeholder), the buffer downgrades to "buffer-only" mode and
    only the flush() will actually emit a message — no streaming, but
    the conversation still completes.

    The ``max_chars`` knob caps the displayed text per edit so we don't
    hit the platform's per-message length limit during streaming.
    Excess text shows up in the final flush, which the adapter can chunk
    via its existing send path.
    """

    send_initial: Callable[[str], Awaitable[Any]]
    edit: Callable[[Any, str], Awaitable[None]]
    edit_min_interval_seconds: float = 1.0
    max_chars: int = 3500
    placeholder_text: str = "…"  # ellipsis

    # Internal state — don't construct directly.
    _accumulated: str = field(default="", init=False)
    _message_handle: Any = field(default=None, init=False)
    _last_edit_at: float = field(default=0.0, init=False)
    _pending_edit: asyncio.Task | None = field(default=None, init=False)
    _flushed: bool = field(default=False, init=False)
    _initial_send_failed: bool = field(default=False, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def feed(self, chunk: str) -> None:
        """Append ``chunk`` to the buffer; maybe edit the live message.

        Safe to call concurrently. The lock serializes edits so we never
        send two overlapping ``edit_message_text`` calls — the platform
        APIs aren't transactional, and a stale edit landing after a fresh
        one would visually rewind the message.
        """
        if self._flushed or not chunk:
            return
        async with self._lock:
            self._accumulated += chunk
            if self._initial_send_failed:
                return  # platform refused placeholder; only flush will emit
            if self._message_handle is None:
                # First chunk: send the placeholder so we have a target
                # to edit. If THIS fails, abandon streaming silently.
                try:
                    self._message_handle = await self.send_initial(
                        self._render(self._accumulated, with_cursor=True),
                    )
                except Exception as exc:  # noqa: BLE001 — platform may rate-limit
                    logger.debug(
                        "stream initial send failed (%s) — falling back to "
                        "non-streamed flush", exc,
                    )
                    self._initial_send_failed = True
                    return
                if self._message_handle is None:
                    self._initial_send_failed = True
                    return
                self._last_edit_at = time.monotonic()
                return
            await self._maybe_edit_locked(with_cursor=True)

    async def flush(self, final_text: str | None = None) -> Any:
        """Final edit / emit. ``final_text`` overrides accumulated text.

        Returns the message handle so the caller can decide what to do
        if streaming was disabled (e.g. fall back to ``adapter.send``).
        Returns None when nothing was ever sent (no chunks fed AND no
        final_text), or when the initial placeholder send failed AND no
        final_text was provided — caller should send normally.
        """
        async with self._lock:
            if self._flushed:
                return self._message_handle
            self._flushed = True
            if final_text is not None:
                self._accumulated = final_text

            if self._initial_send_failed or self._message_handle is None:
                # Streaming never started; nothing for us to flush. The
                # caller will run its normal send path.
                return None

            # Cancel any pending throttled edit — we're about to do the
            # authoritative final write.
            if self._pending_edit is not None and not self._pending_edit.done():
                self._pending_edit.cancel()
                try:
                    await self._pending_edit
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

            try:
                await self.edit(
                    self._message_handle,
                    self._render(self._accumulated, with_cursor=False),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("stream final edit failed: %s", exc)
            return self._message_handle

    # ── Internal ──────────────────────────────────────────────────────

    def _render(self, text: str, *, with_cursor: bool) -> str:
        """Format the streaming text for display: trim + optional cursor."""
        if len(text) > self.max_chars:
            text = text[: self.max_chars]
        if with_cursor:
            return text + _CURSOR
        return text

    async def _maybe_edit_locked(self, *, with_cursor: bool) -> None:
        """Edit immediately if throttle allows; else schedule a deferred edit.

        Caller must hold ``self._lock``.
        """
        now = time.monotonic()
        elapsed = now - self._last_edit_at
        if elapsed >= self.edit_min_interval_seconds:
            # Window is open — edit now.
            try:
                await self.edit(
                    self._message_handle,
                    self._render(self._accumulated, with_cursor=with_cursor),
                )
                self._last_edit_at = time.monotonic()
            except Exception as exc:  # noqa: BLE001
                logger.debug("stream edit failed (will retry on next chunk): %s", exc)
            return

        # Window not open — schedule a deferred edit if one isn't already.
        # Re-running an edit ABSORBS later chunks: by the time the deferred
        # task fires, ``self._accumulated`` reflects everything received,
        # so we don't need a queue.
        if self._pending_edit is None or self._pending_edit.done():
            wait = self.edit_min_interval_seconds - elapsed
            self._pending_edit = asyncio.create_task(
                self._deferred_edit(wait, with_cursor=with_cursor)
            )

    async def _deferred_edit(self, wait: float, *, with_cursor: bool) -> None:
        try:
            await asyncio.sleep(wait)
        except asyncio.CancelledError:
            return
        # Re-acquire lock and emit the most recent accumulated text.
        async with self._lock:
            if self._flushed or self._message_handle is None:
                return
            try:
                await self.edit(
                    self._message_handle,
                    self._render(self._accumulated, with_cursor=with_cursor),
                )
                self._last_edit_at = time.monotonic()
            except Exception as exc:  # noqa: BLE001
                logger.debug("stream deferred edit failed: %s", exc)
