"""Tests for the channel streaming buffer (T9).

Five behavioral guarantees verified:
  1. First chunk triggers send_initial; nothing edited yet.
  2. Two chunks within the throttle window collapse into a single edit
     (the deferred edit absorbs the second chunk's text).
  3. flush() after streaming overrides accumulated text and emits a
     final edit without the cursor.
  4. send_initial failure → buffer marks "streaming disabled" so the
     adapter's fallback send path runs (flush returns None).
  5. Concurrent feeds don't issue overlapping edits (lock serializes).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from predacore.channels.streaming import StreamingMessageBuffer


@pytest.mark.asyncio
async def test_first_feed_calls_send_initial_only():
    send_initial = AsyncMock(return_value="msg-1")
    edit = AsyncMock()
    buf = StreamingMessageBuffer(
        send_initial=send_initial, edit=edit, edit_min_interval_seconds=0.05,
    )
    await buf.feed("hello")
    send_initial.assert_awaited_once()
    edit.assert_not_called()
    # send_initial received accumulated text + cursor
    text_arg = send_initial.await_args.args[0]
    assert "hello" in text_arg


@pytest.mark.asyncio
async def test_throttled_chunks_collapse_into_one_edit():
    """Two rapid feeds should collapse: one initial, one deferred edit
    that includes BOTH chunks' text (proves debounce absorbs)."""
    send_initial = AsyncMock(return_value="msg-1")
    edit = AsyncMock()
    buf = StreamingMessageBuffer(
        send_initial=send_initial, edit=edit, edit_min_interval_seconds=0.10,
    )
    await buf.feed("AAA")
    await buf.feed("BBB")
    # Wait past the throttle window so the deferred edit fires.
    await asyncio.sleep(0.15)
    assert edit.await_count == 1
    edited_text = edit.await_args.args[1]
    assert "AAA" in edited_text and "BBB" in edited_text


@pytest.mark.asyncio
async def test_flush_emits_final_edit_without_cursor():
    send_initial = AsyncMock(return_value="msg-1")
    edit = AsyncMock()
    buf = StreamingMessageBuffer(
        send_initial=send_initial, edit=edit, edit_min_interval_seconds=10.0,
    )
    await buf.feed("partial")
    handle = await buf.flush(final_text="final answer")
    assert handle == "msg-1"
    # Last edit must be the final text, no cursor block character.
    final_text = edit.await_args.args[1]
    assert final_text == "final answer"


@pytest.mark.asyncio
async def test_initial_send_failure_falls_back():
    send_initial = AsyncMock(side_effect=RuntimeError("rate limited"))
    edit = AsyncMock()
    buf = StreamingMessageBuffer(
        send_initial=send_initial, edit=edit, edit_min_interval_seconds=0.05,
    )
    await buf.feed("anything")
    # initial failed → no edits attempted, even on subsequent feeds
    await buf.feed("more")
    edit.assert_not_called()
    # flush returns None so the adapter knows to use its fallback send.
    handle = await buf.flush(final_text="full reply")
    assert handle is None


@pytest.mark.asyncio
async def test_concurrent_feeds_do_not_overlap_edits():
    """Two feeds running concurrently must not produce overlapping edits.
    The internal lock serializes; one fires send_initial, the other
    waits and contributes to the deferred edit."""
    send_calls: list[float] = []

    async def _slow_send(text: str):
        send_calls.append(asyncio.get_event_loop().time())
        await asyncio.sleep(0.05)
        return "msg-1"

    edit_started: list[float] = []

    async def _slow_edit(handle, text: str):
        edit_started.append(asyncio.get_event_loop().time())
        await asyncio.sleep(0.05)

    buf = StreamingMessageBuffer(
        send_initial=_slow_send, edit=_slow_edit, edit_min_interval_seconds=0.10,
    )
    await asyncio.gather(buf.feed("X"), buf.feed("Y"))
    # Only one initial send fired
    assert len(send_calls) == 1
    # Wait for any deferred edits to fire
    await asyncio.sleep(0.20)
    # No two edits overlap: any edit's start > prior edit's start by at
    # least the edit duration (~0.05s) under serialization.
    for i in range(1, len(edit_started)):
        assert edit_started[i] - edit_started[i - 1] >= 0.04


@pytest.mark.asyncio
async def test_max_chars_truncates_during_streaming():
    send_initial = AsyncMock(return_value="msg-1")
    edit = AsyncMock()
    buf = StreamingMessageBuffer(
        send_initial=send_initial,
        edit=edit,
        edit_min_interval_seconds=0.05,
        max_chars=10,
    )
    # Feed a 50-char string; placeholder should be truncated to 10 + cursor.
    await buf.feed("X" * 50)
    text_arg = send_initial.await_args.args[0]
    # 10 X's + 1 cursor block = 11 chars
    assert text_arg.count("X") == 10
