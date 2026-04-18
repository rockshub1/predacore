"""
Session Transcript Persistence — JSONL export for auditing and replay.

Provides:
  - Per-session JSONL transcript files
  - Atomic append-only writes
  - Session metadata headers
  - Full message history with timestamps, tool calls, and metadata
  - Replay support for debugging

Usage:
    writer = TranscriptWriter(output_dir=Path("./transcripts"))
    writer.start_session("session-123", user_id="user-1", channel="telegram")
    writer.append_message("session-123", role="user", content="Hello")
    writer.append_message("session-123", role="assistant", content="Hi!")
    writer.close_session("session-123")
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TranscriptEntry:
    """A single entry in a session transcript."""

    timestamp: float
    event_type: str  # "message" | "tool_call" | "tool_result" | "error" | "meta"
    role: str | None  # "user" | "assistant" | "system" | None
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.timestamp,
            "event": self.event_type,
            "role": self.role,
            "content": self.content,
            "meta": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptEntry:
        return cls(
            timestamp=data["ts"],
            event_type=data["event"],
            role=data.get("role"),
            content=data.get("content", ""),
            metadata=data.get("meta", {}),
        )


class TranscriptWriter:
    """
    Append-only JSONL writer for session transcripts.

    Each session gets its own file: {output_dir}/{session_id}.jsonl
    Each line is a JSON object representing one event.
    """

    def __init__(self, output_dir: Path):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._handles: dict[str, Any] = {}  # session_id -> file handle

    def _get_path(self, session_id: str) -> Path:
        """Get the transcript file path for a session."""
        safe_id = session_id.replace("/", "_").replace("..", "_")
        return self._output_dir / f"{safe_id}.jsonl"

    def start_session(
        self,
        session_id: str,
        user_id: str = "",
        channel: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Start a new transcript for a session.

        Writes a header entry with session metadata.
        Returns the path to the transcript file.
        """
        path = self._get_path(session_id)

        # Write header
        header = TranscriptEntry(
            timestamp=time.time(),
            event_type="session_start",
            role=None,
            content="",
            metadata={
                "session_id": session_id,
                "user_id": user_id,
                "channel": channel,
                **(metadata or {}),
            },
        )
        self._append(path, header)

        logger.debug("Transcript started: %s", path)
        return path

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a message to the session transcript."""
        path = self._get_path(session_id)
        entry = TranscriptEntry(
            timestamp=time.time(),
            event_type="message",
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self._append(path, entry)

    def append_tool_call(
        self,
        session_id: str,
        tool_name: str,
        args: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a tool call event to the transcript."""
        path = self._get_path(session_id)
        entry = TranscriptEntry(
            timestamp=time.time(),
            event_type="tool_call",
            role=None,
            content=json.dumps({"tool": tool_name, "args": args}),
            metadata=metadata or {},
        )
        self._append(path, entry)

    def append_tool_result(
        self,
        session_id: str,
        tool_name: str,
        result: str,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a tool result to the transcript."""
        path = self._get_path(session_id)
        entry = TranscriptEntry(
            timestamp=time.time(),
            event_type="tool_result",
            role=None,
            content=json.dumps(
                {"tool": tool_name, "result": result, "success": success}
            ),
            metadata=metadata or {},
        )
        self._append(path, entry)

    def append_error(
        self,
        session_id: str,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append an error event to the transcript."""
        path = self._get_path(session_id)
        entry = TranscriptEntry(
            timestamp=time.time(),
            event_type="error",
            role=None,
            content=error,
            metadata=metadata or {},
        )
        self._append(path, entry)

    def close_session(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write a session_end entry and close."""
        path = self._get_path(session_id)
        entry = TranscriptEntry(
            timestamp=time.time(),
            event_type="session_end",
            role=None,
            content="",
            metadata=metadata or {},
        )
        self._append(path, entry)
        logger.debug("Transcript closed: %s", path)

    def _append(self, path: Path, entry: TranscriptEntry) -> None:
        """Append a single JSONL line to the transcript file."""
        line = json.dumps(entry.to_dict(), separators=(",", ":")) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    def read_transcript(self, session_id: str) -> list[TranscriptEntry]:
        """Read all entries from a session transcript."""
        path = self._get_path(session_id)
        if not path.exists():
            return []

        entries = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        entries.append(TranscriptEntry.from_dict(data))
                    except json.JSONDecodeError:
                        logger.warning("Corrupt transcript line in %s", path)
        return entries

    def list_sessions(self) -> list[str]:
        """List all session IDs that have transcripts."""
        return [p.stem for p in sorted(self._output_dir.glob("*.jsonl"))]

    def get_transcript_path(self, session_id: str) -> Path | None:
        """Get the file path for a session transcript, if it exists."""
        path = self._get_path(session_id)
        return path if path.exists() else None

    def prune_old(self, max_age_days: int = 90) -> int:
        """Delete transcript files older than max_age_days. Returns count deleted."""
        import os
        cutoff = time.time() - (max_age_days * 86400)
        deleted = 0
        for path in self._output_dir.glob("*.jsonl"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    deleted += 1
            except OSError:
                continue
        if deleted:
            logger.info("Pruned %d transcript files older than %d days", deleted, max_age_days)
        return deleted

    def get_stats(self) -> dict[str, Any]:
        """Get transcript statistics."""
        files = list(self._output_dir.glob("*.jsonl"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            "total_sessions": len(files),
            "total_size_bytes": total_size,
            "output_dir": str(self._output_dir),
        }
