"""
Session Persistence — Conversation history stored as JSONL.

Every interaction with PredaCore is persisted so conversations survive
restarts. Sessions are stored in ~/.predacore/sessions/ as individual
JSONL files (one JSON object per line = easy to append, read, grep).

Usage:
    store = SessionStore("/path/to/sessions")
    session = store.create("user_123")
    session.add_message("user", "Hello PredaCore")
    session.add_message("assistant", "Hello! How can I help?")
    store.save(session)
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Legacy stub sanitizer ─────────────────────────────────────────────
# Older agent-loop turns saved synthetic `[Calling tool: X]` /
# `[Tool Result: X]` text stubs into session history because the loop
# used flat strings instead of structured Anthropic content blocks for
# tool round trips. Replaying those stubs into the prompt teaches the
# model — via in-context learning — to emit the bracket syntax as prose
# instead of real tool_use blocks. Strip them on load so existing
# sessions self-heal the next time they're used.
_LEGACY_CALLING_STUB_RE = re.compile(
    r"\s*\[\s*Calling tool\s*:\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\]\s*",
    re.IGNORECASE,
)
_LEGACY_TOOL_RESULT_STUB_RE = re.compile(
    r"\s*\[\s*Tool Result\s*:\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\][^\n]*",
    re.IGNORECASE,
)


def _sanitize_legacy_tool_stubs(text: str) -> str:
    """Strip legacy synthetic tool-call / tool-result stubs from saved text."""
    if not text:
        return text
    cleaned = _LEGACY_CALLING_STUB_RE.sub("\n", text)
    cleaned = _LEGACY_TOOL_RESULT_STUB_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Optional fields for tool calls
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        if self.tool_name:
            d["tool_name"] = self.tool_name
        if self.tool_args:
            d["tool_args"] = self.tool_args
        if self.tool_result is not None:
            d["tool_result"] = self.tool_result
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
            tool_name=data.get("tool_name"),
            tool_args=data.get("tool_args"),
            tool_result=data.get("tool_result"),
        )

    def to_llm_format(self) -> dict[str, str]:
        """Convert to {"role": ..., "content": ...} for LLM API."""
        return {"role": self.role, "content": self.content}


@dataclass
class Session:
    """A conversation session with full history."""

    MAX_MESSAGES: int = 500  # Configurable cap to prevent unbounded memory growth
    CONTEXT_KEEP_RECENT_MESSAGES: int = 24
    CONTEXT_MAX_HISTORY_TOKENS: int = 24_000
    CONTEXT_SUMMARY_MAX_TOKENS: int = 1_200

    # Token budget limits for context window packing
    OLDER_MSG_TOKEN_LIMIT = 500          # Max tokens per older (non-recent) message
    OLDER_MSG_TOKEN_FLOOR = 180          # Min tokens guaranteed per older message
    RECENT_MSG_TOKEN_LIMIT = 2_400       # Max tokens for very recent messages (rank < 8)
    RECENT_MSG_TOKEN_FLOOR = 320         # Min tokens guaranteed for very recent messages
    STALE_RECENT_MSG_TOKEN_LIMIT = 700   # Max tokens for less-recent messages (rank >= 8)
    STALE_RECENT_MSG_TOKEN_FLOOR = 220   # Min tokens guaranteed for less-recent messages
    TOP_RECENT_RANK_THRESHOLD = 8        # Messages within this rank get higher token budget
    MIN_SUMMARY_BUDGET = 80              # Min remaining budget to generate a summary
    MIN_BACKFILL_RESERVE = 120           # Stop backfilling older messages when budget drops below this + summary reserve

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default"
    title: str = ""
    channel_origin: str = ""  # Channel where session was created
    messages: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    max_messages: int = MAX_MESSAGES

    def __post_init__(self) -> None:
        """Allow config-driven overrides for session constants."""
        config = self.metadata.get("_config") if self.metadata else None
        if config is not None:
            self.MAX_MESSAGES = getattr(config, "max_session_messages", self.MAX_MESSAGES)
            self.CONTEXT_MAX_HISTORY_TOKENS = getattr(config, "context_max_history_tokens", self.CONTEXT_MAX_HISTORY_TOKENS)
            self.CONTEXT_KEEP_RECENT_MESSAGES = getattr(config, "context_keep_recent_messages", self.CONTEXT_KEEP_RECENT_MESSAGES)
            self.max_messages = self.MAX_MESSAGES
    _context_cache: dict[str, list[dict[str, str]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def add_message(self, role: str, content: str, **kwargs: Any) -> Message:
        """Add a message and update timestamp.  Truncates oldest when over limit."""
        msg = Message(role=role, content=content, **kwargs)
        self.messages.append(msg)
        # Truncate oldest messages when the cap is exceeded
        if self.max_messages > 0 and len(self.messages) > self.max_messages:
            overflow = len(self.messages) - self.max_messages
            self.messages = self.messages[overflow:]
        self.updated_at = time.time()
        self._context_cache.clear()
        if not self.title and role == "user" and len(self.messages) == 1:
            # Auto-title from first user message (smart truncation at word boundary)
            self.title = self._smart_title(content)
        return msg

    @staticmethod
    def _smart_title(text: str, max_len: int = 80) -> str:
        """Generate a clean title from user message text."""
        # Strip common greetings/fillers
        text = text.strip()
        # If short enough, use as-is
        if len(text) <= max_len:
            return text
        # Truncate at word boundary
        truncated = text[:max_len]
        last_space = truncated.rfind(" ")
        if last_space > max_len // 2:
            truncated = truncated[:last_space]
        return truncated.rstrip(".,;:!? ") + "..."

    def get_llm_messages(self, max_messages: int | None = None) -> list[dict[str, str]]:
        """
        Get messages formatted for LLM API calls.
        Optionally limit to last N messages for context window management.

        If session is very long (>50 messages) and max_messages is set,
        prepends a compressed summary of older messages so context isn't
        completely lost.
        """
        msgs = self.messages
        if max_messages and len(msgs) > max_messages:
            older = msgs[:-max_messages]
            msgs = msgs[-max_messages:]
            # Auto-compress: summarize older messages into a context note
            if len(older) > 4:
                user_topics = []
                for m in older:
                    if m.role == "user" and m.content.strip():
                        topic = m.content[:100].strip()
                        if topic:
                            user_topics.append(topic)
                if user_topics:
                    # Keep last 5 topic summaries to avoid too much noise
                    topics_str = "\n".join(f"- {t}" for t in user_topics[-5:])
                    summary_msg = Message(
                        role="system",
                        content=(
                            f"[Earlier in this session ({len(older)} messages ago), "
                            f"user discussed these topics:\n{topics_str}]"
                        ),
                    )
                    return [summary_msg.to_llm_format()] + [
                        m.to_llm_format()
                        for m in msgs
                        if m.role in ("user", "assistant", "system")
                    ]
        return [
            m.to_llm_format() for m in msgs if m.role in ("user", "assistant", "system")
        ]

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count without a tokenizer dependency."""
        if not text:
            return 0
        char_estimate = math.ceil(len(text) / 4)
        word_estimate = math.ceil(len(text.split()) * 1.15)
        return max(1, min(max(char_estimate, word_estimate), char_estimate * 2))

    @staticmethod
    def trim_content_for_context(
        content: str,
        *,
        max_tokens: int,
        head_ratio: float = 0.72,
        marker: str = "\n...[trimmed for context]...\n",
    ) -> str:
        """Compact a long message while preserving the beginning and end."""
        if max_tokens <= 0 or not content:
            return ""
        max_chars = max_tokens * 4
        if len(content) <= max_chars:
            return content
        marker_chars = len(marker)
        if max_chars <= marker_chars + 80:
            return content[:max_chars]
        head_chars = max(80, int((max_chars - marker_chars) * head_ratio))
        tail_chars = max(40, max_chars - marker_chars - head_chars)
        return content[:head_chars] + marker + content[-tail_chars:]

    @staticmethod
    def _summary_snippet(content: str, limit: int = 160) -> str:
        text = " ".join(content.split()).strip()
        if text.startswith("[Tool Result:"):
            _, _, remainder = text.partition("]")
            text = remainder.strip()
        if text.startswith("[Calling tool:"):
            return ""
        if len(text) <= limit:
            return text
        clipped = text[:limit]
        last_space = clipped.rfind(" ")
        if last_space > limit // 2:
            clipped = clipped[:last_space]
        return clipped.rstrip(".,;:!? ") + "..."

    def _build_context_summary(
        self,
        dropped_messages: list[Message],
        *,
        dropped_count: int,
        max_tokens: int,
    ) -> str:
        """Summarize older turns so the latest raw turns fit in context."""
        user_topics: list[str] = []
        assistant_notes: list[str] = []

        for msg in dropped_messages:
            snippet = self._summary_snippet(msg.content)
            if not snippet:
                continue
            if msg.role == "user" and snippet not in user_topics:
                user_topics.append(snippet)
            elif msg.role == "assistant" and snippet not in assistant_notes:
                assistant_notes.append(snippet)

        lines = [f"[Session summary: {dropped_count} earlier messages compressed.]"]
        if user_topics:
            lines.append("User topics:")
            lines.extend(f"- {topic}" for topic in user_topics[-6:])
        if assistant_notes:
            lines.append("Assistant progress:")
            lines.extend(f"- {note}" for note in assistant_notes[-4:])
        if len(lines) == 1:
            lines.append("- Earlier turns were trimmed to preserve recent context.")

        summary = "\n".join(lines)
        return self.trim_content_for_context(
            summary,
            max_tokens=max_tokens,
            head_ratio=0.82,
        )

    def build_context_window(
        self,
        *,
        max_total_tokens: int = CONTEXT_MAX_HISTORY_TOKENS,
        keep_recent_messages: int = CONTEXT_KEEP_RECENT_MESSAGES,
        summary_max_tokens: int = CONTEXT_SUMMARY_MAX_TOKENS,
    ) -> list[dict[str, str]]:
        """
        Build a compact LLM-ready history window.

        Strategy:
          - always prefer the newest raw turns
          - backfill older turns while budget allows
          - summarize anything dropped into one system message
          - cache packed output until the session changes
        """
        cache_key = (
            f"{len(self.messages)}:{self.updated_at:.6f}:"
            f"{max_total_tokens}:{keep_recent_messages}:{summary_max_tokens}"
        )
        cached = self._context_cache.get(cache_key)
        if cached is not None:
            return [dict(msg) for msg in cached]

        llm_messages = [
            m for m in self.messages if m.role in ("user", "assistant", "system")
        ]
        if not llm_messages or max_total_tokens <= 0:
            return []

        recent_source = (
            llm_messages[-keep_recent_messages:] if keep_recent_messages else []
        )
        older_source = (
            llm_messages[:-keep_recent_messages]
            if keep_recent_messages
            else llm_messages
        )

        selected: list[dict[str, str]] = []
        budget_remaining = max_total_tokens
        summary_reserve = (
            min(summary_max_tokens, max_total_tokens // 8) if older_source else 0
        )

        def _pack_message(
            msg: Message, *, recent_rank: int | None
        ) -> dict[str, str] | None:
            content = msg.content
            # Strip legacy [Calling tool: X] / [Tool Result: X] stubs from
            # assistant messages so they never re-enter the LLM prompt and
            # retrain the model to emit the bracket syntax as prose.
            if msg.role == "assistant" and content:
                content = _sanitize_legacy_tool_stubs(content)
                if not content:
                    content = "(tool call)"
            token_count = self.estimate_tokens(content)
            if recent_rank is None:
                token_limit = min(self.OLDER_MSG_TOKEN_LIMIT, max(self.OLDER_MSG_TOKEN_FLOOR, budget_remaining))
            elif recent_rank < self.TOP_RECENT_RANK_THRESHOLD:
                token_limit = min(self.RECENT_MSG_TOKEN_LIMIT, max(self.RECENT_MSG_TOKEN_FLOOR, budget_remaining))
            else:
                token_limit = min(self.STALE_RECENT_MSG_TOKEN_LIMIT, max(self.STALE_RECENT_MSG_TOKEN_FLOOR, budget_remaining))
            if token_count > token_limit:
                content = self.trim_content_for_context(content, max_tokens=token_limit)
                token_count = self.estimate_tokens(content)
            if token_count > budget_remaining:
                return None
            return {"role": msg.role, "content": content}

        for recent_rank, msg in enumerate(reversed(recent_source)):
            packed = _pack_message(msg, recent_rank=recent_rank)
            if packed is None:
                break
            selected.append(packed)
            budget_remaining -= self.estimate_tokens(packed["content"])

        for msg in reversed(older_source):
            if budget_remaining <= summary_reserve + self.MIN_BACKFILL_RESERVE:
                break
            packed = _pack_message(msg, recent_rank=None)
            if packed is None:
                break
            selected.append(packed)
            budget_remaining -= self.estimate_tokens(packed["content"])

        selected.reverse()

        dropped_count = len(llm_messages) - len(selected)
        if dropped_count > 0 and budget_remaining > self.MIN_SUMMARY_BUDGET:
            summary = self._build_context_summary(
                llm_messages[:dropped_count],
                dropped_count=dropped_count,
                max_tokens=min(summary_max_tokens, budget_remaining),
            )
            if summary:
                summary_tokens = self.estimate_tokens(summary)
                if summary_tokens <= budget_remaining:
                    selected.insert(0, {"role": "system", "content": summary})

        self._context_cache[cache_key] = [dict(msg) for msg in selected]
        return [dict(msg) for msg in selected]

    def get_context_summary(self) -> str:
        """Generate a brief summary of this session for context."""
        msg_count = len(self.messages)
        user_msgs = sum(1 for m in self.messages if m.role == "user")
        return f"Session '{self.title}' — {msg_count} messages ({user_msgs} from user)"

    @property
    def message_count(self) -> int:
        return len(self.messages)


class SessionStore:
    """
    Persistent session storage using JSONL files.

    Each session is stored as a directory under sessions_dir:
      sessions_dir/
        {session_id}/
          meta.json        — session metadata
          messages.jsonl    — one message per line (append-only)
    """

    _SESSION_CACHE_MAX = 128

    def __init__(self, sessions_dir: str):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._cache: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()  # Sync lock — session store is called from both sync and async paths
        logger.info("SessionStore initialized at %s", self.sessions_dir)

    def _cache_put(self, session_id: str, session: Session) -> None:
        """Insert or refresh a session in the LRU cache, evicting oldest if full."""
        with self._lock:
            if session_id in self._cache:
                self._cache.move_to_end(session_id)
            self._cache[session_id] = session
            while len(self._cache) > self._SESSION_CACHE_MAX:
                self._cache.popitem(last=False)

    def create(
        self,
        user_id: str = "default",
        title: str = "",
        session_id: str | None = None,
    ) -> Session:
        """Create a new session."""
        session = Session(
            session_id=session_id or str(uuid.uuid4())[:12],
            user_id=user_id,
            title=title,
        )
        self._cache_put(session.session_id, session)
        self._save_meta(session)
        logger.info("Created session %s for user %s", session.session_id, user_id)
        return session

    @staticmethod
    def _is_safe_session_id(session_id: str) -> bool:
        """Reject session IDs that could cause path traversal."""
        return (
            session_id == Path(session_id).name
            and ".." not in session_id
            and "/" not in session_id
            and "\\" not in session_id
        )

    def get(self, session_id: str) -> Session | None:
        """Load a session by ID."""
        if not self._is_safe_session_id(session_id):
            logger.warning("Rejected unsafe session_id: %s", session_id)
            return None

        with self._lock:
            if session_id in self._cache:
                self._cache.move_to_end(session_id)
                return self._cache[session_id]

        session_dir = self.sessions_dir / session_id
        if not session_dir.exists():
            return None

        return self._load_session(session_id)

    def get_or_create(self, session_id: str, user_id: str = "default") -> Session:
        """Get existing session or create a new one."""
        session = self.get(session_id)
        if session is None:
            session = self.create(user_id=user_id, session_id=session_id)
        return session

    def append_message(
        self, session_id: str, role: str, content: str, **kwargs: Any
    ) -> Message:
        """Append a message to a session and persist immediately."""
        session = self.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        msg = session.add_message(role, content, **kwargs)
        self._append_message_to_disk(session_id, msg)
        self._save_meta(session)  # Update timestamps
        return msg

    def list_sessions(
        self, user_id: str | None = None, limit: int = 20
    ) -> list[Session]:
        """List recent sessions, optionally filtered by user."""
        sessions: list[Session] = []

        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue
            meta_file = session_dir / "meta.json"
            if not meta_file.exists():
                continue

            try:
                meta = json.loads(meta_file.read_text())
                if user_id and meta.get("user_id") != user_id:
                    continue

                sessions.append(
                    Session(
                        session_id=meta["session_id"],
                        user_id=meta.get("user_id", "default"),
                        title=meta.get("title", ""),
                        created_at=meta.get("created_at", 0),
                        updated_at=meta.get("updated_at", 0),
                        metadata=meta.get("metadata", {}),
                    )
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Skipping corrupt session dir %s: %s", session_dir, e)
                continue

        # Sort by updated_at descending (most recent first)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions[:limit]

    def delete(self, session_id: str) -> bool:
        """Delete a session and its files."""
        import shutil

        # Sanitize session_id to prevent path traversal
        safe_id = Path(session_id).name
        if safe_id != session_id or ".." in session_id or "/" in session_id:
            logger.warning("Rejected suspicious session_id for delete: %s", session_id)
            return False
        session_dir = self.sessions_dir / safe_id
        if (
            session_dir.exists()
            and session_dir.resolve().parent == self.sessions_dir.resolve()
        ):
            shutil.rmtree(session_dir)
            with self._lock:
                self._cache.pop(session_id, None)
            logger.info("Deleted session %s", session_id)
            return True
        return False

    # ── Private persistence methods ──────────────────────────────

    def _save_meta(self, session: Session) -> None:
        """Write session metadata to meta.json atomically.

        Writes to a temp file in the same directory, then atomically
        replaces the target via os.replace() to prevent corruption
        from crashes mid-write.
        """
        session_dir = self.sessions_dir / session.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "message_count": session.message_count,
            "metadata": session.metadata,
        }
        final_path = session_dir / "meta.json"
        fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=str(session_dir))
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(meta, indent=2, default=str))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(final_path))
        except BaseException:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _append_message_to_disk(self, session_id: str, msg: Message) -> None:
        """Append a single message to the JSONL file (O(1) write).

        Validates JSON before writing, then appends directly with fsync.
        """
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        messages_file = session_dir / "messages.jsonl"
        line = json.dumps(msg.to_dict(), default=str) + "\n"
        # Validate the line is valid JSON before writing to disk
        json.loads(line.strip())
        with open(messages_file, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def _load_session(self, session_id: str) -> Session | None:
        """Load a full session from disk."""
        session_dir = self.sessions_dir / session_id
        meta_file = session_dir / "meta.json"
        messages_file = session_dir / "messages.jsonl"

        if not meta_file.exists():
            return None

        try:
            meta = json.loads(meta_file.read_text())
            session = Session(
                session_id=meta["session_id"],
                user_id=meta.get("user_id", "default"),
                title=meta.get("title", ""),
                created_at=meta.get("created_at", time.time()),
                updated_at=meta.get("updated_at", time.time()),
                metadata=meta.get("metadata", {}),
            )

            # Load messages
            if messages_file.exists():
                with open(messages_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                session.messages.append(
                                    Message.from_dict(json.loads(line))
                                )
                            except (json.JSONDecodeError, KeyError):
                                continue

            self._cache_put(session_id, session)
            return session

        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to load session %s: %s", session_id, e)
            return None
