"""
Comprehensive tests for predacore.sessions — JSONL-based session persistence.

Tests: Message dataclass, Session (add, truncate, context window, tokens),
SessionStore (create, get, persist, delete, LRU cache, path traversal).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from predacore.sessions import Message, Session, SessionStore


# ── Message Dataclass ──────────────────────────────────────────────


class TestMessage:
    """Tests for the Message dataclass."""

    def test_basic_creation(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.timestamp > 0
        assert msg.metadata == {}
        assert msg.tool_name is None
        assert msg.tool_args is None
        assert msg.tool_result is None

    def test_to_dict_minimal(self):
        msg = Message(role="user", content="hi", timestamp=1000.0)
        d = msg.to_dict()
        assert d == {"role": "user", "content": "hi", "timestamp": 1000.0}
        assert "metadata" not in d
        assert "tool_name" not in d

    def test_to_dict_with_tool(self):
        msg = Message(
            role="tool",
            content="result",
            tool_name="web_search",
            tool_args={"query": "test"},
            tool_result="found it",
        )
        d = msg.to_dict()
        assert d["tool_name"] == "web_search"
        assert d["tool_args"] == {"query": "test"}
        assert d["tool_result"] == "found it"

    def test_to_dict_with_metadata(self):
        msg = Message(role="user", content="x", metadata={"source": "telegram"})
        d = msg.to_dict()
        assert d["metadata"] == {"source": "telegram"}

    def test_from_dict_minimal(self):
        msg = Message.from_dict({"role": "assistant", "content": "hello"})
        assert msg.role == "assistant"
        assert msg.content == "hello"
        assert msg.tool_name is None

    def test_from_dict_with_tools(self):
        data = {
            "role": "tool",
            "content": "res",
            "tool_name": "read_file",
            "tool_args": {"path": "/tmp/x"},
            "tool_result": "data",
        }
        msg = Message.from_dict(data)
        assert msg.tool_name == "read_file"
        assert msg.tool_args == {"path": "/tmp/x"}
        assert msg.tool_result == "data"

    def test_roundtrip(self):
        original = Message(
            role="assistant",
            content="response",
            timestamp=12345.0,
            metadata={"tokens": 50},
            tool_name="speak",
            tool_args={"text": "hi"},
            tool_result="spoken",
        )
        restored = Message.from_dict(original.to_dict())
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.timestamp == original.timestamp
        assert restored.tool_name == original.tool_name

    def test_to_llm_format(self):
        msg = Message(role="user", content="test", tool_name="x")
        llm = msg.to_llm_format()
        assert llm == {"role": "user", "content": "test"}
        assert "tool_name" not in llm


# ── Session ────────────────────────────────────────────────────────


class TestSession:
    """Tests for the Session dataclass."""

    def test_basic_creation(self):
        s = Session()
        assert s.session_id  # UUID generated
        assert s.user_id == "default"
        assert s.title == ""
        assert s.messages == []
        assert s.message_count == 0

    def test_add_message(self):
        s = Session()
        msg = s.add_message("user", "hello")
        assert isinstance(msg, Message)
        assert msg.role == "user"
        assert msg.content == "hello"
        assert s.message_count == 1

    def test_add_message_updates_timestamp(self):
        s = Session()
        old_time = s.updated_at
        time.sleep(0.01)
        s.add_message("user", "hi")
        assert s.updated_at >= old_time

    def test_auto_title_from_first_user_message(self):
        s = Session()
        s.add_message("user", "How do I configure PredaCore?")
        assert s.title == "How do I configure PredaCore?"

    def test_auto_title_only_on_first(self):
        s = Session()
        s.add_message("user", "First message")
        s.add_message("user", "Second message")
        assert s.title == "First message"

    def test_auto_title_not_from_assistant(self):
        s = Session()
        s.add_message("assistant", "Hi there!")
        assert s.title == ""

    def test_smart_title_short(self):
        assert Session._smart_title("Short text") == "Short text"

    def test_smart_title_truncation(self):
        long_text = "a " * 100  # 200 chars
        title = Session._smart_title(long_text, max_len=80)
        assert len(title) <= 83  # 80 + "..."
        assert title.endswith("...")

    def test_smart_title_word_boundary(self):
        text = "This is a very long sentence that needs to be truncated at a word boundary for readability"
        title = Session._smart_title(text, max_len=40)
        assert not title.endswith(" ...")  # Should not end with space before ellipsis
        assert title.endswith("...")

    def test_max_messages_truncation(self):
        s = Session(max_messages=5)
        for i in range(10):
            s.add_message("user", f"Message {i}")
        assert s.message_count == 5
        assert s.messages[0].content == "Message 5"  # Oldest kept
        assert s.messages[-1].content == "Message 9"  # Newest

    def test_context_cache_cleared_on_add(self):
        s = Session()
        s.add_message("user", "hello")
        # Build cache
        s.build_context_window()
        assert len(s._context_cache) > 0
        # Adding message should clear cache
        s.add_message("assistant", "hi")
        assert len(s._context_cache) == 0

    def test_get_llm_messages_basic(self):
        s = Session()
        s.add_message("user", "hello")
        s.add_message("assistant", "hi there")
        msgs = s.get_llm_messages()
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "hello"}
        assert msgs[1] == {"role": "assistant", "content": "hi there"}

    def test_get_llm_messages_filters_tool_role(self):
        s = Session()
        s.add_message("user", "hello")
        s.add_message("tool", "tool result")
        s.add_message("assistant", "done")
        msgs = s.get_llm_messages()
        assert len(msgs) == 2  # tool messages filtered

    def test_get_llm_messages_with_limit(self):
        s = Session()
        for i in range(20):
            s.add_message("user", f"msg {i}")
        msgs = s.get_llm_messages(max_messages=5)
        assert len(msgs) <= 6  # 5 recent + possible summary

    def test_get_llm_messages_compression_summary(self):
        s = Session()
        for i in range(60):
            role = "user" if i % 2 == 0 else "assistant"
            s.add_message(role, f"{'User' if role == 'user' else 'Assistant'} message {i}")
        msgs = s.get_llm_messages(max_messages=10)
        # Should have a summary system message prepended
        assert any("Earlier in this session" in m.get("content", "") for m in msgs)


class TestSessionEstimateTokens:
    """Tests for token estimation."""

    def test_empty_string(self):
        assert Session.estimate_tokens("") == 0

    def test_short_text(self):
        tokens = Session.estimate_tokens("hello world")
        assert tokens > 0
        assert tokens < 10

    def test_longer_text(self):
        text = "This is a longer piece of text that should be estimated properly. " * 10
        tokens = Session.estimate_tokens(text)
        assert 50 < tokens < 300

    def test_minimum_one(self):
        assert Session.estimate_tokens("a") >= 1


class TestSessionTrimContent:
    """Tests for content trimming."""

    def test_short_content_unchanged(self):
        result = Session.trim_content_for_context("short", max_tokens=100)
        assert result == "short"

    def test_empty_content(self):
        assert Session.trim_content_for_context("", max_tokens=100) == ""

    def test_zero_tokens(self):
        assert Session.trim_content_for_context("text", max_tokens=0) == ""

    def test_long_content_trimmed(self):
        long_text = "x" * 10000
        result = Session.trim_content_for_context(long_text, max_tokens=100)
        assert len(result) < len(long_text)
        assert "trimmed for context" in result

    def test_head_tail_preserved(self):
        text = "HEAD_MARKER " + "x" * 5000 + " TAIL_MARKER"
        result = Session.trim_content_for_context(text, max_tokens=50)
        assert result.startswith("HEAD_MARKER")
        assert result.endswith("TAIL_MARKER")


class TestBuildContextWindow:
    """Tests for the context window packing algorithm."""

    def test_empty_session(self):
        s = Session()
        assert s.build_context_window() == []

    def test_few_messages_all_included(self):
        s = Session()
        s.add_message("user", "hello")
        s.add_message("assistant", "hi")
        window = s.build_context_window()
        assert len(window) == 2

    def test_many_messages_capped(self):
        s = Session()
        for i in range(100):
            role = "user" if i % 2 == 0 else "assistant"
            s.add_message(role, f"Message {i} with some content here")
        window = s.build_context_window(max_total_tokens=500)
        # Should be fewer than 100 messages
        assert len(window) < 100
        assert len(window) > 0

    def test_summary_prepended_when_messages_dropped(self):
        s = Session()
        # Use longer messages to exceed token budget
        for i in range(100):
            role = "user" if i % 2 == 0 else "assistant"
            s.add_message(role, f"Discussion about topic {i}: " + "detail " * 50)
        window = s.build_context_window(max_total_tokens=2000, keep_recent_messages=10)
        # With long messages exceeding budget, some must be dropped
        assert len(window) < 100

    def test_caching_works(self):
        s = Session()
        s.add_message("user", "test")
        s.add_message("assistant", "response")
        w1 = s.build_context_window()
        w2 = s.build_context_window()
        assert w1 == w2

    def test_cache_invalidated_on_new_message(self):
        s = Session()
        s.add_message("user", "test")
        w1 = s.build_context_window()
        s.add_message("assistant", "reply")
        w2 = s.build_context_window()
        assert len(w2) > len(w1)

    def test_recent_messages_prioritized(self):
        s = Session()
        for i in range(50):
            s.add_message("user", f"old message {i}")
        s.add_message("user", "THE_MOST_RECENT_MESSAGE")
        window = s.build_context_window(max_total_tokens=500, keep_recent_messages=5)
        contents = [m["content"] for m in window]
        assert any("THE_MOST_RECENT_MESSAGE" in c for c in contents)

    def test_get_context_summary(self):
        s = Session(title="Test Session")
        s.add_message("user", "hello")
        s.add_message("assistant", "hi")
        summary = s.get_context_summary()
        assert "Test Session" in summary
        assert "2 messages" in summary


# ── SessionStore ───────────────────────────────────────────────────


class TestSessionStore:
    """Tests for persistent session storage."""

    def test_init_creates_directory(self, tmp_path):
        store_dir = tmp_path / "sessions"
        store = SessionStore(str(store_dir))
        assert store_dir.exists()

    def test_create_session(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        session = store.create("user-1", "Test Session")
        assert session.user_id == "user-1"
        assert session.title == "Test Session"
        assert session.session_id

    def test_create_with_custom_id(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        session = store.create(session_id="custom-123")
        assert session.session_id == "custom-123"

    def test_get_session(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        created = store.create("user-1")
        retrieved = store.get(created.session_id)
        assert retrieved is not None
        assert retrieved.session_id == created.session_id

    def test_get_nonexistent_returns_none(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        assert store.get("nonexistent") is None

    def test_get_or_create_existing(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        created = store.create("user-1", session_id="sess-1")
        retrieved = store.get_or_create("sess-1", "user-1")
        assert retrieved.session_id == created.session_id

    def test_get_or_create_new(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        session = store.get_or_create("new-sess", "user-1")
        assert session.session_id == "new-sess"

    def test_append_message(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        session = store.create("user-1")
        msg = store.append_message(session.session_id, "user", "hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert session.message_count == 1

    def test_append_to_nonexistent_raises(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        with pytest.raises(ValueError, match="not found"):
            store.append_message("nonexistent", "user", "hello")

    def test_message_persistence(self, tmp_path):
        store_dir = tmp_path / "sessions"
        store = SessionStore(str(store_dir))
        session = store.create("user-1")
        store.append_message(session.session_id, "user", "hello world")

        # Create new store pointing to same directory (simulates restart)
        store2 = SessionStore(str(store_dir))
        loaded = store2.get(session.session_id)
        assert loaded is not None
        assert loaded.message_count == 1
        assert loaded.messages[0].content == "hello world"

    def test_meta_persistence(self, tmp_path):
        store_dir = tmp_path / "sessions"
        store = SessionStore(str(store_dir))
        session = store.create("user-1", "My Session")

        store2 = SessionStore(str(store_dir))
        loaded = store2.get(session.session_id)
        assert loaded.title == "My Session"
        assert loaded.user_id == "user-1"

    def test_list_sessions(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        store.create("user-1", "Session A")
        store.create("user-1", "Session B")
        store.create("user-2", "Session C")

        all_sessions = store.list_sessions()
        assert len(all_sessions) == 3

        user1_sessions = store.list_sessions(user_id="user-1")
        assert len(user1_sessions) == 2

    def test_list_sessions_sorted_by_updated(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        s1 = store.create("user-1", "Old")
        time.sleep(0.01)
        s2 = store.create("user-1", "New")
        sessions = store.list_sessions()
        assert sessions[0].title == "New"

    def test_list_sessions_limit(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        for i in range(10):
            store.create("user-1", f"Session {i}")
        sessions = store.list_sessions(limit=3)
        assert len(sessions) == 3

    def test_delete_session(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        session = store.create("user-1")
        sid = session.session_id
        assert store.delete(sid) is True
        assert store.get(sid) is None

    def test_delete_nonexistent(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        assert store.delete("nonexistent") is False

    def test_path_traversal_protection_get(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        assert store.get("../../../etc/passwd") is None

    def test_path_traversal_protection_delete(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        assert store.delete("../../../etc/passwd") is False

    def test_unsafe_session_ids(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        assert SessionStore._is_safe_session_id("normal-id") is True
        assert SessionStore._is_safe_session_id("../hack") is False
        assert SessionStore._is_safe_session_id("path/slash") is False
        assert SessionStore._is_safe_session_id("back\\slash") is False

    def test_lru_cache_eviction(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        store._SESSION_CACHE_MAX = 3  # Small cache for testing
        ids = []
        for i in range(5):
            s = store.create("user-1", f"Session {i}")
            ids.append(s.session_id)
        # Cache should only have 3 most recent
        assert len(store._cache) <= 3

    def test_corrupt_jsonl_tolerance(self, tmp_path):
        store_dir = tmp_path / "sessions"
        store = SessionStore(str(store_dir))
        session = store.create("user-1")
        store.append_message(session.session_id, "user", "good message")

        # Inject corrupt line into messages.jsonl
        messages_file = store_dir / session.session_id / "messages.jsonl"
        with open(messages_file, "a") as f:
            f.write("THIS IS NOT VALID JSON\n")
        store.append_message(session.session_id, "user", "another good message")

        # Reload from disk (clear cache)
        store2 = SessionStore(str(store_dir))
        loaded = store2.get(session.session_id)
        # Should load the valid messages and skip the corrupt line
        assert loaded is not None
        assert loaded.message_count == 2  # Both good messages

    def test_atomic_meta_write(self, tmp_path):
        store = SessionStore(str(tmp_path / "sessions"))
        session = store.create("user-1", "Test")
        meta_file = tmp_path / "sessions" / session.session_id / "meta.json"
        # Verify meta.json exists and is valid JSON
        assert meta_file.exists()
        data = json.loads(meta_file.read_text())
        assert data["session_id"] == session.session_id
        assert data["title"] == "Test"

    def test_append_no_temp_files_left(self, tmp_path):
        """Bug fix: _append_message_to_disk should not leave temp files."""
        store = SessionStore(str(tmp_path / "sessions"))
        session = store.create("user-1")
        store.append_message(session.session_id, "user", "hello")
        store.append_message(session.session_id, "assistant", "hi there")

        session_dir = tmp_path / "sessions" / session.session_id
        tmp_files = list(session_dir.glob("*.tmp"))
        assert len(tmp_files) == 0, f"Temp files left behind: {tmp_files}"

    def test_append_validates_json(self, tmp_path):
        """Messages written to disk are valid JSON."""
        store = SessionStore(str(tmp_path / "sessions"))
        session = store.create("user-1")
        store.append_message(session.session_id, "user", "test message")

        messages_file = tmp_path / "sessions" / session.session_id / "messages.jsonl"
        for line in messages_file.read_text().strip().split("\n"):
            data = json.loads(line)  # Should not raise
            assert data["role"] == "user"
            assert data["content"] == "test message"


# ── Legacy Stub Sanitizer ──────────────────────────────────────────
#
# The sanitizer strips old synthetic [Calling tool: X] / [Tool Result: X]
# stubs from saved assistant messages at load time. These stubs were
# introduced by the pre-migration flat-text tool round-trip and would
# re-teach the model the bracket syntax via in-context learning if left
# in loaded history. Sanitizer keeps existing sessions self-healing.


class TestLegacyStubSanitizer:
    """Tests for _sanitize_legacy_tool_stubs in predacore.sessions."""

    def test_empty_string_returns_empty(self):
        from predacore.sessions import _sanitize_legacy_tool_stubs
        assert _sanitize_legacy_tool_stubs("") == ""

    def test_none_returns_none(self):
        from predacore.sessions import _sanitize_legacy_tool_stubs
        # Function should handle falsy input gracefully
        assert _sanitize_legacy_tool_stubs(None) is None  # type: ignore

    def test_plain_text_unchanged(self):
        from predacore.sessions import _sanitize_legacy_tool_stubs
        assert (
            _sanitize_legacy_tool_stubs("hello world")
            == "hello world"
        )

    def test_strips_calling_tool_stub_at_end(self):
        from predacore.sessions import _sanitize_legacy_tool_stubs
        dirty = "now build again ✦\n\n[Calling tool: run_command]"
        assert _sanitize_legacy_tool_stubs(dirty) == "now build again ✦"

    def test_strips_calling_tool_stub_in_middle(self):
        from predacore.sessions import _sanitize_legacy_tool_stubs
        dirty = "step 1\n[Calling tool: read_file]\nstep 2"
        result = _sanitize_legacy_tool_stubs(dirty)
        assert "[Calling tool:" not in result
        assert "step 1" in result
        assert "step 2" in result

    def test_strips_tool_result_stub(self):
        from predacore.sessions import _sanitize_legacy_tool_stubs
        dirty = "[Tool Result: read_file]\nfile contents here"
        result = _sanitize_legacy_tool_stubs(dirty)
        assert "Tool Result" not in result

    def test_strips_multiple_calling_tool_stubs(self):
        from predacore.sessions import _sanitize_legacy_tool_stubs
        dirty = (
            "summary\n"
            "[Calling tool: read_file]\n"
            "more context\n"
            "[Calling tool: write_file]\n"
            "final note"
        )
        result = _sanitize_legacy_tool_stubs(dirty)
        assert "[Calling tool:" not in result
        assert "summary" in result
        assert "more context" in result
        assert "final note" in result

    def test_case_insensitive_match(self):
        from predacore.sessions import _sanitize_legacy_tool_stubs
        dirty = "note\n[calling tool: foo]\nend"
        result = _sanitize_legacy_tool_stubs(dirty)
        assert "[calling tool:" not in result.lower()

    def test_collapses_triple_newlines(self):
        from predacore.sessions import _sanitize_legacy_tool_stubs
        dirty = "line 1\n\n\n\nline 2"
        result = _sanitize_legacy_tool_stubs(dirty)
        # No 3+ consecutive newlines after cleanup
        assert "\n\n\n" not in result

    def test_real_world_poisoned_message_from_bug(self):
        """The exact string from the production poisoned session that
        was cleaned up as part of the migration — messages.jsonl line 50."""
        from predacore.sessions import _sanitize_legacy_tool_stubs
        dirty = "\n\nnow build again \u2726\n\n[Calling tool: run_command]"
        result = _sanitize_legacy_tool_stubs(dirty)
        assert result == "now build again ✦"

    def test_session_build_context_window_applies_sanitizer(self, tmp_path):
        """End-to-end: session history with poisoned stubs gets sanitized
        when build_context_window constructs the prompt."""
        store = SessionStore(str(tmp_path / "sessions"))
        session = store.create("user-1")
        store.append_message(session.session_id, "user", "build it")
        store.append_message(
            session.session_id,
            "assistant",
            "now building ✦\n\n[Calling tool: run_command]",
        )

        fresh = store.get(session.session_id)
        context = fresh.build_context_window(
            max_total_tokens=4000,
            keep_recent_messages=10,
            summary_max_tokens=500,
        )
        assistant_msgs = [m for m in context if m["role"] == "assistant"]
        assert assistant_msgs
        for m in assistant_msgs:
            assert "[Calling tool:" not in m["content"], (
                f"stub leaked into context: {m['content']!r}"
            )
