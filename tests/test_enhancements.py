"""
Comprehensive tests for JARVIS enhancement features.

Covers:
  1. Error base class enhancements (errors.py)
  2. Notification dispatchers (alerting.py)
  3. Ethical compliance guard (dispatcher.py)
  4. Vector layer filtering (memory/store.py)
  5. Knowledge graph queries (memory/store.py)
  6. UserProfile (identity/engine.py)
  7. DB Server / Client (services/db_server.py, db_client.py)
  8. Agent SDK Adapter (tools/agent_sdk_adapter.py)
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import struct
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import pytest_asyncio

# ===========================================================================
# 1. Error Base Class Enhancements
# ===========================================================================

from jarvis.errors import (
    JARVISError,
    ToolError,
    ToolNotFoundError,
    ToolTimeoutError,
    LLMError,
    LLMRateLimitError,
    SecurityError,
    SSRFBlockedError,
)


class TestErrorEnhancements:
    """Test JARVISError structured fields (error_code, recoverable, context)."""

    def test_error_code_field(self):
        err = JARVISError("exec failed", error_code="TOOL_EXEC_FAILED")
        assert err.error_code == "TOOL_EXEC_FAILED"

    def test_recoverable_field(self):
        err = JARVISError("transient", recoverable=True)
        assert err.recoverable is True

    def test_recoverable_defaults_false(self):
        err = JARVISError("something broke")
        assert err.recoverable is False

    def test_context_field(self):
        ctx = {"tool": "web_search", "attempt": 3}
        err = JARVISError("search failed", context=ctx)
        assert err.context == ctx
        assert err.context["tool"] == "web_search"

    def test_context_defaults_empty(self):
        err = JARVISError("msg")
        assert err.context == {}

    def test_details_field(self):
        err = JARVISError("msg", details={"key": "val"})
        assert err.details == {"key": "val"}

    def test_str_includes_error_code(self):
        err = JARVISError("exec failed", error_code="TOOL_EXEC_FAILED")
        assert str(err) == "[TOOL_EXEC_FAILED] exec failed"

    def test_str_without_error_code(self):
        err = JARVISError("plain message")
        assert str(err) == "plain message"

    def test_backwards_compatible(self):
        """JARVISError('msg') still works without any new keyword args."""
        err = JARVISError("just a message")
        assert str(err) == "just a message"
        assert err.error_code == ""
        assert err.recoverable is False
        assert err.context == {}
        assert err.details == {}

    def test_inherits_from_exception(self):
        err = JARVISError("test")
        assert isinstance(err, Exception)

    def test_subclass_tool_error(self):
        err = ToolError("broken", tool_name="web_search")
        assert err.tool_name == "web_search"
        assert isinstance(err, JARVISError)

    def test_tool_timeout_error(self):
        err = ToolTimeoutError("web_search", 30.0)
        assert err.tool_name == "web_search"
        assert err.timeout_seconds == 30.0
        assert "timed out" in str(err)
        assert "30" in str(err)

    def test_llm_rate_limit_error(self):
        err = LLMRateLimitError("openai", retry_after=5.0)
        assert err.provider == "openai"
        assert err.retry_after == 5.0
        assert "Rate limited" in str(err)

    def test_error_code_only_set_when_provided(self):
        """If error_code is empty string, the class default stays."""
        err = JARVISError("msg", error_code="")
        assert err.error_code == ""
        assert str(err) == "msg"


# ===========================================================================
# 2. Notification Dispatchers (alerting.py)
# ===========================================================================

from jarvis.services.alerting import (
    Alert,
    AlertSeverity,
    AlertChannel,
    EmailDispatcher,
    DiscordDispatcher,
    SlackDispatcher,
    WebhookDispatcher,
    AlertManager,
    _is_safe_url,
)


class TestAlert:
    """Test the Alert dataclass."""

    def test_alert_to_dict(self):
        alert = Alert(
            title="test alert",
            message="something happened",
            severity=AlertSeverity.WARNING,
        )
        d = alert.to_dict()
        assert d["title"] == "test alert"
        assert d["severity"] == "warning"
        assert "timestamp" in d

    def test_alert_defaults(self):
        alert = Alert(title="t", message="m")
        assert alert.severity == AlertSeverity.WARNING
        assert alert.source == "jarvis"
        assert alert.labels == {}


class TestEmailDispatcher:
    """Test EmailDispatcher init, config reading, and email formatting."""

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("JARVIS_ALERT_SMTP_HOST", "smtp.example.com")
        monkeypatch.setenv("JARVIS_ALERT_SMTP_PORT", "465")
        monkeypatch.setenv("JARVIS_ALERT_SMTP_FROM", "jarvis@example.com")
        monkeypatch.setenv("JARVIS_ALERT_SMTP_TO", "admin@example.com,ops@example.com")
        monkeypatch.setenv("JARVIS_ALERT_SMTP_USER", "myuser")
        monkeypatch.setenv("JARVIS_ALERT_SMTP_PASS", "mypass")

        d = EmailDispatcher()
        assert d.smtp_host == "smtp.example.com"
        assert d.smtp_port == 465
        assert d.from_addr == "jarvis@example.com"
        assert d.to_addrs == ["admin@example.com", "ops@example.com"]
        assert d.username == "myuser"
        assert d.password == "mypass"
        assert d.is_configured is True

    def test_not_configured(self):
        d = EmailDispatcher()
        assert d.is_configured is False

    def test_send_format(self):
        """Mock smtplib and verify email content."""
        d = EmailDispatcher(
            smtp_host="smtp.test.com",
            smtp_port=587,
            from_addr="jarvis@test.com",
            to_addrs=["admin@test.com"],
            username="user",
            password="pass",
        )
        alert = Alert(
            title="Server Down",
            message="The main server is not responding.",
            severity=AlertSeverity.CRITICAL,
            labels={"env": "prod"},
        )

        with patch("jarvis.services.alerting.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

            result = d.send(alert)
            assert result is True

            # Verify sendmail was called
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("user", "pass")
            mock_server.sendmail.assert_called_once()

            # Inspect the email content
            call_args = mock_server.sendmail.call_args
            assert call_args[0][0] == "jarvis@test.com"
            assert call_args[0][1] == ["admin@test.com"]
            email_body = call_args[0][2]
            assert "CRITICAL" in email_body
            assert "Server Down" in email_body

    def test_send_unconfigured_returns_false(self):
        d = EmailDispatcher()
        alert = Alert(title="t", message="m")
        assert d.send(alert) is False


class TestDiscordDispatcher:
    """Test DiscordDispatcher severity colors and embed format."""

    def test_severity_colors(self):
        d = DiscordDispatcher(webhook_url="https://discord.com/api/webhooks/test")
        # INFO = green
        assert d.SEVERITY_COLORS[AlertSeverity.INFO] == 0x2ECC71
        # WARNING = yellow
        assert d.SEVERITY_COLORS[AlertSeverity.WARNING] == 0xF1C40F
        # CRITICAL = red
        assert d.SEVERITY_COLORS[AlertSeverity.CRITICAL] == 0xE74C3C
        # RESOLVED = blue
        assert d.SEVERITY_COLORS[AlertSeverity.RESOLVED] == 0x3498DB

    def test_send_embed(self):
        d = DiscordDispatcher(webhook_url="https://discord.com/api/webhooks/test")
        alert = Alert(
            title="Deploy Failed",
            message="Deployment to prod failed.",
            severity=AlertSeverity.CRITICAL,
            labels={"service": "api"},
        )

        with patch("jarvis.services.alerting.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()
            result = d.send(alert)
            assert result is True

            # Verify the request payload
            call_args = mock_urlopen.call_args
            req = call_args[0][0]
            payload = json.loads(req.data.decode("utf-8"))

            assert "embeds" in payload
            embed = payload["embeds"][0]
            assert "[CRITICAL]" in embed["title"]
            assert embed["color"] == 0xE74C3C
            assert embed["description"] == "Deployment to prod failed."
            assert len(embed["fields"]) == 1
            assert embed["fields"][0]["name"] == "service"

    def test_not_configured(self):
        d = DiscordDispatcher()
        assert d.is_configured is False
        alert = Alert(title="t", message="m")
        assert d.send(alert) is False


class TestSSRFProtection:
    """Test _is_safe_url blocks private IPs and allows public ones."""

    def test_blocks_loopback(self):
        with patch("jarvis.services.alerting.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("127.0.0.1", 80))]
            assert _is_safe_url("http://localhost/hook") is False

    def test_blocks_private_10(self):
        with patch("jarvis.services.alerting.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("10.0.0.5", 80))]
            assert _is_safe_url("http://internal.corp/hook") is False

    def test_blocks_private_192(self):
        with patch("jarvis.services.alerting.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("192.168.1.1", 80))]
            assert _is_safe_url("http://router.local/hook") is False

    def test_blocks_private_172(self):
        with patch("jarvis.services.alerting.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("172.16.0.1", 80))]
            assert _is_safe_url("http://private/hook") is False

    def test_blocks_link_local(self):
        with patch("jarvis.services.alerting.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("169.254.1.1", 80))]
            assert _is_safe_url("http://link-local/hook") is False

    def test_blocks_ipv6_loopback(self):
        with patch("jarvis.services.alerting.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("::1", 80))]
            assert _is_safe_url("http://localhost6/hook") is False

    def test_allows_public_urls(self):
        with patch("jarvis.services.alerting.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("52.12.34.56", 443))]
            assert _is_safe_url("https://hooks.slack.com/services/test") is True

    def test_blocks_non_http_schemes(self):
        assert _is_safe_url("ftp://evil.com/data") is False
        assert _is_safe_url("file:///etc/passwd") is False

    def test_blocks_dns_failure(self):
        with patch("jarvis.services.alerting.socket.getaddrinfo") as mock_gai:
            mock_gai.side_effect = socket.gaierror("Name resolution failed")
            assert _is_safe_url("http://nonexistent.invalid/hook") is False


# ===========================================================================
# 3. Ethical Compliance (dispatcher.py)
# ===========================================================================

from jarvis.tools.dispatcher import _check_ethical_compliance, _FORBIDDEN_KEYWORDS


class TestEthicalCompliance:
    """Test the _check_ethical_compliance function."""

    def test_yolo_skips_check(self):
        result = _check_ethical_compliance(
            "dangerous_tool",
            {"action": "delete_user_data"},
            trust_level="yolo",
        )
        assert result is None

    def test_paranoid_blocks_forbidden(self):
        result = _check_ethical_compliance(
            "db_tool",
            {"query": "please delete_user_data now"},
            trust_level="paranoid",
        )
        assert result is not None
        assert "Blocked by ethical compliance" in result
        assert "delete_user_data" in result

    def test_paranoid_blocks_drop_table(self):
        result = _check_ethical_compliance(
            "sql_tool",
            {"sql": "drop_table users"},
            trust_level="paranoid",
        )
        assert result is not None
        assert "drop_table" in result

    def test_normal_warns_only(self):
        """Normal trust level logs a warning but does NOT block."""
        result = _check_ethical_compliance(
            "sql_tool",
            {"sql": "drop_table users"},
            trust_level="normal",
        )
        assert result is None

    def test_clean_args_pass(self):
        """Normal args without any forbidden keywords pass cleanly."""
        result = _check_ethical_compliance(
            "web_search",
            {"query": "weather in paris"},
            trust_level="paranoid",
        )
        assert result is None

    def test_multiple_forbidden_keywords(self):
        result = _check_ethical_compliance(
            "admin_tool",
            {"cmd": "bypass_auth and disable_safety"},
            trust_level="paranoid",
        )
        assert result is not None
        assert "bypass_auth" in result
        assert "disable_safety" in result

    def test_forbidden_keywords_set(self):
        """Verify the known forbidden keywords are present."""
        assert "delete_user_data" in _FORBIDDEN_KEYWORDS
        assert "disable_safety" in _FORBIDDEN_KEYWORDS
        assert "bypass_auth" in _FORBIDDEN_KEYWORDS
        assert "drop_table" in _FORBIDDEN_KEYWORDS
        assert "truncate_table" in _FORBIDDEN_KEYWORDS
        assert "format_disk" in _FORBIDDEN_KEYWORDS

    def test_case_insensitive_matching(self):
        """Keywords are matched against lowercased args."""
        result = _check_ethical_compliance(
            "tool",
            {"cmd": "DELETE_USER_DATA"},
            trust_level="paranoid",
        )
        assert result is not None


# ===========================================================================
# 4. Vector Layer Filtering (memory/store.py - _NumpyVectorIndex)
# ===========================================================================

from jarvis.memory.store import _NumpyVectorIndex


class TestVectorLayerFiltering:
    """Test that the _NumpyVectorIndex.search() method supports layer filtering."""

    @pytest.fixture
    def vec_index(self):
        """Create a small in-memory vector index (no save path)."""
        return _NumpyVectorIndex(dimensions=3, save_path=None)

    @pytest.mark.asyncio
    async def test_search_without_layers(self, vec_index):
        """Default search returns all matching vectors regardless of layer."""
        await vec_index.add("a", [1.0, 0.0, 0.0], metadata={"layer": "core"})
        await vec_index.add("b", [0.9, 0.1, 0.0], metadata={"layer": "episodic"})
        await vec_index.add("c", [0.8, 0.2, 0.0], metadata={"layer": "semantic"})

        results = await vec_index.search([1.0, 0.0, 0.0], top_k=10, layers=None)
        ids = [r[0] for r in results]
        assert len(ids) == 3
        assert "a" in ids
        assert "b" in ids
        assert "c" in ids

    @pytest.mark.asyncio
    async def test_search_with_layers_filter(self, vec_index):
        """Searching with a layers filter should only return vectors with matching metadata."""
        await vec_index.add("a", [1.0, 0.0, 0.0], metadata={"layer": "core"})
        await vec_index.add("b", [0.9, 0.1, 0.0], metadata={"layer": "episodic"})
        await vec_index.add("c", [0.8, 0.2, 0.0], metadata={"layer": "semantic"})

        results = await vec_index.search(
            [1.0, 0.0, 0.0], top_k=10, layers={"core"}
        )
        ids = [r[0] for r in results]
        assert ids == ["a"]

    @pytest.mark.asyncio
    async def test_search_with_multiple_layers(self, vec_index):
        """Multiple layers in the filter set returns vectors from any of those layers."""
        await vec_index.add("a", [1.0, 0.0, 0.0], metadata={"layer": "core"})
        await vec_index.add("b", [0.9, 0.1, 0.0], metadata={"layer": "episodic"})
        await vec_index.add("c", [0.8, 0.2, 0.0], metadata={"layer": "semantic"})

        results = await vec_index.search(
            [1.0, 0.0, 0.0], top_k=10, layers={"core", "semantic"}
        )
        ids = [r[0] for r in results]
        assert "a" in ids
        assert "c" in ids
        assert "b" not in ids

    @pytest.mark.asyncio
    async def test_search_layer_no_match(self, vec_index):
        """If no vectors match the layer filter, return empty."""
        await vec_index.add("a", [1.0, 0.0, 0.0], metadata={"layer": "core"})

        results = await vec_index.search(
            [1.0, 0.0, 0.0], top_k=10, layers={"nonexistent"}
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_search_empty_index(self, vec_index):
        results = await vec_index.search([1.0, 0.0, 0.0], top_k=5)
        assert results == []


# ===========================================================================
# 5. Knowledge Graph Queries (memory/store.py - UnifiedMemoryStore)
# ===========================================================================

from jarvis.memory.store import UnifiedMemoryStore


class TestKnowledgeGraphQueries:
    """Test entity/relation CRUD and graph query methods on UnifiedMemoryStore."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a fresh UnifiedMemoryStore in a temp directory."""
        db_file = str(tmp_path / "test_memory.db")
        return UnifiedMemoryStore(db_path=db_file)

    @pytest.mark.asyncio
    async def test_upsert_entity(self, store):
        eid = await store.upsert_entity("Alice", entity_type="person")
        assert isinstance(eid, str)
        assert len(eid) > 0

    @pytest.mark.asyncio
    async def test_upsert_entity_idempotent(self, store):
        eid1 = await store.upsert_entity("Alice", entity_type="person")
        eid2 = await store.upsert_entity("Alice", entity_type="person")
        assert eid1 == eid2

    @pytest.mark.asyncio
    async def test_query_nodes_by_type(self, store):
        await store.upsert_entity("Alice", entity_type="person")
        await store.upsert_entity("Python", entity_type="tool")
        await store.upsert_entity("Bob", entity_type="person")

        people = await store.query_nodes(entity_type="person")
        names = [n["name"] for n in people]
        assert "Alice" in names
        assert "Bob" in names
        assert "Python" not in names

    @pytest.mark.asyncio
    async def test_query_nodes_all(self, store):
        await store.upsert_entity("Alice", entity_type="person")
        await store.upsert_entity("Python", entity_type="tool")

        all_nodes = await store.query_nodes()
        assert len(all_nodes) >= 2

    @pytest.mark.asyncio
    async def test_query_edges_by_type(self, store):
        alice_id = await store.upsert_entity("Alice", entity_type="person")
        python_id = await store.upsert_entity("Python", entity_type="tool")
        bob_id = await store.upsert_entity("Bob", entity_type="person")

        await store.add_relation(alice_id, python_id, relation_type="uses")
        await store.add_relation(alice_id, bob_id, relation_type="knows")

        uses_edges = await store.query_edges(relation_type="uses")
        assert len(uses_edges) == 1
        assert uses_edges[0]["relation_type"] == "uses"
        assert uses_edges[0]["source_entity_id"] == alice_id
        assert uses_edges[0]["target_entity_id"] == python_id

    @pytest.mark.asyncio
    async def test_query_edges_all(self, store):
        alice_id = await store.upsert_entity("Alice", entity_type="person")
        python_id = await store.upsert_entity("Python", entity_type="tool")
        bob_id = await store.upsert_entity("Bob", entity_type="person")

        await store.add_relation(alice_id, python_id, relation_type="uses")
        await store.add_relation(alice_id, bob_id, relation_type="knows")

        all_edges = await store.query_edges()
        assert len(all_edges) == 2

    @pytest.mark.asyncio
    async def test_get_neighbors(self, store):
        alice_id = await store.upsert_entity("Alice", entity_type="person")
        python_id = await store.upsert_entity("Python", entity_type="tool")

        await store.add_relation(alice_id, python_id, relation_type="uses")

        neighbors = await store.get_neighbors(alice_id, direction="outgoing")
        assert len(neighbors) == 1
        assert neighbors[0]["entity"]["name"] == "Python"
        assert neighbors[0]["relation"]["relation_type"] == "uses"

    @pytest.mark.asyncio
    async def test_get_neighbors_incoming(self, store):
        alice_id = await store.upsert_entity("Alice", entity_type="person")
        python_id = await store.upsert_entity("Python", entity_type="tool")

        await store.add_relation(alice_id, python_id, relation_type="uses")

        # Python has Alice as an incoming neighbor
        neighbors = await store.get_neighbors(python_id, direction="incoming")
        assert len(neighbors) == 1
        assert neighbors[0]["entity"]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_get_neighbors_both(self, store):
        alice_id = await store.upsert_entity("Alice", entity_type="person")
        bob_id = await store.upsert_entity("Bob", entity_type="person")

        await store.add_relation(alice_id, bob_id, relation_type="knows")
        await store.add_relation(bob_id, alice_id, relation_type="knows")

        neighbors = await store.get_neighbors(alice_id, direction="both")
        # Should find Bob as both outgoing (alice->bob) and incoming (bob->alice)
        assert len(neighbors) == 2

    @pytest.mark.asyncio
    async def test_cascade_delete(self, store):
        """Deleting a memory/entity also deletes its relations."""
        alice_id = await store.upsert_entity("Alice", entity_type="person")
        bob_id = await store.upsert_entity("Bob", entity_type="person")
        rel_id = await store.add_relation(alice_id, bob_id, relation_type="knows")

        # Store a memory with alice_id as its ID (for cascade delete test)
        # The delete method cascades to remove relations referencing the deleted ID
        edges_before = await store.query_edges(source_id=alice_id)
        assert len(edges_before) == 1

        # Use the delete method which also cascade-deletes relations
        await store.delete(alice_id)

        # Relations involving alice_id should be gone
        edges_after = await store.query_edges(source_id=alice_id)
        assert len(edges_after) == 0

    @pytest.mark.asyncio
    async def test_add_relation_returns_id(self, store):
        alice_id = await store.upsert_entity("Alice", entity_type="person")
        bob_id = await store.upsert_entity("Bob", entity_type="person")

        rel_id = await store.add_relation(alice_id, bob_id, relation_type="knows")
        assert isinstance(rel_id, str)
        assert len(rel_id) > 0


# ===========================================================================
# 6. UserProfile (identity/engine.py)
# ===========================================================================

from jarvis.identity.engine import UserProfile, IdentityEngine


class TestUserProfile:
    """Test UserProfile dataclass and IdentityEngine profile methods."""

    def test_profile_to_dict(self):
        p = UserProfile(
            user_id="user1",
            goals=["learn rust", "ship product"],
            knowledge_areas={"python": "expert", "rust": "beginner"},
            cognitive_style="analytical",
        )
        d = p.to_dict()
        assert d["user_id"] == "user1"
        assert "learn rust" in d["goals"]
        assert d["knowledge_areas"]["python"] == "expert"
        assert d["cognitive_style"] == "analytical"

    def test_profile_to_dict_excludes_empty(self):
        """to_dict should exclude falsy values."""
        p = UserProfile(user_id="u1")
        d = p.to_dict()
        assert "user_id" in d
        # Empty list/dict/string fields should be excluded
        assert "goals" not in d
        assert "notes" not in d

    def test_profile_from_dict(self):
        data = {
            "user_id": "user2",
            "goals": ["be productive"],
            "cognitive_style": "visual",
        }
        p = UserProfile.from_dict(data)
        assert p.user_id == "user2"
        assert p.goals == ["be productive"]
        assert p.cognitive_style == "visual"

    def test_profile_from_dict_ignores_unknown_fields(self):
        data = {
            "user_id": "u3",
            "goals": ["test"],
            "unknown_field": "should be ignored",
            "another_unknown": 42,
        }
        p = UserProfile.from_dict(data)
        assert p.user_id == "u3"
        assert p.goals == ["test"]
        assert not hasattr(p, "unknown_field")

    def test_save_and_load(self, tmp_path):
        engine = IdentityEngine(str(tmp_path), agent_name="jarvis")
        profile = UserProfile(
            user_id="roundtrip_user",
            goals=["ship v2"],
            knowledge_areas={"Go": "intermediate"},
            cognitive_style="creative",
            notes="prefers dark mode",
        )
        engine.save_profile(profile)

        loaded = engine.load_profile()
        assert loaded.user_id == "roundtrip_user"
        assert loaded.goals == ["ship v2"]
        assert loaded.knowledge_areas == {"Go": "intermediate"}
        assert loaded.cognitive_style == "creative"
        assert loaded.notes == "prefers dark mode"
        # save_profile sets updated_at
        assert loaded.updated_at != ""

    def test_load_default_when_missing(self, tmp_path):
        engine = IdentityEngine(str(tmp_path), agent_name="jarvis")
        profile = engine.load_profile()
        assert profile.user_id == "default"
        assert profile.goals == []

    def test_record_interaction(self, tmp_path):
        engine = IdentityEngine(str(tmp_path), agent_name="jarvis")
        # Save a baseline profile first
        engine.save_profile(UserProfile(user_id="interactive_user"))

        engine.record_interaction()

        loaded = engine.load_profile()
        assert loaded.last_interaction_at != ""
        # Verify ISO format timestamp
        assert "T" in loaded.last_interaction_at

    def test_profile_in_prompt(self, tmp_path):
        engine = IdentityEngine(str(tmp_path), agent_name="jarvis")

        # Write IDENTITY.md so the engine considers itself bootstrapped
        (tmp_path / "agents" / "jarvis" / "IDENTITY.md").write_text("I am JARVIS.")

        # Save a profile with goals and knowledge
        profile = UserProfile(
            user_id="prompt_user",
            goals=["learn AI", "build robots"],
            knowledge_areas={"ML": "intermediate", "robotics": "beginner"},
            cognitive_style="analytical",
            preferences={"verbosity": "concise"},
        )
        engine.save_profile(profile)

        prompt = engine.build_identity_prompt()
        # The prompt should include profile data
        assert "learn AI" in prompt
        assert "build robots" in prompt
        assert "ML" in prompt
        assert "analytical" in prompt
        assert "verbosity" in prompt

    def test_profile_not_in_prompt_when_empty(self, tmp_path):
        engine = IdentityEngine(str(tmp_path), agent_name="jarvis")
        # Write IDENTITY.md to mark as bootstrapped
        (tmp_path / "agents" / "jarvis" / "IDENTITY.md").write_text("I am JARVIS.")

        # Do NOT save any profile
        prompt = engine.build_identity_prompt()
        # Profile-specific keywords should not appear
        assert "User Goals:" not in prompt
        assert "Knowledge Areas:" not in prompt


# ===========================================================================
# 7. DB Server / Client (services/db_server.py, services/db_client.py)
# ===========================================================================

from jarvis.services.db_server import DBServer
from jarvis.services.db_client import DBClient


class TestDBServer:
    """Integration tests for DBServer + DBClient over a Unix socket."""

    @pytest_asyncio.fixture
    async def server_and_client(self, tmp_path):
        """Start a DBServer with a test DB and connect a client."""
        # Use /tmp to avoid AF_UNIX 104-char path limit on macOS
        import tempfile
        _tmpdir = tempfile.mkdtemp(prefix="jarvis_test_")
        sock_path = str(Path(_tmpdir) / "db.sock")
        db_path = str(tmp_path / "test.db")
        registry = {"test": db_path}

        server = DBServer(db_registry=registry, socket_path=sock_path)
        await server.start()

        client = DBClient(socket_path=sock_path)
        await client.connect()

        yield server, client

        await client.close()
        await server.stop()

    @pytest.mark.asyncio
    async def test_ping(self, server_and_client):
        server, client = server_and_client
        result = await client.ping()
        assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_execute_and_query(self, server_and_client):
        server, client = server_and_client

        # Create table
        await client.executescript(
            "test",
            "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT);",
        )

        # Insert row
        result = await client.execute(
            "test",
            "INSERT INTO users (id, name) VALUES (?, ?)",
            [1, "Alice"],
        )
        assert result["rowcount"] == 1

        # Query
        rows = await client.query("test", "SELECT * FROM users WHERE id = ?", [1])
        assert len(rows) == 1
        assert rows[0][0] == 1
        assert rows[0][1] == "Alice"

    @pytest.mark.asyncio
    async def test_query_dicts(self, server_and_client):
        server, client = server_and_client

        await client.executescript(
            "test",
            "CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, label TEXT);",
        )
        await client.execute(
            "test",
            "INSERT INTO items (id, label) VALUES (?, ?)",
            [10, "widget"],
        )

        rows = await client.query_dicts(
            "test", "SELECT * FROM items WHERE id = ?", [10]
        )
        assert len(rows) == 1
        assert rows[0]["id"] == 10
        assert rows[0]["label"] == "widget"

    @pytest.mark.asyncio
    async def test_multiple_inserts_and_query(self, server_and_client):
        server, client = server_and_client

        await client.executescript(
            "test",
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, val TEXT);",
        )
        for i in range(5):
            await client.execute(
                "test",
                "INSERT INTO kv (key, val) VALUES (?, ?)",
                [f"k{i}", f"v{i}"],
            )

        rows = await client.query("test", "SELECT * FROM kv ORDER BY key")
        assert len(rows) == 5

    @pytest.mark.asyncio
    async def test_executescript(self, server_and_client):
        server, client = server_and_client

        result = await client.executescript(
            "test",
            """
            CREATE TABLE IF NOT EXISTS t1 (a TEXT);
            CREATE TABLE IF NOT EXISTS t2 (b TEXT);
            """,
        )
        assert result["ok"] is True

        # Verify both tables exist by inserting into them
        await client.execute("test", "INSERT INTO t1 (a) VALUES (?)", ["x"])
        await client.execute("test", "INSERT INTO t2 (b) VALUES (?)", ["y"])

        r1 = await client.query("test", "SELECT * FROM t1")
        r2 = await client.query("test", "SELECT * FROM t2")
        assert len(r1) == 1
        assert len(r2) == 1


# ===========================================================================
# 8. Agent SDK Adapter (tools/agent_sdk_adapter.py)
# ===========================================================================

from jarvis.tools.agent_sdk_adapter import (
    JarvisToolAdapter,
    _json_type_to_python,
    _JSON_TYPE_MAP,
)
from jarvis.tools.registry import ToolDefinition, ToolRegistry


class TestAgentSDKAdapter:
    """Test the JarvisToolAdapter wrapper building and type mapping."""

    def _make_registry(self) -> ToolRegistry:
        """Build a minimal ToolRegistry with a single test tool."""
        reg = ToolRegistry()
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool for unit tests.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "count": {"type": "integer", "description": "Max results"},
                    "verbose": {"type": "boolean", "description": "Verbose output"},
                },
                "required": ["query"],
            },
        )
        reg.register(tool)
        return reg

    def test_build_wrapper_signature(self):
        """Wrapper function has correct __name__, __doc__, __annotations__."""
        reg = self._make_registry()
        mock_dispatcher = MagicMock()
        adapter = JarvisToolAdapter(mock_dispatcher, registry=reg)

        tools = adapter.get_sdk_tools()
        assert len(tools) == 1

        wrapper = tools[0]
        assert wrapper.__name__ == "test_tool"
        assert "A test tool" in wrapper.__doc__
        assert wrapper.__annotations__["query"] is str
        assert wrapper.__annotations__["return"] is str
        # count is optional (not in required)
        # For required params, the type should be int directly
        # For optional params, it should be int | None or similar
        assert "count" in wrapper.__annotations__

    def test_build_wrapper_qualname(self):
        reg = self._make_registry()
        mock_dispatcher = MagicMock()
        adapter = JarvisToolAdapter(mock_dispatcher, registry=reg)
        tools = adapter.get_sdk_tools()
        assert tools[0].__qualname__ == "JarvisToolAdapter.test_tool"

    def test_build_wrapper_caches(self):
        """get_sdk_tools caches the result on second call."""
        reg = self._make_registry()
        mock_dispatcher = MagicMock()
        adapter = JarvisToolAdapter(mock_dispatcher, registry=reg)

        first = adapter.get_sdk_tools()
        second = adapter.get_sdk_tools()
        assert first is second

    def test_json_type_mapping(self):
        assert _json_type_to_python("string") is str
        assert _json_type_to_python("integer") is int
        assert _json_type_to_python("number") is float
        assert _json_type_to_python("boolean") is bool
        assert _json_type_to_python("array") is list
        assert _json_type_to_python("object") is dict

    def test_json_type_mapping_unknown_fallback(self):
        """Unknown type names fall back to str."""
        assert _json_type_to_python("foobar") is str
        assert _json_type_to_python("") is str

    def test_json_type_map_dict(self):
        """Verify the _JSON_TYPE_MAP constant has all expected entries."""
        assert len(_JSON_TYPE_MAP) == 6
        for key in ("string", "integer", "number", "boolean", "array", "object"):
            assert key in _JSON_TYPE_MAP

    def test_empty_registry(self):
        """Adapter with empty registry produces no tools."""
        reg = ToolRegistry()
        mock_dispatcher = MagicMock()
        adapter = JarvisToolAdapter(mock_dispatcher, registry=reg)
        tools = adapter.get_sdk_tools()
        assert tools == []

    def test_wrapper_has_jarvis_tool_def(self):
        """Each wrapper has the original ToolDefinition attached."""
        reg = self._make_registry()
        mock_dispatcher = MagicMock()
        adapter = JarvisToolAdapter(mock_dispatcher, registry=reg)
        tools = adapter.get_sdk_tools()
        assert hasattr(tools[0], "_jarvis_tool_def")
        assert tools[0]._jarvis_tool_def.name == "test_tool"

    @pytest.mark.asyncio
    async def test_wrapper_calls_dispatcher(self):
        """Calling the wrapper delegates to dispatcher.dispatch()."""
        reg = self._make_registry()
        mock_dispatcher = MagicMock()

        async def mock_dispatch(tool_name, args, **kwargs):
            return f"result for {tool_name}"

        mock_dispatcher.dispatch = mock_dispatch
        adapter = JarvisToolAdapter(mock_dispatcher, registry=reg)
        tools = adapter.get_sdk_tools()

        result = await tools[0](query="hello")
        assert result == "result for test_tool"

    @pytest.mark.asyncio
    async def test_wrapper_strips_none_values(self):
        """None values for optional params are stripped before dispatch."""
        reg = self._make_registry()
        mock_dispatcher = MagicMock()
        captured_args = {}

        async def mock_dispatch(tool_name, args, **kwargs):
            captured_args.update(args)
            return "ok"

        mock_dispatcher.dispatch = mock_dispatch
        adapter = JarvisToolAdapter(mock_dispatcher, registry=reg)
        tools = adapter.get_sdk_tools()

        await tools[0](query="hi", count=None, verbose=None)
        assert "query" in captured_args
        assert "count" not in captured_args
        assert "verbose" not in captured_args
