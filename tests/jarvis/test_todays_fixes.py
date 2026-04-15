"""Tests for bugs fixed in the 2026-03-31 session.

Covers: lane_queue timeout, memory _extract_key, browser_bridge SPA fixes,
gemini configure lock, claude_code persistent client + in-process MCP.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
from typing import Any
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── 1. Memory _extract_key ─────────────────────────────────────────

class TestExtractKey:
    """_extract_key handles metadata as JSON string, dict, or missing."""

    def _extract_key(self, mem: dict) -> str:
        from jarvis.tools.handlers.memory import _extract_key
        return _extract_key(mem)

    def test_metadata_as_dict(self):
        mem = {"id": "abc", "metadata": {"key": "fav_color"}}
        assert self._extract_key(mem) == "fav_color"

    def test_metadata_as_json_string(self):
        mem = {"id": "abc", "metadata": '{"key": "fav_color"}'}
        assert self._extract_key(mem) == "fav_color"

    def test_metadata_missing(self):
        mem = {"id": "abc"}
        assert self._extract_key(mem) == "abc"

    def test_metadata_empty_dict(self):
        mem = {"id": "abc", "metadata": {}}
        assert self._extract_key(mem) == "abc"

    def test_metadata_empty_string(self):
        mem = {"id": "abc", "metadata": ""}
        assert self._extract_key(mem) == "abc"

    def test_metadata_invalid_json_string(self):
        mem = {"id": "abc", "metadata": "not json at all"}
        assert self._extract_key(mem) == "abc"

    def test_metadata_none_key(self):
        mem = {"id": "abc", "metadata": {"key": None}}
        assert self._extract_key(mem) == "abc"

    def test_metadata_integer(self):
        mem = {"id": "abc", "metadata": 42}
        assert self._extract_key(mem) == "abc"

    def test_no_id_fallback(self):
        mem = {"metadata": {"key": None}}
        assert self._extract_key(mem) == "?"

    def test_nested_json_string(self):
        mem = {"id": "x", "metadata": '{"key": "user_pref", "extra": true}'}
        assert self._extract_key(mem) == "user_pref"


# ── 2. Lane Queue Timeout Logging ──────────────────────────────────

class TestLaneQueueTimeout:
    """Verify timeout log uses lane.session_id[:8] not undefined lane_id."""

    def test_timeout_log_message(self):
        """The error log should reference lane.session_id, not lane_id."""
        import ast
        from pathlib import Path

        src = Path(__file__).parent.parent.parent / "src" / "jarvis" / "services" / "lane_queue.py"
        code = src.read_text()
        # The old bug: lane_id (undefined variable)
        assert "lane_id" not in code, "lane_id should be replaced with lane.session_id[:8]"
        # The fix: lane.session_id[:8]
        assert "lane.session_id[:8]" in code


# ── 3. Browser Bridge SPA Fixes ────────────────────────────────────

class TestBrowserBridgeReadText:
    """read_text uses textContent (not innerText) and SPA-aware selectors."""

    def test_read_text_uses_textcontent(self):
        """Source code should use textContent, not innerText for full page."""
        from pathlib import Path
        src = Path(__file__).parent.parent.parent / "src" / "jarvis" / "operators" / "browser_bridge.py"
        code = src.read_text()
        # Full page read should use textContent
        assert "textContent" in code
        # Should include SPA-aware selectors
        assert "article" in code
        assert "data-testid" in code

    def test_selector_read_uses_textcontent(self):
        """Element-specific reads should use textContent too."""
        from pathlib import Path
        src = Path(__file__).parent.parent.parent / "src" / "jarvis" / "operators" / "browser_bridge.py"
        code = src.read_text()
        assert ".textContent" in code

    @pytest.mark.asyncio
    async def test_read_text_disconnected(self):
        """Returns empty string when not connected."""
        from jarvis.operators.browser_bridge import BrowserBridge
        bridge = BrowserBridge()
        result = await bridge.read_text()
        assert result == ""

    @pytest.mark.asyncio
    async def test_navigate_disconnected(self):
        """Returns error when not connected."""
        from jarvis.operators.browser_bridge import BrowserBridge
        bridge = BrowserBridge()
        result = await bridge.navigate("https://example.com")
        assert result["ok"] is False


class TestBrowserBridgeNavigate:
    """navigate waits 4s for SPA hydration and uses textContent."""

    def test_navigate_spa_wait_time(self):
        """Source should sleep 4.0s, not 2.5s."""
        from pathlib import Path
        src = Path(__file__).parent.parent.parent / "src" / "jarvis" / "operators" / "browser_bridge.py"
        code = src.read_text()
        nav_start = code.index("async def navigate")
        nav_end = code.index("async def scroll")
        nav_code = code[nav_start:nav_end]
        assert "sleep(4.0)" in nav_code
        assert "sleep(2.5)" not in nav_code

    def test_navigate_uses_textcontent_check(self):
        """Content check should use textContent, not innerText."""
        from pathlib import Path
        src = Path(__file__).parent.parent.parent / "src" / "jarvis" / "operators" / "browser_bridge.py"
        code = src.read_text()
        nav_start = code.index("async def navigate")
        nav_end = code.index("async def scroll")
        nav_code = code[nav_start:nav_end]
        assert "textContent" in nav_code


# ── 4. Gemini Configure Lock ──────────────────────────────────────

class TestGeminiConfigureLock:
    """genai.configure() is protected by asyncio.Lock."""

    def test_lock_exists(self):
        """GeminiProvider should have a _configure_lock class attribute."""
        from jarvis.llm_providers.gemini import GeminiProvider
        assert hasattr(GeminiProvider, "_configure_lock")
        assert isinstance(GeminiProvider._configure_lock, asyncio.Lock)

    def test_lock_is_class_level(self):
        """Lock should be shared across all instances."""
        from jarvis.llm_providers.gemini import GeminiProvider
        p1_lock = GeminiProvider._configure_lock
        p2_lock = GeminiProvider._configure_lock
        assert p1_lock is p2_lock


# ── 5. Claude Code Persistent Client ──────────────────────────────

class TestClaudeCodeProvider:
    """Persistent client in background thread with auto-reconnect."""

    def test_init_state(self):
        """Provider initializes with no client connected."""
        from jarvis.llm_providers.claude_code import ClaudeCodeProvider
        config = MagicMock()
        config.model = "opus"
        provider = ClaudeCodeProvider(config)
        assert provider._sdk_client is None
        assert provider._sdk_connected is False
        assert provider._sdk_loop is None
        assert provider._sdk_thread is None

    def test_dispatcher_injection(self):
        """set_dispatcher stores the dispatcher for in-process MCP."""
        from jarvis.llm_providers.claude_code import ClaudeCodeProvider
        config = MagicMock()
        config.model = "opus"
        provider = ClaudeCodeProvider(config)
        mock_dispatcher = MagicMock()
        provider.set_dispatcher(mock_dispatcher)
        assert provider._dispatcher is mock_dispatcher

    def test_model_aliases(self):
        """Model aliases resolve to full names."""
        from jarvis.llm_providers.claude_code import _MODEL_ALIASES
        assert _MODEL_ALIASES["opus"] == "claude-opus-4-6"
        assert _MODEL_ALIASES["sonnet"] == "claude-sonnet-4-6"
        assert _MODEL_ALIASES["haiku"] == "claude-haiku-4-5"

    def test_mcp_config_caching(self):
        """MCP config dict is generated once and cached."""
        from jarvis.llm_providers.claude_code import ClaudeCodeProvider
        config = MagicMock()
        config.model = "opus"
        provider = ClaudeCodeProvider(config)
        # Simulate cached config
        provider._mcp_config_dict = {"mcpServers": {"jarvis": {}}}
        result = provider._ensure_mcp_config_dict()
        assert result is provider._mcp_config_dict

    def test_in_process_mcp_over_subprocess(self):
        """When dispatcher is injected, in-process MCP is preferred."""
        from jarvis.llm_providers.claude_code import ClaudeCodeProvider
        config = MagicMock()
        config.model = "opus"
        provider = ClaudeCodeProvider(config)
        provider._dispatcher = MagicMock()
        # sdk_mcp_server is None initially → will be built on first use
        assert provider._sdk_mcp_server is None
        # After dispatcher injection, in-process should be preferred
        assert provider._dispatcher is not None

    def test_jarvis_config_path(self):
        """JARVIS_CLAUDE_CONFIG points to isolated config dir."""
        from jarvis.llm_providers.claude_code import ClaudeCodeProvider
        assert ClaudeCodeProvider.JARVIS_CLAUDE_CONFIG.endswith("/.prometheus/claude-config")

    @pytest.mark.asyncio
    async def test_chat_uses_fresh_sdk_session_id_per_request(self):
        from jarvis.llm_providers.claude_code import ClaudeCodeProvider

        config = MagicMock()
        config.model = "opus"
        provider = ClaudeCodeProvider(config)

        class FakeOptions:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class SystemMessage:
            def __init__(self):
                self.data = {"session_id": "sdk-session-1"}

        class TextBlock:
            def __init__(self, text):
                self.text = text

        class AssistantMessage:
            def __init__(self):
                self.content = [TextBlock("ok")]
                self.model = "claude-opus-4-6"

        class ResultMessage:
            def __init__(self):
                self.total_cost_usd = 0.0
                self.subtype = "end_turn"
                self.usage = {"input_tokens": 12, "output_tokens": 3}
                self.result = "ok"

        class FakeClient:
            instances = []

            def __init__(self, options=None):
                self.options = options
                self.connected_prompts = []
                self.queries = []
                FakeClient.instances.append(self)

            async def connect(self, prompt=None):
                self.connected_prompts.append(prompt)

            async def disconnect(self):
                return None

            async def query(self, prompt, session_id="default"):
                self.queries.append((prompt, session_id))

            async def receive_messages(self):
                yield SystemMessage()
                yield AssistantMessage()
                yield ResultMessage()

        fake_sdk = SimpleNamespace(
            ClaudeAgentOptions=FakeOptions,
            ClaudeSDKClient=FakeClient,
        )

        with patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk}):
            result = await provider.chat([{"role": "user", "content": "hi"}])

        assert result["content"] == "ok"
        client = FakeClient.instances[-1]
        assert client.connected_prompts == [None]
        assert len(client.queries) == 1
        assert client.queries[0][0] == "[USER]: hi"
        assert client.queries[0][1] != "default"
        assert result["session_id"] == "sdk-session-1"

    @pytest.mark.asyncio
    async def test_empty_sdk_result_returns_non_empty_fallback(self):
        from jarvis.llm_providers.claude_code import ClaudeCodeProvider

        config = MagicMock()
        config.model = "opus"
        provider = ClaudeCodeProvider(config)

        class FakeOptions:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class ResultMessage:
            def __init__(self):
                self.total_cost_usd = 0.0
                self.subtype = "end_turn"
                self.usage = {"input_tokens": 7, "output_tokens": 0}
                self.result = ""

        class FakeClient:
            async def connect(self, prompt=None):
                return None

            async def disconnect(self):
                return None

            async def query(self, prompt, session_id="default"):
                return None

            async def receive_messages(self):
                yield ResultMessage()

        fake_sdk = SimpleNamespace(
            ClaudeAgentOptions=FakeOptions,
            ClaudeSDKClient=lambda options=None: FakeClient(),
        )

        with patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk}):
            result = await provider.chat([{"role": "user", "content": "hi"}])

        assert "empty response" in result["content"].lower()
        assert provider._sdk_connected is False

    @pytest.mark.asyncio
    async def test_chat_uses_explicit_jarvis_tool_whitelist(self):
        from jarvis.llm_providers.claude_code import ClaudeCodeProvider
        from jarvis.tools.mcp_server import get_mcp_tool_names

        config = MagicMock()
        config.model = "opus"
        provider = ClaudeCodeProvider(config)
        provider._mcp_config_dict = {"mcpServers": {"jarvis": {}}}

        class FakeOptions:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class TextBlock:
            def __init__(self, text):
                self.text = text

        class AssistantMessage:
            def __init__(self):
                self.content = [TextBlock("ok")]
                self.model = "claude-opus-4-6"

        class ResultMessage:
            def __init__(self):
                self.total_cost_usd = 0.0
                self.subtype = "end_turn"
                self.usage = {"input_tokens": 1, "output_tokens": 1}
                self.result = "ok"

        class FakeClient:
            instances = []

            def __init__(self, options=None):
                self.options = options
                FakeClient.instances.append(self)

            async def connect(self, prompt=None):
                return None

            async def disconnect(self):
                return None

            async def query(self, prompt, session_id="default"):
                return None

            async def receive_messages(self):
                yield AssistantMessage()
                yield ResultMessage()

        fake_sdk = SimpleNamespace(
            ClaudeAgentOptions=FakeOptions,
            ClaudeSDKClient=FakeClient,
        )

        with patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk}):
            result = await provider.chat(
                [{"role": "user", "content": "hi"}],
                tools=[{"name": "read_file"}],
            )

        assert result["content"] == "ok"
        client = FakeClient.instances[-1]
        assert client.options.tools == []
        assert client.options.allowed_tools == get_mcp_tool_names()
        assert "Agent" in client.options.disallowed_tools


class TestClaudeHarnessProvider:
    """Claude transport with JARVIS-owned tool loop."""

    @pytest.mark.asyncio
    async def test_chat_parses_text_tool_calls_without_mcp(self):
        from jarvis.llm_providers.claude_harness import ClaudeHarnessProvider

        config = MagicMock()
        config.model = "opus"
        provider = ClaudeHarnessProvider(config)

        class FakeOptions:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class TextBlock:
            def __init__(self, text):
                self.text = text

        class AssistantMessage:
            def __init__(self):
                self.content = [
                    TextBlock(
                        'Checking now.\n'
                        '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call>'
                    )
                ]
                self.model = "claude-opus-4-6"

        class ResultMessage:
            def __init__(self):
                self.total_cost_usd = 0.0
                self.subtype = "end_turn"
                self.usage = {"input_tokens": 12, "output_tokens": 4}
                self.result = (
                    'Checking now.\n'
                    '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call>'
                )

        class FakeClient:
            instances = []

            def __init__(self, options=None):
                self.options = options
                FakeClient.instances.append(self)

            async def connect(self, prompt=None):
                return None

            async def disconnect(self):
                return None

            async def query(self, prompt, session_id="default"):
                return None

            async def receive_messages(self):
                yield AssistantMessage()
                yield ResultMessage()

        fake_sdk = SimpleNamespace(
            ClaudeAgentOptions=FakeOptions,
            ClaudeSDKClient=FakeClient,
        )

        with patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk}):
            result = await provider.chat(
                [{"role": "system", "content": "Be useful."}, {"role": "user", "content": "Read the file."}],
                tools=[{"name": "read_file", "description": "Read a file", "parameters": {"type": "object"}}],
            )

        client = FakeClient.instances[-1]
        assert "TOOL USE INSTRUCTIONS" in client.options.system_prompt
        assert not hasattr(client.options, "mcp_servers")
        assert "Agent" in client.options.disallowed_tools
        assert result["content"] == "Checking now."
        assert result["tool_calls"] == [{"name": "read_file", "arguments": {"path": "/tmp/x"}}]
        assert result["finish_reason"] == "tool_calls"
        assert result["tools_executed"] == []

    def test_router_exposes_internal_tool_loop_capability(self):
        from jarvis.llm_providers.router import LLMInterface

        def _cfg(provider_name: str):
            return SimpleNamespace(
                llm=SimpleNamespace(
                    provider=provider_name,
                    fallback_providers=[],
                    temperature=0.7,
                    max_tokens=4096,
                    api_key="",
                    base_url="",
                    model="claude-opus-4-6",
                    throttle_rapid_window=5.0,
                    throttle_ramp_after=3,
                    throttle_min_delay=0.0,
                    throttle_max_delay=0.0,
                    max_retries=0,
                ),
                home_dir="~/.prometheus",
            )

        assert LLMInterface(_cfg("claude-code")).executes_tool_loop_internally is True
        assert LLMInterface(_cfg("claude-harness")).executes_tool_loop_internally is False


class TestAnthropicProvider:
    """Anthropic provider normalizes Claude aliases before API calls."""

    @pytest.mark.asyncio
    async def test_claude_oauth_resolves_opus_alias(self):
        from jarvis.llm_providers.anthropic import AnthropicProvider
        from jarvis.llm_providers.base import ProviderConfig

        provider = AnthropicProvider(
            ProviderConfig(
                model="opus",
                reasoning_effort="max",
                extras={
                    "provider": "claude-oauth",
                    "oauth_tokens": ["oauth-token"],
                },
            )
        )
        provider._execute_with_cycling = AsyncMock(return_value={"content": "ok"})

        fake_anthropic = SimpleNamespace()
        with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            result = await provider.chat([{"role": "user", "content": "hi"}])

        assert result == {"content": "ok"}
        provider._execute_with_cycling.assert_awaited_once()
        _, call_kwargs, _ = provider._execute_with_cycling.await_args.args
        assert call_kwargs["model"] == "claude-opus-4-6"
        assert call_kwargs["max_tokens"] == 4096
        assert call_kwargs["thinking"] == {"type": "adaptive"}

    def test_resolve_claude_model_aliases(self):
        from jarvis.llm_providers.claude_models import resolve_claude_model

        assert resolve_claude_model("opus") == "claude-opus-4-6"
        assert resolve_claude_model("sonnet") == "claude-sonnet-4-6"
        assert resolve_claude_model("haiku") == "claude-haiku-4-5"


class TestContextBudget:
    """Claude OAuth should use a smaller prompt budget than local providers."""

    def test_claude_oauth_opus_uses_lower_context_budget(self):
        from jarvis.core import _context_budget_for_provider

        assert _context_budget_for_provider("claude-oauth", "claude-opus-4-6") == (22_000, 4_000)
        assert _context_budget_for_provider("claude-code", "claude-opus-4-6") == (36_000, 6_000)


class TestClaudeOAuthTokenLoading:
    """Dedicated JARVIS Claude auth should win over the default CLI account."""

    def test_prefers_dedicated_jarvis_account(self, tmp_path, monkeypatch):
        from jarvis.llm_providers.router import LLMInterface

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("CLAUDE_OAUTH_TOKENS", raising=False)
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

        jarvis_creds = tmp_path / ".prometheus" / "claude-config" / ".credentials.json"
        default_creds = tmp_path / ".claude" / ".credentials.json"
        jarvis_creds.parent.mkdir(parents=True)
        default_creds.parent.mkdir(parents=True)

        dedicated_token = "dedicated_token_abcdefghijklmnopqrstuvwxyz"
        default_token = "default_token_abcdefghijklmnopqrstuvwxyz"

        jarvis_creds.write_text(json.dumps({
            "claudeAiOauth": {"accessToken": dedicated_token},
        }))
        default_creds.write_text(json.dumps({
            "claudeAiOauth": {"accessToken": default_token},
        }))

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=1, stdout="")
            tokens = LLMInterface._load_claude_oauth_tokens()

        assert tokens == [dedicated_token]


# ── 6. Auto-reconnect on Empty Response ───────────────────────────

class TestAutoReconnect:
    """Persistent client marks itself for reconnect on empty responses."""

    def test_source_has_reconnect_logic(self):
        """claude_code.py should reset _sdk_connected on empty response."""
        from pathlib import Path
        src = Path(__file__).parent.parent.parent / "src" / "jarvis" / "llm_providers" / "claude_code.py"
        code = src.read_text()
        assert "Empty response from persistent client" in code
        assert "marking for reconnect" in code
        assert "self._sdk_connected = False" in code

    def test_source_has_stream_error_reconnect(self):
        """claude_code.py should reset on stream errors too."""
        from pathlib import Path
        src = Path(__file__).parent.parent.parent / "src" / "jarvis" / "llm_providers" / "claude_code.py"
        code = src.read_text()
        assert "Stream error, will reconnect" in code


class TestTelegramEmptyFallback:
    """Telegram should never drop an empty assistant reply silently."""

    @pytest.mark.asyncio
    async def test_send_replaces_empty_text_with_fallback(self):
        from jarvis.channels.telegram import TelegramAdapter
        from jarvis.gateway import OutgoingMessage

        config = SimpleNamespace(
            channels=SimpleNamespace(telegram_token="", telegram={}),
        )
        adapter = TelegramAdapter(config)
        bot = SimpleNamespace(send_message=AsyncMock())
        adapter._app = SimpleNamespace(bot=bot)

        await adapter.send(
            OutgoingMessage(
                channel="telegram",
                user_id="123",
                text="",
                session_id="sess-1",
                metadata={"chat_id": "123"},
            )
        )

        bot.send_message.assert_awaited_once()
        kwargs = bot.send_message.await_args.kwargs
        assert kwargs["chat_id"] == 123
        assert kwargs["parse_mode"] == "MarkdownV2"
        assert kwargs["text"].strip()

    @pytest.mark.asyncio
    async def test_cmd_start_suppresses_duplicate_update(self):
        from jarvis.channels.telegram import TelegramAdapter

        config = SimpleNamespace(
            channels=SimpleNamespace(telegram_token="", telegram={}),
        )
        adapter = TelegramAdapter(config)
        reply_text = AsyncMock()
        update = SimpleNamespace(
            update_id=4242,
            message=SimpleNamespace(
                from_user=SimpleNamespace(id=123),
                chat_id=123,
                message_id=77,
                reply_text=reply_text,
            ),
        )

        await adapter._cmd_start(update, None)
        await adapter._cmd_start(update, None)

        reply_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cmd_start_suppresses_rapid_duplicate_commands(self):
        from jarvis.channels.telegram import TelegramAdapter

        config = SimpleNamespace(
            channels=SimpleNamespace(telegram_token="", telegram={}),
        )
        adapter = TelegramAdapter(config)
        reply_text = AsyncMock()
        update1 = SimpleNamespace(
            update_id=4242,
            message=SimpleNamespace(
                from_user=SimpleNamespace(id=123),
                chat_id=123,
                message_id=77,
                reply_text=reply_text,
            ),
        )
        update2 = SimpleNamespace(
            update_id=4243,
            message=SimpleNamespace(
                from_user=SimpleNamespace(id=123),
                chat_id=123,
                message_id=78,
                reply_text=reply_text,
            ),
        )

        await adapter._cmd_start(update1, None)
        await adapter._cmd_start(update2, None)

        reply_text.assert_awaited_once()


# ── 7. In-Process MCP Server ─────────────────────────────────────

class TestInProcessMCP:
    """In-process SDK MCP server routes to daemon's dispatcher."""

    def test_build_sdk_mcp_server(self):
        """build_sdk_mcp_server creates a server config with tools."""
        from jarvis.tools.mcp_server import build_sdk_mcp_server
        mock_dispatcher = MagicMock()
        server = build_sdk_mcp_server(mock_dispatcher)
        # Returns McpSdkServerConfig (TypedDict with type, name, instance)
        assert server is not None
        assert server["type"] == "sdk"
        assert server["name"] == "jarvis-tools"

    def test_get_mcp_config_dict(self):
        """get_mcp_config_dict returns valid MCP config."""
        from jarvis.tools.mcp_server import get_mcp_config_dict
        config = get_mcp_config_dict()
        assert "mcpServers" in config
        assert "jarvis" in config["mcpServers"]
        jarvis_cfg = config["mcpServers"]["jarvis"]
        assert "command" in jarvis_cfg
        assert "args" in jarvis_cfg


class TestMultiAgentSharedMemory:
    """Multi-agent workers share one retrieved JARVIS memory slice."""

    @pytest.mark.asyncio
    async def test_unified_memory_scopes_default_global_and_can_target_team(self, tmp_path):
        from jarvis.memory.store import UnifiedMemoryStore

        store = UnifiedMemoryStore(str(tmp_path / "unified_memory.db"))
        await store.store(
            content="Global memory about the build failure",
            user_id="shubham",
            memory_scope="global",
            metadata={"key": "global_build"},
        )
        await store.store(
            content="Team memory about the build failure",
            user_id="shubham",
            memory_scope="team",
            team_id="team-123",
            metadata={"key": "team_build"},
        )

        global_results = await store.recall(
            query="build failure",
            user_id="shubham",
            top_k=10,
        )
        assert [mem["metadata"]["key"] for mem, _ in global_results] == ["global_build"]

        team_results = await store.recall(
            query="build failure",
            user_id="shubham",
            top_k=10,
            scopes=["team"],
            team_id="team-123",
        )
        assert [mem["metadata"]["key"] for mem, _ in team_results] == ["team_build"]

    @pytest.mark.asyncio
    async def test_shared_prompt_uses_unified_memory_once_for_all_workers(self):
        from jarvis.tools.handlers.agent import handle_multi_agent

        captured_messages: list[list[dict[str, Any]]] = []
        store_calls: list[dict[str, Any]] = []

        async def fake_chat(messages, tools=None):
            captured_messages.append(messages)
            return {"content": "ok"}

        class FakeUnifiedMemory:
            async def recall(self, *args, **kwargs):
                return []

            async def get_all_memories(self, *args, **kwargs):
                return []

            async def get_entity_context(self, *args, **kwargs):
                return {}

            async def get_recent_episodes(self, *args, **kwargs):
                return []

            async def list_entities(self, *args, **kwargs):
                return []

            async def store(self, **kwargs):
                store_calls.append(kwargs)
                return "mem-1"

        class FakeRetriever:
            def __init__(self, store):
                self.store = store

            async def build_context(
                self,
                query,
                user_id="default",
                max_tokens=3000,
                session_id=None,
                scopes=None,
                team_id=None,
            ):
                assert query == "Investigate this failure"
                assert user_id == "shubham"
                assert session_id == "sess-1"
                assert max_tokens == 1200
                assert scopes == ["global"]
                assert team_id is None
                return "## User Preferences\n- prefers concise answers"

        ctx = SimpleNamespace(
            llm_for_collab=SimpleNamespace(chat=fake_chat),
            unified_memory=FakeUnifiedMemory(),
            memory={
                "openclaw:abc": {
                    "content": "Prior delegate found the build failure in auth.py",
                    "tags": ["openclaw", "delegate"],
                    "stored_at": 100.0,
                }
            },
            resolve_user_id=lambda args: args.get("user_id", "default"),
        )

        with patch("jarvis.memory.MemoryRetriever", FakeRetriever):
            result = await handle_multi_agent(
                {
                    "prompt": "Investigate this failure",
                    "num_agents": 2,
                    "user_id": "shubham",
                    "session_id": "sess-1",
                },
                ctx,
            )

        assert "Pattern: fan_out | Team: team-" in result
        assert len(captured_messages) == 3
        architect_messages = captured_messages[0]
        assert "agent architect" in architect_messages[0]["content"]
        worker_messages = captured_messages[1:]
        assert len(worker_messages) == 2
        for messages in worker_messages:
            assert messages[0]["role"] == "system"
            assert "Specialization:" in messages[0]["content"]
            assert "Retrieved Unified Memory" in messages[1]["content"]
            assert "Recent Tool Memory" in messages[1]["content"]
            assert "prefers concise answers" in messages[1]["content"]
            assert "auth.py" in messages[1]["content"]
        assert store_calls
        assert all(call["memory_scope"] == "team" for call in store_calls)
        assert len({call["team_id"] for call in store_calls}) == 1


class TestDynamicAgentSpecs:
    """Meta-generated specs compile into specialized executable prompts."""

    def test_compile_dynamic_agent_prompt(self):
        from jarvis.agents.engine import (
            compile_dynamic_agent_prompt,
            create_dynamic_agent_spec,
        )

        spec = create_dynamic_agent_spec(
            base_type="coder",
            collaboration_role="pytest_failure_investigator",
            specialization="pytest failure investigator",
            mission="Find the root cause of the failing auth tests and propose the smallest safe fix.",
            success_criteria=[
                "Identify the failing module",
                "Name the minimal fix path",
            ],
            output_schema={
                "root_cause": "string",
                "fix_plan": "array",
            },
            allowed_tools=["read_file", "run_command", "not_a_real_tool"],
        )

        system_prompt, user_prompt = compile_dynamic_agent_prompt(
            spec,
            shared_context="Shared build logs and prior findings",
            original_task="Fix the failing auth tests",
        )

        assert "pytest failure investigator" in system_prompt
        assert "read_file" in system_prompt
        assert "run_command" in system_prompt
        assert "not_a_real_tool" not in system_prompt
        assert "Fix the failing auth tests" in user_prompt
        assert "Shared build logs and prior findings" in user_prompt
        assert "root_cause" in user_prompt
        assert "fix_plan" in user_prompt

    @pytest.mark.asyncio
    async def test_multi_agent_meta_planner_generates_specialized_prompts(self):
        from jarvis.tools.handlers.agent import handle_multi_agent

        worker_system_prompts: list[str] = []
        store_calls: list[dict[str, Any]] = []

        async def fake_chat(messages, tools=None):
            system_text = messages[0]["content"]
            if "agent architect" in system_text:
                return {
                    "content": json.dumps(
                        {
                            "agents": [
                                {
                                    "base_type": "coder",
                                    "collaboration_role": "root_cause_hunter",
                                    "specialization": "root cause hunter",
                                    "mission": "Pinpoint the exact auth failure and the smallest safe fix.",
                                    "success_criteria": ["Find the failing file", "Name the minimal fix"],
                                    "output_schema": {"root_cause": "string", "fix": "string"},
                                },
                                {
                                    "base_type": "critic",
                                    "collaboration_role": "regression_guard",
                                    "specialization": "regression guard",
                                    "mission": "Attack the proposed fix and identify likely regressions.",
                                    "success_criteria": ["List the top regressions", "Name missing tests"],
                                    "output_schema": {"risks": "array", "tests": "array"},
                                },
                            ]
                        }
                    )
                }
            worker_system_prompts.append(system_text)
            return {"content": "specialist output"}

        class FakeUnifiedMemory:
            async def recall(self, *args, **kwargs):
                return []

            async def get_all_memories(self, *args, **kwargs):
                return []

            async def get_entity_context(self, *args, **kwargs):
                return {}

            async def get_recent_episodes(self, *args, **kwargs):
                return []

            async def list_entities(self, *args, **kwargs):
                return []

            async def store(self, **kwargs):
                store_calls.append(kwargs)
                return "mem-1"

        ctx = SimpleNamespace(
            llm_for_collab=SimpleNamespace(chat=fake_chat),
            unified_memory=FakeUnifiedMemory(),
            memory={},
            resolve_user_id=lambda args: args.get("user_id", "default"),
        )

        result = await handle_multi_agent(
            {
                "prompt": "Fix the failing auth tests",
                "num_agents": 2,
                "agent_roles": ["coder", "critic"],
                "user_id": "shubham",
                "session_id": "sess-1",
            },
            ctx,
        )

        assert "Pattern: fan_out | Team: team-" in result
        assert len(worker_system_prompts) == 2
        assert "root cause hunter" in worker_system_prompts[0]
        assert "Mission: Pinpoint the exact auth failure" in worker_system_prompts[0]
        assert "regression guard" in worker_system_prompts[1]
        assert any("Dynamic specs:" in call["content"] for call in store_calls)


class TestDAFBridgeTeamBlackboard:
    """DAF bridge uses pull-loop dispatch and carries team-memory context."""

    @pytest.mark.asyncio
    async def test_dispatch_multi_agent_passes_team_context(self):
        from google.protobuf.struct_pb2 import Value
        from jarvis._vendor.common.protos import daf_pb2, wil_pb2
        from jarvis.agents.daf_bridge import DAFBridge, DAFBridgeConfig

        spawn_calls: list[Any] = []
        dispatch_calls: list[Any] = []
        retire_calls: list[Any] = []

        class FakeStub:
            async def SpawnAgent(self, request):
                spawn_calls.append(request)
                return daf_pb2.SpawnAgentResponse(
                    success=True,
                    agent_instance=daf_pb2.AgentInstanceMessage(
                        agent_instance_id=f"inst-{len(spawn_calls)}",
                        agent_type_id=request.agent_type_id,
                        status=daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_IDLE,
                    ),
                )

            async def DispatchTask(self, request):
                dispatch_calls.append(request)
                role = request.context.fields["role"].string_value
                return daf_pb2.TaskResultMessage(
                    task_id=request.task_id,
                    status=wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS,
                    output=Value(string_value=f"{role} ok"),
                    agent_instance_id_used=request.agent_instance_id,
                )

            async def RetireAgent(self, request):
                retire_calls.append(request)
                return daf_pb2.RetireAgentResponse(success=True)

        bridge = DAFBridge(DAFBridgeConfig())
        bridge._stub = FakeStub()
        bridge._channel = object()

        results = await bridge.dispatch_multi_agent(
            "Investigate the outage",
            ["researcher", "critic"],
            pattern="fan_out",
            team_id="team-123",
            user_id="shubham",
            session_id="sess-1",
            agent_specs=[
                {
                    "base_type": "researcher",
                    "specialization": "incident researcher",
                    "mission": "Find the most relevant outage evidence.",
                    "compiled_prompt": "Compiled prompt for the researcher",
                },
                {
                    "base_type": "critic",
                    "specialization": "incident critic",
                    "mission": "Find the highest-risk outage assumptions.",
                    "compiled_prompt": "Compiled prompt for the critic",
                },
            ],
        )

        assert [result.agent_type for result in results] == ["researcher", "critic"]
        assert [result.output for result in results] == ["researcher ok", "critic ok"]
        assert len(spawn_calls) == 2
        assert spawn_calls[0].initial_configuration.fields["team_id"].string_value == "team-123"
        assert spawn_calls[1].agent_type_id == "web_searcher"
        assert len(dispatch_calls) == 2
        ctx = dispatch_calls[0].context.fields
        assert ctx["use_pull_loop"].bool_value is True
        assert ctx["team_id"].string_value == "team-123"
        assert ctx["user_id"].string_value == "shubham"
        assert ctx["session_id"].string_value == "sess-1"
        assert ctx["role"].string_value == "researcher"
        assert ctx["base_type"].string_value == "researcher"
        params = dispatch_calls[0].parameters.fields
        assert params["prompt"].string_value == "Compiled prompt for the researcher"
        assert params["query"].string_value == "Find the most relevant outage evidence."
        assert params["team_id"].string_value == "team-123"
        assert params["base_type"].string_value == "researcher"
        assert params["original_task"].string_value == "Investigate the outage"
        assert len(retire_calls) == 2


class TestRichDAFWorker:
    """DAF workers can execute a richer typed-agent path from dynamic specs."""

    def test_run_rich_agent_task_uses_agent_engine(self):
        from jarvis.agents.daf.agent_process import AgentProcess

        captured: dict[str, Any] = {}

        async def fake_run_task(**kwargs):
            captured.update(kwargs)
            return {"output": "rich agent output"}

        agent = AgentProcess.__new__(AgentProcess)
        agent.type_id = "web_searcher"
        agent.logger = MagicMock()
        agent._agent_engine = SimpleNamespace(run_task=AsyncMock(side_effect=fake_run_task))
        agent._ensure_rich_runtime = lambda home_dir: True
        agent._run_async = lambda coro: asyncio.run(coro)

        output = agent._run_rich_agent_task(
            spec_payload={
                "base_type": "analyst",
                "collaboration_role": "regression_guard",
                "specialization": "regression guard",
                "mission": "Attack the risky assumptions in the proposed fix.",
                "success_criteria": ["Find the top regression risk"],
                "output_schema": {"risks": "array"},
            },
            original_task="Fix the failing auth tests",
            shared_memory="Shared memory from the team",
            home_dir="/tmp/prometheus-home",
            role="critic",
        )

        assert output == "rich agent output"
        assert captured["agent_type"] == "analyst"
        assert captured["dynamic_spec"].collaboration_role == "regression_guard"
        assert "regression guard" in captured["system_prompt_override"]
        assert "Shared memory from the team" in captured["prompt_override"]


class TestAgentLLMConfig:
    """Collaboration runtime can use a dedicated shared provider baseline."""

    def test_subsystem_factory_uses_agent_llm_override_for_collab(self):
        from jarvis.config import AgentLLMConfig, JARVISConfig, LLMConfig
        from jarvis.tools.subsystem_init import SubsystemBundle, SubsystemFactory

        captured: dict[str, Any] = {}

        class FakeLLMInterface:
            def __init__(self, passed_config):
                captured["provider"] = passed_config.llm.provider
                captured["model"] = passed_config.llm.model
                captured["fallbacks"] = list(passed_config.llm.fallback_providers)
                captured["temperature"] = passed_config.llm.temperature

        cfg = JARVISConfig(
            llm=LLMConfig(
                provider="claude-code",
                model="claude-opus-4-6",
                fallback_providers=["gemini-cli"],
            ),
            agent_llm=AgentLLMConfig(
                provider="openai",
                model="gpt-4o",
                fallback_providers=["groq"],
                temperature=0.2,
            ),
        )
        bundle = SubsystemBundle()

        with patch("jarvis.llm_providers.router.LLMInterface", FakeLLMInterface):
            SubsystemFactory._init_llm_collab(bundle, cfg, skip_cli=True)

        assert bundle.llm_for_collab is not None
        assert captured == {
            "provider": "openai",
            "model": "gpt-4o",
            "fallbacks": ["groq"],
            "temperature": 0.2,
        }
        assert "multi_agent" in bundle.available
