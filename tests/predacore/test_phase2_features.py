"""
Tests for Phase 2 new modules:
  - PluginRegistry / Plugin SDK
  - TranscriptWriter / session persistence
  - ConfigWatcher / hot-reload
  - SessionSandboxPool / per-session Docker isolation
  - VoiceInterface / TTS/STT
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════
# Plugin SDK Tests
# ═══════════════════════════════════════════════════════════════════

class TestPluginSDK:
    """Tests for the Plugin SDK: decorators, Plugin class, PluginRegistry."""

    def test_tool_decorator_metadata(self):
        from predacore.services.plugins import tool

        @tool(description="Add two numbers", name="add")
        async def add(a: int, b: int) -> int:
            return a + b

        assert hasattr(add, "_tool_meta")
        assert add._tool_meta.name == "add"
        assert add._tool_meta.description == "Add two numbers"

    def test_tool_decorator_auto_params(self):
        from predacore.services.plugins import tool

        @tool(description="Greet someone")
        def greet(name: str, loud: bool) -> str:
            return f"Hello {name}"

        params = greet._tool_meta.parameters
        assert params["type"] == "object"
        assert "name" in params["properties"]
        assert params["properties"]["name"]["type"] == "string"
        assert params["properties"]["loud"]["type"] == "boolean"
        assert "name" in params["required"]

    def test_hook_decorator_metadata(self):
        from predacore.services.plugins import hook

        @hook("on_message", priority=50)
        async def log_msg(message):
            pass

        assert hasattr(log_msg, "_hook_meta")
        assert log_msg._hook_meta.event == "on_message"
        assert log_msg._hook_meta.priority == 50

    def test_plugin_class_tools_discovery(self):
        from predacore.services.plugins import Plugin, tool

        class TestPlugin(Plugin):
            name = "test"
            version = "1.0.0"

            @tool(description="Say hi")
            def say_hi(self, name: str) -> str:
                return f"Hi {name}"

        p = TestPlugin()
        tools = p.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "say_hi"

    def test_plugin_class_hooks_discovery(self):
        from predacore.services.plugins import Plugin, hook

        class TestPlugin(Plugin):
            name = "test"
            version = "1.0.0"

            @hook("on_message")
            async def log_msg(self, message=None):
                pass

            @hook("on_error")
            async def handle_error(self, message=None):
                pass

        p = TestPlugin()
        hooks = p.get_hooks()
        assert len(hooks) == 2

    def test_plugin_enable_disable(self):
        from predacore.services.plugins import Plugin

        class TestPlugin(Plugin):
            name = "test"

        p = TestPlugin()
        assert p.enabled is True
        p.disable()
        assert p.enabled is False
        p.enable()
        assert p.enabled is True

    def test_plugin_config(self):
        from predacore.services.plugins import Plugin

        class TestPlugin(Plugin):
            name = "test"

        p = TestPlugin(config={"key": "value"})
        assert p.config["key"] == "value"

    def test_plugin_get_tool_definitions(self):
        from predacore.services.plugins import Plugin, tool

        class TestPlugin(Plugin):
            name = "weather"
            version = "1.0.0"

            @tool(description="Get weather")
            def get_weather(self, city: str) -> str:
                return "sunny"

        p = TestPlugin()
        defs = p.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "weather.get_weather"
        assert "[Plugin: weather]" in defs[0]["description"]

    @pytest.mark.asyncio
    async def test_registry_load_plugin(self):
        from predacore.services.plugins import Plugin, PluginRegistry, tool

        class TestPlugin(Plugin):
            name = "test"
            version = "1.0.0"
            loaded = False

            async def on_load(self):
                self.loaded = True

            @tool(description="Echo")
            async def echo(self, text: str) -> str:
                return text

        registry = PluginRegistry()
        plugin = await registry.load_plugin(TestPlugin)
        assert plugin.loaded is True
        assert registry.plugin_count == 1

    @pytest.mark.asyncio
    async def test_registry_duplicate_name_error(self):
        from predacore.services.plugins import Plugin, PluginRegistry

        class TestPlugin(Plugin):
            name = "test"

        registry = PluginRegistry()
        await registry.load_plugin(TestPlugin)
        with pytest.raises(ValueError, match="already loaded"):
            await registry.load_plugin(TestPlugin)

    @pytest.mark.asyncio
    async def test_registry_unload_plugin(self):
        from predacore.services.plugins import Plugin, PluginRegistry

        class TestPlugin(Plugin):
            name = "test"
            unloaded = False

            async def on_unload(self):
                self.unloaded = True

        registry = PluginRegistry()
        plugin = await registry.load_plugin(TestPlugin)
        assert registry.plugin_count == 1
        await registry.unload_plugin("test")
        assert registry.plugin_count == 0
        assert plugin.unloaded is True

    @pytest.mark.asyncio
    async def test_registry_call_tool(self):
        from predacore.services.plugins import Plugin, PluginRegistry, tool

        class TestPlugin(Plugin):
            name = "math"

            @tool(description="Add numbers")
            async def add(self, a: int, b: int) -> int:
                return a + b

        registry = PluginRegistry()
        await registry.load_plugin(TestPlugin)
        result = await registry.call_tool("math.add", {"a": 3, "b": 5})
        assert result == "8"

    @pytest.mark.asyncio
    async def test_registry_dispatch_hook(self):
        from predacore.services.plugins import Plugin, PluginRegistry, hook

        results = []

        class TestPlugin(Plugin):
            name = "test"

            @hook("on_message")
            async def log_msg(self, message=""):
                results.append(f"logged: {message}")
                return True

        registry = PluginRegistry()
        await registry.load_plugin(TestPlugin)
        hook_results = await registry.dispatch_hook("on_message", message="hello")
        assert len(hook_results) == 1
        assert results == ["logged: hello"]

    @pytest.mark.asyncio
    async def test_registry_get_all_tool_definitions(self):
        from predacore.services.plugins import Plugin, PluginRegistry, tool

        class Plugin1(Plugin):
            name = "p1"
            @tool(description="A")
            def a(self): pass

        class Plugin2(Plugin):
            name = "p2"
            @tool(description="B")
            def b(self): pass

        registry = PluginRegistry()
        await registry.load_plugin(Plugin1)
        await registry.load_plugin(Plugin2)
        defs = registry.get_all_tool_definitions()
        assert len(defs) == 2

    @pytest.mark.asyncio
    async def test_registry_get_loaded_plugins(self):
        from predacore.services.plugins import Plugin, PluginRegistry

        class TestPlugin(Plugin):
            name = "test"
            version = "2.0.0"
            description = "A test plugin"

        registry = PluginRegistry()
        await registry.load_plugin(TestPlugin)
        plugins = registry.get_loaded_plugins()
        assert len(plugins) == 1
        assert plugins[0]["name"] == "test"
        assert plugins[0]["version"] == "2.0.0"


# ═══════════════════════════════════════════════════════════════════
# Transcript Writer Tests
# ═══════════════════════════════════════════════════════════════════

class TestTranscriptWriter:
    """Tests for session transcript JSONL persistence."""

    def test_start_session_creates_file(self):
        from predacore.services.transcripts import TranscriptWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TranscriptWriter(output_dir=Path(tmpdir))
            path = writer.start_session("sess-1", user_id="u1", channel="telegram")
            assert path.exists()
            assert path.suffix == ".jsonl"

    def test_append_message(self):
        from predacore.services.transcripts import TranscriptWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TranscriptWriter(output_dir=Path(tmpdir))
            writer.start_session("sess-1")
            writer.append_message("sess-1", role="user", content="Hello")
            writer.append_message("sess-1", role="assistant", content="Hi!")

            entries = writer.read_transcript("sess-1")
            assert len(entries) == 3  # header + 2 messages
            assert entries[1].role == "user"
            assert entries[1].content == "Hello"
            assert entries[2].role == "assistant"

    def test_append_tool_call(self):
        from predacore.services.transcripts import TranscriptWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TranscriptWriter(output_dir=Path(tmpdir))
            writer.start_session("sess-1")
            writer.append_tool_call("sess-1", "weather", {"city": "London"})

            entries = writer.read_transcript("sess-1")
            assert entries[1].event_type == "tool_call"
            data = json.loads(entries[1].content)
            assert data["tool"] == "weather"

    def test_append_error(self):
        from predacore.services.transcripts import TranscriptWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TranscriptWriter(output_dir=Path(tmpdir))
            writer.start_session("sess-1")
            writer.append_error("sess-1", "Something failed")

            entries = writer.read_transcript("sess-1")
            assert entries[1].event_type == "error"
            assert entries[1].content == "Something failed"

    def test_close_session(self):
        from predacore.services.transcripts import TranscriptWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TranscriptWriter(output_dir=Path(tmpdir))
            writer.start_session("sess-1")
            writer.close_session("sess-1")

            entries = writer.read_transcript("sess-1")
            assert entries[-1].event_type == "session_end"

    def test_list_sessions(self):
        from predacore.services.transcripts import TranscriptWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TranscriptWriter(output_dir=Path(tmpdir))
            writer.start_session("sess-1")
            writer.start_session("sess-2")
            writer.start_session("sess-3")

            sessions = writer.list_sessions()
            assert len(sessions) == 3

    def test_read_empty_transcript(self):
        from predacore.services.transcripts import TranscriptWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TranscriptWriter(output_dir=Path(tmpdir))
            entries = writer.read_transcript("nonexistent")
            assert entries == []

    def test_get_stats(self):
        from predacore.services.transcripts import TranscriptWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TranscriptWriter(output_dir=Path(tmpdir))
            writer.start_session("sess-1")
            writer.append_message("sess-1", "user", "Hello")

            stats = writer.get_stats()
            assert stats["total_sessions"] == 1
            assert stats["total_size_bytes"] > 0

    def test_transcript_entry_serialization(self):
        from predacore.services.transcripts import TranscriptEntry

        entry = TranscriptEntry(
            timestamp=1234567890.0,
            event_type="message",
            role="user",
            content="Hello",
            metadata={"key": "value"},
        )
        d = entry.to_dict()
        restored = TranscriptEntry.from_dict(d)
        assert restored.timestamp == entry.timestamp
        assert restored.content == entry.content
        assert restored.metadata == entry.metadata

    def test_path_sanitization(self):
        from predacore.services.transcripts import TranscriptWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TranscriptWriter(output_dir=Path(tmpdir))
            path = writer.start_session("../../../etc/passwd")
            assert ".." not in path.name


# ═══════════════════════════════════════════════════════════════════
# Config Watcher Tests
# ═══════════════════════════════════════════════════════════════════

class TestConfigWatcher:
    """Tests for hot-reload config watcher."""

    def test_init(self):
        from predacore.services.config_watcher import ConfigWatcher

        watcher = ConfigWatcher(config_path=Path("/tmp/test.yaml"))
        assert watcher.is_running is False
        assert watcher.reload_count == 0

    def test_on_change_registers_callback(self):
        from predacore.services.config_watcher import ConfigWatcher

        watcher = ConfigWatcher(config_path=Path("/tmp/test.yaml"))
        callback = lambda data: None
        watcher.on_change(callback)
        assert len(watcher._callbacks) == 1

    @pytest.mark.asyncio
    async def test_start_stop(self):
        from predacore.services.config_watcher import ConfigWatcher

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"key": "value"}, f)
            config_path = Path(f.name)

        try:
            watcher = ConfigWatcher(config_path=config_path, poll_interval=0.1)
            await watcher.start()
            assert watcher.is_running is True
            await watcher.stop()
            assert watcher.is_running is False
        finally:
            os.unlink(config_path)

    @pytest.mark.asyncio
    async def test_detects_change(self):
        from predacore.services.config_watcher import ConfigWatcher

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"version": 1}, f)
            config_path = Path(f.name)

        received = []

        try:
            watcher = ConfigWatcher(config_path=config_path, poll_interval=0.1)
            watcher.on_change(lambda data: received.append(data))
            await watcher.start()

            # Wait a moment, then modify the file
            await asyncio.sleep(0.2)
            with open(config_path, "w") as f:
                json.dump({"version": 2}, f)

            # Wait for detection
            await asyncio.sleep(0.5)
            await watcher.stop()

            assert len(received) == 1
            assert received[0]["version"] == 2
            assert watcher.reload_count == 1
        finally:
            os.unlink(config_path)

    def test_get_stats(self):
        from predacore.services.config_watcher import ConfigWatcher

        watcher = ConfigWatcher(config_path=Path("/tmp/test.yaml"), poll_interval=5.0)
        stats = watcher.get_stats()
        assert stats["poll_interval"] == 5.0
        assert stats["reload_count"] == 0
        assert stats["is_running"] is False


# ═══════════════════════════════════════════════════════════════════
# Sandbox Sessions Tests
# ═══════════════════════════════════════════════════════════════════

class TestSandboxSessions:
    """Tests for Docker per-session sandbox isolation."""

    def test_sandbox_config_defaults(self):
        from predacore.auth.sandbox import SandboxConfig

        config = SandboxConfig()
        assert config.image == "python:3.11-slim"
        assert config.cpu_limit == 0.5
        assert config.mem_limit == "256m"
        assert config.timeout == 30
        assert config.network_allowed is False

    def test_session_sandbox_init(self):
        from predacore.auth.sandbox import SandboxConfig, SessionSandbox

        sandbox = SessionSandbox(session_id="test-1", config=SandboxConfig())
        assert sandbox.session_id == "test-1"
        assert sandbox.execution_count == 0
        assert sandbox.total_cpu_time == 0.0

    def test_session_sandbox_volume_name(self):
        from predacore.auth.sandbox import SandboxConfig, SessionSandbox

        sandbox = SessionSandbox(session_id="user/session-123", config=SandboxConfig())
        assert "predacore-sandbox-" in sandbox.volume_name
        assert "/" not in sandbox.volume_name

    def test_session_sandbox_stats(self):
        from predacore.auth.sandbox import SandboxConfig, SessionSandbox

        sandbox = SessionSandbox(session_id="test-1", config=SandboxConfig())
        stats = sandbox.get_stats()
        assert stats["session_id"] == "test-1"
        assert stats["execution_count"] == 0

    @pytest.mark.asyncio
    async def test_pool_acquire(self):
        from predacore.auth.sandbox import SessionSandboxPool

        pool = SessionSandboxPool()
        sandbox = await pool.acquire("sess-1")
        assert sandbox.session_id == "sess-1"
        assert pool.get_stats()["active_sessions"] == 1

    @pytest.mark.asyncio
    async def test_pool_acquire_same_session(self):
        from predacore.auth.sandbox import SessionSandboxPool

        pool = SessionSandboxPool()
        s1 = await pool.acquire("sess-1")
        s2 = await pool.acquire("sess-1")
        assert s1 is s2  # Same instance reused

    @pytest.mark.asyncio
    async def test_pool_release(self):
        from predacore.auth.sandbox import SessionSandboxPool

        pool = SessionSandboxPool()
        await pool.acquire("sess-1")
        await pool.release("sess-1")
        assert pool.get_stats()["active_sessions"] == 0

    @pytest.mark.asyncio
    async def test_pool_max_sessions_eviction(self):
        from predacore.auth.sandbox import SessionSandboxPool

        pool = SessionSandboxPool(max_sessions=2)
        await pool.acquire("sess-1")
        await pool.acquire("sess-2")
        # This should evict the oldest
        await pool.acquire("sess-3")
        stats = pool.get_stats()
        assert stats["active_sessions"] == 2
        assert "sess-1" not in stats["sandboxes"]

    def test_language_config(self):
        from predacore.auth.sandbox import _get_language_config

        py = _get_language_config("python")
        assert py is not None
        assert py["extension"] == ".py"

        js = _get_language_config("javascript")
        assert js is not None
        assert js["extension"] == ".js"

        unknown = _get_language_config("fortran")
        assert unknown is None

    @pytest.mark.asyncio
    async def test_pool_stop_cleanup(self):
        from predacore.auth.sandbox import SessionSandboxPool

        pool = SessionSandboxPool()
        await pool.acquire("sess-1")
        await pool.acquire("sess-2")
        await pool.stop_cleanup_loop()
        assert pool.get_stats()["active_sessions"] == 0


# ═══════════════════════════════════════════════════════════════════
# Voice Interface Tests
# ═══════════════════════════════════════════════════════════════════

class TestVoiceInterface:
    """Tests for voice TTS/STT interface."""

    def test_voice_config_defaults(self):
        from predacore.services.voice import VoiceConfig, VoiceProvider

        config = VoiceConfig()
        assert config.tts_provider == VoiceProvider.KOKORO
        assert config.stt_provider == VoiceProvider.OPENAI
        assert config.voice_id == "bm_george"

    def test_voice_interface_init(self):
        from predacore.services.voice import VoiceInterface

        vi = VoiceInterface()
        assert vi._tts is not None

    def test_voice_interface_stats(self):
        from predacore.services.voice import VoiceInterface

        vi = VoiceInterface()
        stats = vi.get_stats()
        assert stats["tts_provider"] == "kokoro"
        assert stats["voice_id"] == "bm_george"

    def test_transcription_result(self):
        from predacore.services.voice import TranscriptionResult

        result = TranscriptionResult(
            text="Hello world",
            language="en",
            confidence=0.95,
            duration_ms=500,
        )
        d = result.to_dict()
        assert d["text"] == "Hello world"
        assert d["confidence"] == 0.95

    def test_synthesis_result(self):
        from predacore.services.voice import SynthesisResult

        result = SynthesisResult(
            audio_bytes=b"fake audio data",
            format="mp3",
            duration_ms=1000,
        )
        d = result.to_dict()
        assert d["format"] == "mp3"
        assert d["size_bytes"] == len(b"fake audio data")
        assert len(result.audio_base64) > 0

    def test_voice_provider_enum(self):
        from predacore.services.voice import VoiceProvider

        assert VoiceProvider.SYSTEM == "system"
        assert VoiceProvider.OPENAI == "openai"
        assert VoiceProvider.GOOGLE == "google"

    @pytest.mark.asyncio
    async def test_transcribe_raises_without_stt(self):
        from predacore.services.voice import VoiceConfig, VoiceInterface, VoiceProvider

        # Use system for both (no STT for system)
        config = VoiceConfig(stt_provider=VoiceProvider.SYSTEM)
        vi = VoiceInterface(config=config)
        vi._stt = None  # Explicitly clear

        with pytest.raises(RuntimeError, match="No STT provider"):
            await vi.transcribe(b"audio")
