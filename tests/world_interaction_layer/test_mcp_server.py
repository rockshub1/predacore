"""
Tests for the Prometheus MCP Server.

Validates that:
1. The MCP server creates successfully with all tools registered
2. Tool catalog resource returns valid JSON
3. Each WIL tool is exposed as an MCP tool
4. Tool dispatch works for lightweight handlers (Wikipedia, web scraper)
5. Stub tools return informative status messages
"""
import pytest

try:
    from predacore._vendor.world_interaction_layer.mcp_server import _dispatch_tool, create_mcp_server
    from predacore._vendor.world_interaction_layer.tool_registry import SimpleToolRegistry
except ImportError:
    pytest.skip("world_interaction_layer not available in _vendor", allow_module_level=True)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    return SimpleToolRegistry()


@pytest.fixture
def mcp_server(registry):
    return create_mcp_server(name="TestPrometheus", tool_registry=registry)


# ---------------------------------------------------------------------------
# Server creation
# ---------------------------------------------------------------------------

class TestMCPServerCreation:
    """Tests for MCP server initialization."""

    def test_server_creates_successfully(self, mcp_server):
        assert mcp_server is not None
        assert mcp_server.name == "TestPrometheus"

    def test_all_wil_tools_registered(self, mcp_server, registry):
        """Every WIL tool should appear as an MCP tool."""
        wil_tools = registry.list_tools()
        mcp_tools = mcp_server._tool_manager._tools
        wil_ids = {t["tool_id"] for t in wil_tools}
        mcp_ids = set(mcp_tools.keys())
        assert wil_ids.issubset(mcp_ids), (
            f"Missing MCP tools: {wil_ids - mcp_ids}"
        )

    def test_tool_count_matches(self, mcp_server, registry):
        wil_count = len(registry.list_tools())
        mcp_count = len(mcp_server._tool_manager._tools)
        # MCP may have >= WIL tools (could add extras)
        assert mcp_count >= wil_count

    def test_specific_tools_present(self, mcp_server):
        """Key tools that must be exposed via MCP."""
        tool_names = set(mcp_server._tool_manager._tools.keys())
        for expected in [
            "browser_automation",
            "google_search_api",
            "basic_web_scraper",
            "python_sandbox",
            "rag_embed",
            "rag_retrieve",
            "rag_answer",
            "wikipedia_api",
            "document_summarizer",
            "llm_classify",
            "email_sender",
        ]:
            assert expected in tool_names, f"Tool '{expected}' not found in MCP server"


# ---------------------------------------------------------------------------
# Tool dispatch (async)
# ---------------------------------------------------------------------------

class TestToolDispatch:
    """Tests for the lightweight tool dispatch layer."""

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool(self, registry):
        result = await _dispatch_tool("nonexistent_tool_xyz", {}, registry)
        assert "error" in result
        assert result["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_dispatch_code_executor_returns_sandbox_required(self, registry):
        result = await _dispatch_tool("python_sandbox", {"code": "print(1)"}, registry)
        assert result["status"] == "sandbox_required"

    @pytest.mark.asyncio
    async def test_dispatch_node_executor_returns_sandbox_required(self, registry):
        result = await _dispatch_tool("node_executor", {"code": "console.log(1)"}, registry)
        assert result["status"] == "sandbox_required"

    @pytest.mark.asyncio
    async def test_dispatch_email_returns_api_key_required(self, registry):
        result = await _dispatch_tool(
            "email_sender",
            {"to": "test@test.com", "subject": "hi", "body": "hello"},
            registry,
        )
        assert result["status"] == "api_key_required"

    @pytest.mark.asyncio
    async def test_dispatch_translation_returns_api_key_required(self, registry):
        result = await _dispatch_tool(
            "translation_api",
            {"text": "hello", "target_lang": "ja"},
            registry,
        )
        assert result["status"] == "api_key_required"

    @pytest.mark.asyncio
    async def test_dispatch_rag_tools_return_service_required(self, registry):
        for tool_id in ("rag_embed", "rag_retrieve", "rag_answer"):
            result = await _dispatch_tool(tool_id, {"namespace": "test", "query": "test"}, registry)
            assert result["status"] == "service_required", f"Failed for {tool_id}"

    @pytest.mark.asyncio
    async def test_dispatch_llm_tools_return_llm_required(self, registry):
        for tool_id in ("llm_classify", "llm_extract", "document_summarizer"):
            result = await _dispatch_tool(tool_id, {"text": "test"}, registry)
            assert result["status"] == "llm_required", f"Failed for {tool_id}"

    @pytest.mark.asyncio
    async def test_dispatch_browser_tools_return_playwright_required(self, registry):
        for tool_id in ("browser_automation", "selector_extract"):
            result = await _dispatch_tool(tool_id, {"url": "https://example.com"}, registry)
            assert result["status"] == "playwright_required", f"Failed for {tool_id}"

    @pytest.mark.asyncio
    async def test_dispatch_google_search_returns_stub(self, registry):
        result = await _dispatch_tool("google_search_api", {"query": "test"}, registry)
        assert "results" in result
        assert result["query"] == "test"

    @pytest.mark.asyncio
    async def test_dispatch_weather_no_api_key(self, registry, monkeypatch):
        monkeypatch.delenv("OPENWEATHERMAP_API_KEY", raising=False)
        result = await _dispatch_tool("openweathermap_api", {"city": "Tokyo"}, registry)
        assert result["status"] == "api_key_required"
        assert result["city"] == "Tokyo"

    @pytest.mark.asyncio
    async def test_dispatch_web_scraper_missing_url(self, registry):
        result = await _dispatch_tool("basic_web_scraper", {}, registry)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_dispatch_edinet_returns_api_required(self, registry):
        result = await _dispatch_tool("edinet_fetch", {}, registry)
        assert result["status"] == "api_required"


# ---------------------------------------------------------------------------
# Multiple server instances (isolation)
# ---------------------------------------------------------------------------

class TestMCPServerIsolation:
    """Verify multiple server instances don't interfere."""

    def test_two_servers_independent(self):
        reg1 = SimpleToolRegistry()
        reg2 = SimpleToolRegistry()
        s1 = create_mcp_server(name="Server1", tool_registry=reg1)
        s2 = create_mcp_server(name="Server2", tool_registry=reg2)
        assert s1.name == "Server1"
        assert s2.name == "Server2"
        # Both should have the same tools
        assert len(s1._tool_manager._tools) == len(s2._tool_manager._tools)
