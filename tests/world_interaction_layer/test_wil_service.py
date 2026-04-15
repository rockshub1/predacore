"""
Unit tests for the World Interaction Layer (WIL) service helpers.
"""
import os
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

try:
    from jarvis._vendor.common.protos import egm_pb2, egm_pb2_grpc, wil_pb2
    from jarvis._vendor.world_interaction_layer.service import (
        AbstractSandboxManager,
        AbstractToolRegistry,
        WorldInteractionLayerService,
    )
except ImportError:
    pytest.skip("world_interaction_layer not available in _vendor", allow_module_level=True)

import grpc
import grpc.aio
import httpx

# Import components to be tested
from prometheus_client import CollectorRegistry

# Use pytest-asyncio for async functions
pytestmark = pytest.mark.asyncio

# --- Fixtures ---

@pytest.fixture(autouse=True)
def _policy_env_defaults(monkeypatch):
    """Keep policy env deterministic across tests in this module."""
    monkeypatch.setenv("EGM_MODE", "strict")
    monkeypatch.setenv("APPROVALS_REQUIRED", "1")


@pytest.fixture
def mock_tool_registry() -> AbstractToolRegistry:
    registry = MagicMock(spec=AbstractToolRegistry)
    registry.list_tools = MagicMock(return_value=[])
    registry.get_tool = MagicMock(return_value=None)
    return registry

@pytest.fixture
def mock_sandbox_manager() -> AbstractSandboxManager:
    manager = MagicMock(spec=AbstractSandboxManager)
    manager.run = AsyncMock(return_value={
        "status": "SUCCESS", "stdout": "", "stderr": "", "result": None, "error_message": ""
    })
    return manager

@pytest.fixture
def mock_egm_stub() -> egm_pb2_grpc.EthicalGovernanceModuleServiceStub:
    stub = MagicMock(spec=egm_pb2_grpc.EthicalGovernanceModuleServiceStub)
    stub.CheckActionCompliance = AsyncMock(return_value=egm_pb2.ComplianceCheckResultMessage(is_compliant=True))
    stub.LogEvent = AsyncMock()
    return stub

@pytest.fixture
def wil_service(mock_tool_registry, mock_sandbox_manager, mock_egm_stub) -> WorldInteractionLayerService:
    """Provides a WIL service instance with mock dependencies."""
    return WorldInteractionLayerService(
        tool_registry=mock_tool_registry,
        sandbox_manager=mock_sandbox_manager,
        egm_stub=mock_egm_stub,
        metrics_registry=CollectorRegistry(),
    )


def _make_async_context():
    """Helper to create an AsyncMock context that properly supports await on abort."""
    return AsyncMock(spec=grpc.aio.ServicerContext)


def _make_aio_rpc_error(code, details_str=""):
    """Helper to construct AioRpcError with proper args."""
    return grpc.aio.AioRpcError(
        code=code,
        initial_metadata=grpc.aio.Metadata(),
        trailing_metadata=grpc.aio.Metadata(),
        details=details_str,
    )


# --- Test Cases for _call_api ---

async def test_call_api_success_get(wil_service: WorldInteractionLayerService, httpx_mock):
    """Test successful GET request via _call_api."""
    tool_info = {
        "tool_id": "test_get_api",
        "metadata": {"base_url": "https://test.com", "endpoint": "/data", "method": "GET", "retries": 1}
    }
    params = {"id": "123"}
    expected_url = "https://test.com/data?id=123"
    mock_response_data = {"result": "success", "value": 42}

    httpx_mock.add_response(url=expected_url, method="GET", json=mock_response_data, status_code=200)

    result = await wil_service._call_api(tool_info, params, None)

    # Service adds __status_code to the result
    assert result["result"] == "success"
    assert result["value"] == 42
    assert result["__status_code"] == 200

async def test_call_api_success_post(wil_service: WorldInteractionLayerService, httpx_mock):
    """Test successful POST request via _call_api."""
    tool_info = {
        "tool_id": "test_post_api",
        "metadata": {"base_url": "https://test.com", "endpoint": "/create", "method": "POST", "retries": 1}
    }
    payload = {"name": "test", "value": True}
    expected_url = "https://test.com/create"
    mock_response_data = {"id": "xyz789", "status": "created"}

    httpx_mock.add_response(url=expected_url, method="POST", json=mock_response_data, status_code=201)

    result = await wil_service._call_api(tool_info, payload, None)

    assert result["id"] == "xyz789"
    assert result["status"] == "created"
    assert result["__status_code"] == 201

async def test_call_api_with_bearer_auth(wil_service: WorldInteractionLayerService, httpx_mock):
    """Test API call with Bearer token authentication."""
    tool_info = {
        "tool_id": "test_auth_api",
        "metadata": {"base_url": "https://secure.com", "endpoint": "/me", "method": "GET", "auth_type": "bearer", "retries": 1}
    }
    credentials = {"token": "mysecrettoken"}
    expected_url = "https://secure.com/me"
    mock_response_data = {"user": "agent007"}

    httpx_mock.add_response(url=expected_url, method="GET", json=mock_response_data, status_code=200)

    result = await wil_service._call_api(tool_info, {}, credentials)

    assert result["user"] == "agent007"
    assert result["__status_code"] == 200
    request = httpx_mock.get_request()
    assert request.headers["authorization"] == "Bearer mysecrettoken"

async def test_call_api_http_error(wil_service: WorldInteractionLayerService, httpx_mock):
    """Test handling of HTTP status errors (4xx/5xx)."""
    tool_info = {
        "tool_id": "error_api",
        "metadata": {"base_url": "https://test.com", "endpoint": "/fail", "retries": 1}
    }
    expected_url = "https://test.com/fail"
    httpx_mock.add_response(url=expected_url, method="GET", status_code=404, text="Not Found")

    with pytest.raises(ValueError) as excinfo:
        await wil_service._call_api(tool_info, {}, None)
    assert "API returned error 404" in str(excinfo.value)

async def test_call_api_request_error(wil_service: WorldInteractionLayerService, httpx_mock):
    """Test handling of request errors (e.g., connection error)."""
    tool_info = {
        "tool_id": "conn_error_api",
        "metadata": {"base_url": "https://nonexistent.domain", "endpoint": "/", "retries": 1}
    }
    expected_url = "https://nonexistent.domain/"
    httpx_mock.add_response(url=expected_url, method="GET", status_code=500)

    with pytest.raises((ConnectionError, ValueError)):
        await wil_service._call_api(tool_info, {}, None)

async def test_call_api_non_json_response(wil_service: WorldInteractionLayerService, httpx_mock):
    """Test handling when API response is not valid JSON."""
    tool_info = {
        "tool_id": "html_api",
        "metadata": {"base_url": "https://test.com", "endpoint": "/page", "retries": 1}
    }
    expected_url = "https://test.com/page"
    html_content = "<html><body>Hello</body></html>"
    httpx_mock.add_response(url=expected_url, method="GET", text=html_content, status_code=200, headers={"content-type": "text/html"})

    result = await wil_service._call_api(tool_info, {}, None)

    # Expect fallback to raw content, plus __status_code
    assert result["raw_content"] == html_content
    assert result["__status_code"] == 200


# --- Test Cases for _scrape_web ---

async def test_scrape_web_success(wil_service: WorldInteractionLayerService, httpx_mock):
    """Test successful web scraping."""
    # Disable robots.txt checking for clean test
    wil_service._respect_robots = False

    tool_info = {"tool_id": "test_scraper"}
    url = "https://example.com/page"
    params = {"url": url}
    html_content = "<html><title>Test Page</title><body>Content</body></html>"

    httpx_mock.add_response(url=url, method="GET", text=html_content, status_code=200)

    result = await wil_service._scrape_web(tool_info, params)

    assert result == html_content
    requests = httpx_mock.get_requests()
    # Should have exactly 1 request (no robots.txt check)
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert str(requests[0].url) == url
    assert "User-Agent" in requests[0].headers

async def test_scrape_web_missing_url(wil_service: WorldInteractionLayerService):
    """Test web scraping when URL parameter is missing."""
    tool_info = {"tool_id": "test_scraper"}
    params = {} # Missing 'url'

    with pytest.raises(ValueError) as excinfo:
        await wil_service._scrape_web(tool_info, params)
    assert "Missing or invalid 'url' parameter" in str(excinfo.value)

async def test_scrape_web_http_error(wil_service: WorldInteractionLayerService, httpx_mock):
    """Test web scraping with HTTP error."""
    wil_service._respect_robots = False

    tool_info = {"tool_id": "test_scraper"}
    url = "https://example.com/notfound"
    params = {"url": url}
    httpx_mock.add_response(url=url, method="GET", status_code=404)

    with pytest.raises(ValueError) as excinfo:
        await wil_service._scrape_web(tool_info, params)
    assert "Web server returned error 404" in str(excinfo.value)

async def test_scrape_web_request_error(wil_service: WorldInteractionLayerService, httpx_mock):
    """Test web scraping with connection error."""
    wil_service._respect_robots = False

    tool_info = {"tool_id": "test_scraper"}
    url = "https://nonexistent.invalid/"
    params = {"url": url}
    httpx_mock.add_exception(httpx.ConnectError("Failed to connect"), url=url, method="GET")

    with pytest.raises(ConnectionError) as excinfo:
        await wil_service._scrape_web(tool_info, params)
    assert "Web scraping request failed" in str(excinfo.value)


# --- Test Cases for ExecuteCode ---

@patch.dict(os.environ, {"APPROVALS_REQUIRED": "0"}, clear=False)
async def test_execute_code_success(wil_service: WorldInteractionLayerService, mock_sandbox_manager, mock_egm_stub):
    """Test successful code execution."""
    request_id = str(uuid4())
    code_to_run = "print('Hello')"
    request = wil_pb2.CodeExecutionRequestMessage(
        request_id=request_id,
        code=code_to_run,
        timeout_seconds=30
    )
    mock_context = _make_async_context()

    # Configure mock sandbox result for this test
    mock_sandbox_manager.run.return_value = {
        "status": "SUCCESS", "stdout": "Hello\n", "stderr": "", "result": None, "error_message": ""
    }

    result = await wil_service.ExecuteCode(request, mock_context)

    # Assertions
    assert result.request_id == request_id
    assert result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS
    assert result.stdout == "Hello\n"
    assert result.stderr == ""
    assert result.error_message == ""
    # Check that sandbox manager was called correctly
    mock_sandbox_manager.run.assert_called_once()
    call_kwargs = mock_sandbox_manager.run.call_args.kwargs
    assert call_kwargs["code"] == code_to_run
    assert call_kwargs["timeout"] == 30
    # Verify EGM check was called
    mock_egm_stub.CheckActionCompliance.assert_called_once()
    mock_context.abort.assert_not_called()

@patch.dict(os.environ, {"APPROVALS_REQUIRED": "0"}, clear=False)
async def test_execute_code_sandbox_failure(wil_service: WorldInteractionLayerService, mock_sandbox_manager, mock_egm_stub):
    """Test code execution when the sandbox reports failure."""
    request_id = str(uuid4())
    request = wil_pb2.CodeExecutionRequestMessage(request_id=request_id, code="invalid code")
    mock_context = _make_async_context()

    # Configure mock sandbox result
    mock_sandbox_manager.run.return_value = {
        "status": "FAILED", "stdout": "", "stderr": "Syntax Error", "result": None, "error_message": "Code failed to execute"
    }

    result = await wil_service.ExecuteCode(request, mock_context)

    assert result.request_id == request_id
    assert result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_FAILED
    assert result.stderr == "Syntax Error"
    assert result.error_message == "Code failed to execute"
    mock_egm_stub.CheckActionCompliance.assert_called_once()
    mock_context.abort.assert_not_called()

@patch.dict(os.environ, {"APPROVALS_REQUIRED": "0"}, clear=False)
async def test_execute_code_sandbox_timeout(wil_service: WorldInteractionLayerService, mock_sandbox_manager, mock_egm_stub):
    """Test code execution when the sandbox reports a timeout."""
    request_id = str(uuid4())
    request = wil_pb2.CodeExecutionRequestMessage(request_id=request_id, code="while True: pass", timeout_seconds=5)
    mock_context = _make_async_context()

    # Configure mock sandbox result
    mock_sandbox_manager.run.return_value = {
        "status": "TIMEOUT", "stdout": "", "stderr": "", "result": None, "error_message": "Execution timed out"
    }

    result = await wil_service.ExecuteCode(request, mock_context)

    assert result.request_id == request_id
    assert result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_TIMEOUT
    assert result.error_message == "Execution timed out"
    mock_egm_stub.CheckActionCompliance.assert_called_once()
    mock_context.abort.assert_not_called()

@patch.dict(os.environ, {"APPROVALS_REQUIRED": "0"}, clear=False)
async def test_execute_code_sandbox_exception(wil_service: WorldInteractionLayerService, mock_sandbox_manager, mock_egm_stub):
    """Test code execution when the sandbox manager itself raises an exception."""
    request_id = str(uuid4())
    request = wil_pb2.CodeExecutionRequestMessage(request_id=request_id, code="some code")
    mock_context = _make_async_context()

    # Configure mock sandbox to raise an error
    mock_sandbox_manager.run.side_effect = Exception("Sandbox infrastructure error")

    result = await wil_service.ExecuteCode(request, mock_context)

    # Service should abort the call and return an error status
    assert result.request_id == request_id
    assert result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_ERROR
    mock_egm_stub.CheckActionCompliance.assert_called_once()
    mock_context.abort.assert_called_once_with(
        grpc.StatusCode.INTERNAL, "Failed to execute code in sandbox"
    )

@pytest.mark.asyncio
@patch.dict(os.environ, {"APPROVALS_REQUIRED": "0"}, clear=False)
async def test_execute_code_egm_blocks(wil_service: WorldInteractionLayerService, mock_egm_stub, mock_sandbox_manager):
    """Test ExecuteCode when EGM check returns non-compliant."""
    request_id = str(uuid4())
    request = wil_pb2.CodeExecutionRequestMessage(request_id=request_id, code="import os; os.rmdir('/')")
    mock_context = _make_async_context()

    # Mock EGM stub to return non-compliant
    violation_msg = "Execution of potentially harmful code blocked."
    mock_egm_stub.CheckActionCompliance.return_value = egm_pb2.ComplianceCheckResultMessage(
        is_compliant=False,
        justification=violation_msg
    )

    result = await wil_service.ExecuteCode(request, mock_context)

    assert result.request_id == request_id
    assert result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_FAILED
    assert f"Code execution blocked by EGM: {violation_msg}" in result.error_message
    mock_egm_stub.CheckActionCompliance.assert_called_once()
    mock_sandbox_manager.run.assert_not_called()
    mock_context.abort.assert_not_called()

@pytest.mark.asyncio
@patch.dict(os.environ, {"APPROVALS_REQUIRED": "0", "EGM_MODE": "off"}, clear=False)
async def test_execute_code_egm_off_skips_check(wil_service: WorldInteractionLayerService, mock_egm_stub, mock_sandbox_manager):
    """EGM_MODE=off should bypass compliance checks for code execution."""
    request = wil_pb2.CodeExecutionRequestMessage(request_id=str(uuid4()), code="print('ok')", timeout_seconds=30)
    mock_context = _make_async_context()

    mock_sandbox_manager.run.return_value = {
        "status": "SUCCESS", "stdout": "ok\n", "stderr": "", "result": None, "error_message": ""
    }

    result = await wil_service.ExecuteCode(request, mock_context)
    assert result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS
    mock_egm_stub.CheckActionCompliance.assert_not_called()
    mock_sandbox_manager.run.assert_called_once()

@pytest.mark.asyncio
@patch.dict(os.environ, {"APPROVALS_REQUIRED": "0"}, clear=False)
async def test_execute_code_egm_grpc_error(wil_service: WorldInteractionLayerService, mock_egm_stub, mock_sandbox_manager):
    """Test ExecuteCode when the EGM check itself fails with a gRPC error."""
    request_id = str(uuid4())
    request = wil_pb2.CodeExecutionRequestMessage(request_id=request_id, code="print('safe')")
    mock_context = _make_async_context()

    # Mock EGM stub to raise gRPC error with correct constructor
    mock_egm_stub.CheckActionCompliance.side_effect = _make_aio_rpc_error(
        grpc.StatusCode.UNAVAILABLE, "EGM service down"
    )

    result = await wil_service.ExecuteCode(request, mock_context)

    assert result.request_id == request_id
    assert result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_ERROR
    assert "Internal error during compliance check" in result.error_message
    mock_egm_stub.CheckActionCompliance.assert_called_once()
    mock_sandbox_manager.run.assert_not_called()
    mock_context.abort.assert_called_once_with(grpc.StatusCode.INTERNAL, "Failed to check code compliance with EGM")


# --- Test Cases for ExecuteTool ---

@pytest.mark.asyncio
async def test_execute_tool_api_success(wil_service: WorldInteractionLayerService, mock_tool_registry, mock_egm_stub):
    """Test successful execution of an API tool."""
    tool_id = "google_search_api"
    request_id = str(uuid4())
    params_struct = wil_pb2.google_dot_protobuf_dot_struct__pb2.Struct()
    params_struct.update({"query": "prometheus"})
    request = wil_pb2.InteractionRequestMessage(request_id=request_id, tool_id=tool_id, parameters=params_struct)
    mock_context = _make_async_context()

    # Mock tool registry response
    tool_info = {
        "tool_id": tool_id, "tool_type": "API", "description": "Search",
        "metadata": {"base_url": "https://api.example.com", "endpoint": "/search"}
    }
    mock_tool_registry.get_tool.return_value = tool_info

    # Mock the internal helper methods directly on the instance
    mock_api_result = {"results": [{"title": "Prometheus - Wikipedia"}]}
    wil_service._call_api = AsyncMock(return_value=mock_api_result)
    wil_service._scrape_web = AsyncMock()

    result = await wil_service.ExecuteTool(request, mock_context)

    assert result.request_id == request_id
    assert result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS
    assert result.error_message == ""
    assert "results" in result.output.struct_value.fields

    mock_tool_registry.get_tool.assert_called_once_with(tool_id)
    wil_service._call_api.assert_called_once_with(tool_info, {"query": "prometheus"}, None)
    wil_service._scrape_web.assert_not_called()
    mock_egm_stub.CheckActionCompliance.assert_called_once()

@pytest.mark.asyncio
async def test_execute_tool_scraper_success(wil_service: WorldInteractionLayerService, mock_tool_registry, mock_egm_stub):
    """Test successful execution of a Web Scraper tool."""
    tool_id = "basic_web_scraper"
    request_id = str(uuid4())
    url_to_scrape = "https://example.com"
    params_struct = wil_pb2.google_dot_protobuf_dot_struct__pb2.Struct()
    params_struct.update({"url": url_to_scrape})
    request = wil_pb2.InteractionRequestMessage(request_id=request_id, tool_id=tool_id, parameters=params_struct)
    mock_context = _make_async_context()

    # Mock tool registry response
    tool_info = {"tool_id": tool_id, "tool_type": "WEB_SCRAPER", "description": "Scrape"}
    mock_tool_registry.get_tool.return_value = tool_info

    # Mock the internal helper methods directly on the instance
    mock_html_content = "<html><body>Scraped</body></html>"
    wil_service._scrape_web = AsyncMock(return_value=mock_html_content)
    wil_service._call_api = AsyncMock()

    result = await wil_service.ExecuteTool(request, mock_context)

    assert result.request_id == request_id
    assert result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS
    assert result.error_message == ""
    assert result.output.string_value == mock_html_content

    mock_tool_registry.get_tool.assert_called_once_with(tool_id)
    wil_service._scrape_web.assert_called_once_with(tool_info, {"url": url_to_scrape})
    wil_service._call_api.assert_not_called()
    mock_egm_stub.CheckActionCompliance.assert_called_once()

@pytest.mark.asyncio
async def test_execute_tool_not_found(wil_service: WorldInteractionLayerService, mock_tool_registry):
    """Test executing a tool that doesn't exist in the registry."""
    tool_id = "nonexistent_tool"
    request = wil_pb2.InteractionRequestMessage(tool_id=tool_id)
    mock_context = _make_async_context()

    mock_tool_registry.get_tool.return_value = None

    result = await wil_service.ExecuteTool(request, mock_context)

    assert result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_ERROR
    assert f"Tool '{tool_id}' not found" in result.error_message
    mock_tool_registry.get_tool.assert_called_once_with(tool_id)
    mock_context.abort.assert_called_once_with(grpc.StatusCode.NOT_FOUND, f"Tool '{tool_id}' not found.")

@pytest.mark.asyncio
async def test_execute_tool_helper_exception(wil_service: WorldInteractionLayerService, mock_tool_registry):
    """Test ExecuteTool when an internal helper (_call_api) raises an exception."""
    tool_id = "api_fails"
    request = wil_pb2.InteractionRequestMessage(tool_id=tool_id)
    mock_context = _make_async_context()

    tool_info = {"tool_id": tool_id, "tool_type": "API"}
    mock_tool_registry.get_tool.return_value = tool_info

    error_message = "Network connection failed"
    wil_service._call_api = AsyncMock(side_effect=ConnectionError(error_message))

    result = await wil_service.ExecuteTool(request, mock_context)

    assert result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_FAILED
    mock_context.abort.assert_called_once_with(grpc.StatusCode.INTERNAL, f"Failed to execute tool '{tool_id}'.")

@pytest.mark.asyncio
async def test_execute_tool_egm_blocks(wil_service: WorldInteractionLayerService, mock_tool_registry, mock_egm_stub):
    """Test ExecuteTool when EGM check returns non-compliant."""
    tool_id = "risky_api"
    request_id = str(uuid4())
    request = wil_pb2.InteractionRequestMessage(request_id=request_id, tool_id=tool_id)
    mock_context = _make_async_context()

    tool_info = {"tool_id": tool_id, "tool_type": "API"}
    mock_tool_registry.get_tool.return_value = tool_info

    violation_msg = "Blocked by policy XYZ"
    mock_egm_stub.CheckActionCompliance.return_value = egm_pb2.ComplianceCheckResultMessage(
        is_compliant=False,
        justification=violation_msg
    )

    wil_service._call_api = AsyncMock()

    result = await wil_service.ExecuteTool(request, mock_context)

    assert result.request_id == request_id
    assert result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_FAILED
    assert f"Tool execution blocked by EGM: {violation_msg}" in result.error_message

    mock_tool_registry.get_tool.assert_called_once_with(tool_id)
    mock_egm_stub.CheckActionCompliance.assert_called_once()
    wil_service._call_api.assert_not_called()
    mock_context.abort.assert_not_called()

@pytest.mark.asyncio
@patch.dict(os.environ, {"EGM_MODE": "log_only"}, clear=False)
async def test_execute_tool_egm_log_only_continues(wil_service: WorldInteractionLayerService, mock_tool_registry, mock_egm_stub):
    """EGM_MODE=log_only should not block tool execution when non-compliant."""
    tool_id = "risky_api"
    request = wil_pb2.InteractionRequestMessage(request_id=str(uuid4()), tool_id=tool_id)
    mock_context = _make_async_context()

    mock_tool_registry.get_tool.return_value = {
        "tool_id": tool_id,
        "tool_type": "API",
        "metadata": {"base_url": "https://api.example.com", "endpoint": "/x"},
    }
    mock_egm_stub.CheckActionCompliance.return_value = egm_pb2.ComplianceCheckResultMessage(
        is_compliant=False,
        justification="would block in strict mode",
    )
    wil_service._call_api = AsyncMock(return_value={"ok": True})

    result = await wil_service.ExecuteTool(request, mock_context)

    assert result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS
    wil_service._call_api.assert_called_once()
    mock_egm_stub.CheckActionCompliance.assert_called_once()

# --- Test Cases for ListAvailableTools ---

@pytest.mark.asyncio
async def test_list_available_tools_success(wil_service: WorldInteractionLayerService, mock_tool_registry):
    """Test successfully listing available tools."""
    mock_context = _make_async_context()
    # Use TOOL_TYPE_ prefix to match proto enum names
    mock_tools_data = [
        {"tool_id": "tool1", "tool_type": "TOOL_TYPE_API", "description": "Desc 1"},
        {"tool_id": "tool2", "tool_type": "TOOL_TYPE_CODE_EXECUTOR", "description": "Desc 2"},
    ]
    mock_tool_registry.list_tools.return_value = mock_tools_data

    response = await wil_service.ListAvailableTools(wil_pb2.ListAvailableToolsRequest(), mock_context)

    assert len(response.tools) == 2
    assert response.tools[0].tool_id == "tool1"
    assert response.tools[0].tool_type == wil_pb2.ToolType.TOOL_TYPE_API
    assert response.tools[0].description == "Desc 1"
    assert response.tools[1].tool_id == "tool2"
    assert response.tools[1].tool_type == wil_pb2.ToolType.TOOL_TYPE_CODE_EXECUTOR
    assert response.tools[1].description == "Desc 2"
    mock_tool_registry.list_tools.assert_called_once()
    mock_context.abort.assert_not_called()

@pytest.mark.asyncio
async def test_list_available_tools_registry_error(wil_service: WorldInteractionLayerService, mock_tool_registry):
    """Test error handling when the tool registry fails."""
    mock_context = _make_async_context()
    mock_tool_registry.list_tools.side_effect = Exception("Registry connection failed")

    response = await wil_service.ListAvailableTools(wil_pb2.ListAvailableToolsRequest(), mock_context)

    assert len(response.tools) == 0
    mock_context.abort.assert_called_once_with(grpc.StatusCode.INTERNAL, "Failed to list tools")
