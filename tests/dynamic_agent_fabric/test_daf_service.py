"""
Unit tests for the Dynamic Agent Fabric (DAF) Controller service.
"""
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import grpc
import grpc.aio
import pytest
from predacore._vendor.common.protos import daf_pb2, egm_pb2_grpc, wil_pb2, wil_pb2_grpc
from google.protobuf.struct_pb2 import Struct, Value

from predacore.agents.daf.agent_registry import (
    AbstractAgentTypeRegistry as AbstractAgentRegistry,
)

# Import components to be tested
from predacore.agents.daf.service import DynamicAgentFabricControllerService

# Use pytest-asyncio for async functions
pytestmark = pytest.mark.asyncio

# --- Fixtures ---

@pytest.fixture
def mock_agent_registry() -> AbstractAgentRegistry:
    """Provides a mock Agent Registry."""
    registry = MagicMock(spec=AbstractAgentRegistry)
    # Define some dummy agent types
    registry._agents = {
        "web_searcher": {"agent_type_id": "web_searcher", "required_tools": ["google_search_api"]},
        "python_executor": {"agent_type_id": "python_executor", "required_tools": ["python_sandbox"]},
    }
    registry.get_agent_type = MagicMock(side_effect=lambda id: registry._agents.get(id))
    registry.find_agent_for_capability = MagicMock(side_effect=lambda cap: "web_searcher" if "search" in cap else ("python_executor" if "code" in cap else None))
    registry.list_agent_types = MagicMock(return_value=list(registry._agents.values()))
    return registry

@pytest.fixture
def mock_wil_stub() -> wil_pb2_grpc.WorldInteractionLayerServiceStub:
    """Provides a mock WIL gRPC stub."""
    stub = MagicMock(spec=wil_pb2_grpc.WorldInteractionLayerServiceStub)
    # Default successful responses
    stub.ExecuteTool = AsyncMock(return_value=wil_pb2.InteractionResultMessage(status=wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS))
    stub.ExecuteCode = AsyncMock(return_value=wil_pb2.CodeExecutionResultMessage(status=wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS))
    return stub

@pytest.fixture
def mock_egm_stub() -> egm_pb2_grpc.EthicalGovernanceModuleServiceStub:
    """Provides a mock EGM gRPC stub (optional)."""
    stub = MagicMock(spec=egm_pb2_grpc.EthicalGovernanceModuleServiceStub)
    return stub

@pytest.fixture
def mock_instance_registry():
    """Provides a mock instance registry with async methods."""
    registry = AsyncMock()
    # find_idle_instance returns an instance ID by default
    registry.find_idle_instance = AsyncMock(return_value="instance-001")
    registry.get_instance = AsyncMock(return_value=None)
    registry.update_instance_status = AsyncMock()
    return registry

@pytest.fixture
def daf_service(mock_agent_registry, mock_wil_stub, mock_egm_stub, mock_instance_registry) -> DynamicAgentFabricControllerService:
    """Provides a DAF service instance with mock dependencies."""
    return DynamicAgentFabricControllerService(
        agent_type_registry=mock_agent_registry,
        agent_instance_registry=mock_instance_registry,
        wil_stub=mock_wil_stub,
        egm_stub=mock_egm_stub
    )

# --- Test Cases for DispatchTask ---

async def test_dispatch_task_by_capability_success_tool(daf_service, mock_agent_registry, mock_wil_stub):
    """Test dispatching a task via capability to a WIL tool."""
    task_id = str(uuid4())
    plan_step_id = str(uuid4())
    params = Struct()
    params.update({"query": "test search"})
    request = daf_pb2.TaskAssignmentMessage(
        task_id=task_id,
        plan_step_id=plan_step_id,
        required_capability="web search",
        parameters=params
    )
    mock_context = AsyncMock(spec=grpc.aio.ServicerContext)

    # Mock WIL response
    mock_wil_output = Value(string_value="Search results here")
    mock_wil_stub.ExecuteTool.return_value = wil_pb2.InteractionResultMessage(
        request_id=task_id, status=wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS, output=mock_wil_output
    )

    result = await daf_service.DispatchTask(request, mock_context)

    # Assertions
    assert result.task_id == task_id
    assert result.plan_step_id == plan_step_id
    assert result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS
    assert result.output == mock_wil_output
    assert result.error_message == ""
    assert result.agent_type_id_used == "web_searcher"

    # Verify mocks
    mock_agent_registry.find_agent_for_capability.assert_called_once_with("web search")
    mock_wil_stub.ExecuteTool.assert_called_once()
    call_args = mock_wil_stub.ExecuteTool.call_args[0][0]
    assert isinstance(call_args, wil_pb2.InteractionRequestMessage)
    assert call_args.tool_id == "google_search_api"
    assert call_args.parameters == params

async def test_dispatch_task_by_agent_id_success_code(daf_service, mock_agent_registry, mock_wil_stub):
    """Test dispatching a task via agent_type_id to WIL code execution."""
    task_id = str(uuid4())
    plan_step_id = str(uuid4())
    code = "print('executing')"
    params = Struct()
    params.update({"code": code, "timeout_seconds": 10})
    request = daf_pb2.TaskAssignmentMessage(
        task_id=task_id,
        plan_step_id=plan_step_id,
        agent_type_id="python_executor",
        parameters=params
    )
    mock_context = AsyncMock(spec=grpc.aio.ServicerContext)

    # Mock WIL response
    mock_wil_output = Value(string_value="Done")
    mock_wil_stub.ExecuteCode.return_value = wil_pb2.CodeExecutionResultMessage(
        request_id=task_id, status=wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS, stdout="executing\n", result=mock_wil_output
    )

    result = await daf_service.DispatchTask(request, mock_context)

    # Assertions
    assert result.task_id == task_id
    assert result.plan_step_id == plan_step_id
    assert result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS
    assert result.output == mock_wil_output
    assert result.error_message == ""
    assert result.agent_type_id_used == "python_executor"

    # Verify mocks
    mock_agent_registry.get_agent_type.assert_called_once_with("python_executor")
    mock_wil_stub.ExecuteCode.assert_called_once()
    call_args = mock_wil_stub.ExecuteCode.call_args[0][0]
    assert isinstance(call_args, wil_pb2.CodeExecutionRequestMessage)
    assert call_args.code == code
    assert call_args.timeout_seconds == 10

async def test_dispatch_task_no_agent_for_capability(daf_service, mock_agent_registry):
    """Test dispatching when no agent supports the required capability."""
    request = daf_pb2.TaskAssignmentMessage(required_capability="unknown_capability")
    mock_context = AsyncMock(spec=grpc.aio.ServicerContext)
    mock_agent_registry.find_agent_for_capability.return_value = None

    result = await daf_service.DispatchTask(request, mock_context)

    # The service catches ValueError and returns ERROR status
    assert result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_ERROR
    assert "No agent type found for capability" in result.error_message

async def test_dispatch_task_agent_not_in_registry(daf_service, mock_agent_registry):
    """Test dispatching when the specified agent type ID doesn't exist."""
    request = daf_pb2.TaskAssignmentMessage(agent_type_id="nonexistent_agent")
    mock_context = AsyncMock(spec=grpc.aio.ServicerContext)
    mock_agent_registry.get_agent_type.return_value = None

    result = await daf_service.DispatchTask(request, mock_context)

    assert result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_ERROR
    assert "Agent type 'nonexistent_agent' not found" in result.error_message

async def test_dispatch_task_no_tool_mapping(daf_service, mock_agent_registry):
    """Test dispatching when the agent type has no associated tool."""
    agent_id = "agent_no_tool"
    mock_agent_registry._agents[agent_id] = {"agent_type_id": agent_id, "required_tools": []}
    mock_agent_registry.get_agent_type = MagicMock(side_effect=lambda id: mock_agent_registry._agents.get(id))

    request = daf_pb2.TaskAssignmentMessage(agent_type_id=agent_id)
    mock_context = AsyncMock(spec=grpc.aio.ServicerContext)

    result = await daf_service.DispatchTask(request, mock_context)

    assert result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_ERROR
    assert "No tool mapping for agent" in result.error_message

async def test_dispatch_task_wil_fails(daf_service, mock_agent_registry, mock_wil_stub):
    """Test dispatching when the WIL call fails."""
    request = daf_pb2.TaskAssignmentMessage(agent_type_id="web_searcher", parameters={})
    mock_context = AsyncMock(spec=grpc.aio.ServicerContext)

    # Mock WIL failure
    mock_wil_stub.ExecuteTool.return_value = wil_pb2.InteractionResultMessage(
        status=wil_pb2.InteractionStatus.INTERACTION_STATUS_FAILED, error_message="API unavailable"
    )

    result = await daf_service.DispatchTask(request, mock_context)

    assert result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_FAILED
    assert result.error_message == "API unavailable"

async def test_dispatch_task_wil_grpc_error(daf_service, mock_agent_registry, mock_wil_stub):
    """Test dispatching when the WIL call raises a gRPC error."""
    request = daf_pb2.TaskAssignmentMessage(agent_type_id="web_searcher", parameters={})
    mock_context = AsyncMock(spec=grpc.aio.ServicerContext)

    # Mock WIL gRPC error - AioRpcError needs code, initial_metadata, trailing_metadata
    mock_wil_stub.ExecuteTool.side_effect = grpc.aio.AioRpcError(
        code=grpc.StatusCode.UNAVAILABLE,
        initial_metadata=grpc.aio.Metadata(),
        trailing_metadata=grpc.aio.Metadata(),
        details="WIL service down"
    )

    result = await daf_service.DispatchTask(request, mock_context)

    assert result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_ERROR
    assert "WIL communication error" in result.error_message


async def test_get_next_task_prefers_instance_specific_queue(
    daf_service,
    mock_instance_registry,
):
    """Targeted pull-loop tasks should be delivered only to the intended instance."""
    task = daf_pb2.TaskAssignmentMessage(
        task_id="task-123",
        agent_instance_id="instance-001",
    )

    import asyncio as _asyncio

    daf_service._queues["instance:instance-001"] = _asyncio.Queue()
    await daf_service._queues["instance:instance-001"].put(task)

    async def _get_instance(instance_id: str):
        return {
            "instance_id": instance_id,
            "type_id": "web_searcher",
            "status": daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_IDLE,
        }

    mock_instance_registry.get_instance.side_effect = _get_instance

    wrong_resp = await daf_service.GetNextTask(
        daf_pb2.GetNextTaskRequest(agent_instance_id="instance-002"),
        AsyncMock(spec=grpc.aio.ServicerContext),
    )
    assert wrong_resp.has_task is False

    correct_resp = await daf_service.GetNextTask(
        daf_pb2.GetNextTaskRequest(agent_instance_id="instance-001"),
        AsyncMock(spec=grpc.aio.ServicerContext),
    )
    assert correct_resp.has_task is True
    assert correct_resp.task.task_id == "task-123"

# --- Test Cases for ListAgentTypes ---

async def test_list_agent_types_success(daf_service, mock_agent_registry):
    """Test successfully listing available agent types."""
    mock_context = AsyncMock(spec=grpc.aio.ServicerContext)
    mock_agents = [
        {"agent_type_id": "web_searcher", "description": "Searches web", "required_tools": ["google_search_api"]},
        {"agent_type_id": "python_executor", "description": "Runs code", "required_tools": ["python_sandbox"]},
    ]
    mock_agent_registry.list_agent_types.return_value = mock_agents

    response = await daf_service.ListAgentTypes(daf_pb2.ListAgentTypesRequest(), mock_context)

    assert len(response.agent_types) == 2
    assert response.agent_types[0].agent_type_id == "web_searcher"
    assert response.agent_types[0].description == "Searches web"
    assert list(response.agent_types[0].required_tools) == ["google_search_api"]
    assert response.agent_types[1].agent_type_id == "python_executor"
    assert response.agent_types[1].description == "Runs code"
    assert list(response.agent_types[1].required_tools) == ["python_sandbox"]

    mock_agent_registry.list_agent_types.assert_called_once()
    mock_context.abort.assert_not_called()

async def test_list_agent_types_registry_error(daf_service, mock_agent_registry):
    """Test error handling when the agent registry fails."""
    # context.abort is awaited, so it must be AsyncMock
    mock_context = AsyncMock(spec=grpc.aio.ServicerContext)
    mock_agent_registry.list_agent_types.side_effect = Exception("Registry DB connection failed")

    response = await daf_service.ListAgentTypes(daf_pb2.ListAgentTypesRequest(), mock_context)

    # Expect empty response and abort call
    assert len(response.agent_types) == 0
    mock_context.abort.assert_called_once_with(grpc.StatusCode.INTERNAL, "Failed to list agent types")
