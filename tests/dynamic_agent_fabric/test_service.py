"""
Integration tests for DynamicAgentFabricControllerService.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest
from predacore._vendor.common.protos import daf_pb2, wil_pb2
from google.protobuf.struct_pb2 import Struct, Value

from predacore.agents.daf.agent_registry import (
    ActiveAgentInstanceRegistry,
    StaticAgentTypeRegistry,
)
from predacore.agents.daf.service import DynamicAgentFabricControllerService


@pytest.fixture
def mock_wil_stub():
    stub = MagicMock()
    stub.ExecuteTool = AsyncMock(return_value=wil_pb2.InteractionResultMessage(
        status=wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS,
        output=Value()
    ))
    stub.ExecuteCode = AsyncMock(return_value=wil_pb2.CodeExecutionResultMessage(
        status=wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS,
        stdout="",
        result=Value()
    ))
    return stub

@pytest.fixture
def daf_service(mock_wil_stub):
    type_registry = StaticAgentTypeRegistry()
    instance_registry = ActiveAgentInstanceRegistry()
    return DynamicAgentFabricControllerService(
        agent_type_registry=type_registry,
        agent_instance_registry=instance_registry,
        wil_stub=mock_wil_stub
    )

@pytest.mark.asyncio
async def test_spawn_and_retire_agent(daf_service):
    # Test spawning a python_executor agent
    spawn_request = daf_pb2.SpawnAgentRequest(agent_type_id="python_executor")
    spawn_response = await daf_service.SpawnAgent(spawn_request, None)

    assert spawn_response.success
    assert spawn_response.agent_instance.agent_type_id == "python_executor"
    assert spawn_response.agent_instance.status == daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_IDLE

    # Test retiring the agent
    retire_request = daf_pb2.RetireAgentRequest(agent_instance_id=spawn_response.agent_instance.agent_instance_id)
    retire_response = await daf_service.RetireAgent(retire_request, None)
    assert retire_response.success

@pytest.mark.asyncio
async def test_task_dispatch(daf_service, mock_wil_stub):
    # Spawn an agent first
    spawn_request = daf_pb2.SpawnAgentRequest(agent_type_id="python_executor")
    spawn_response = await daf_service.SpawnAgent(spawn_request, None)

    # Create a task assignment
    task_params = Struct()
    task_params.update({"code": "print('Hello')"})
    task_request = daf_pb2.TaskAssignmentMessage(
        agent_instance_id=spawn_response.agent_instance.agent_instance_id,
        parameters=task_params
    )

    # Dispatch the task
    task_result = await daf_service.DispatchTask(task_request, None)
    assert task_result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS
    mock_wil_stub.ExecuteCode.assert_called_once()

@pytest.mark.asyncio
async def test_agent_auto_spawn(daf_service, mock_wil_stub):
    # Request a task requiring a python_executor without specifying instance
    task_params = Struct()
    task_params.update({"code": "print('Hello')"})
    task_request = daf_pb2.TaskAssignmentMessage(
        agent_type_id="python_executor",
        parameters=task_params
    )

    # Dispatch should auto-spawn an instance
    task_result = await daf_service.DispatchTask(task_request, None)
    assert task_result.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_SUCCESS
    mock_wil_stub.ExecuteCode.assert_called_once()

@pytest.mark.asyncio
async def test_agent_heartbeat(daf_service):
    # Spawn an agent
    spawn_request = daf_pb2.SpawnAgentRequest(agent_type_id="python_executor")
    spawn_response = await daf_service.SpawnAgent(spawn_request, None)
    instance_id = spawn_response.agent_instance.agent_instance_id

    # Send heartbeat
    heartbeat_request = daf_pb2.AgentHeartbeatMessage(
        agent_instance_id=instance_id
    )
    heartbeat_response = await daf_service.AgentHeartbeat(heartbeat_request, None)
    assert heartbeat_response.success

@pytest.mark.asyncio
async def test_process_cleanup(daf_service):
    # Spawn multiple agents
    spawn_request = daf_pb2.SpawnAgentRequest(agent_type_id="python_executor")
    spawn_response1 = await daf_service.SpawnAgent(spawn_request, None)
    spawn_response2 = await daf_service.SpawnAgent(spawn_request, None)

    # Verify processes are tracked
    assert len(daf_service._agent_processes) == 2

    # Retire one agent
    retire_request = daf_pb2.RetireAgentRequest(agent_instance_id=spawn_response1.agent_instance.agent_instance_id)
    await daf_service.RetireAgent(retire_request, None)
    assert len(daf_service._agent_processes) == 1

if __name__ == '__main__':
    pytest.main()
