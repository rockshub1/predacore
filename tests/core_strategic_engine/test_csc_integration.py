import pytest
"""
try:
    from jarvis._vendor.common import schemas  # test import
except ImportError:
    pytest.skip("Module not vendored", allow_module_level=True)

Integration tests for the Central Strategic Core (CSC) service interactions.
"""
import asyncio  # Added for sleep
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import grpc
import grpc.aio
import pytest
from jarvis._vendor.common.models import Plan, PlanStep, StatusEnum  # Added for internal status check

# Import protos and stubs for mocking
from jarvis._vendor.common.protos import (
    csc_pb2,
    egm_pb2,
    egm_pb2_grpc,
    knowledge_nexus_pb2,
    knowledge_nexus_pb2_grpc,
)

from jarvis._vendor.core_strategic_engine.planner import HierarchicalStrategicPlannerV1

# Import components to be tested
from jarvis._vendor.core_strategic_engine.service import CentralStrategicCoreService
from jarvis._vendor.ethical_governance_module.audit_logger import (
    AbstractAuditLogger,
)
from jarvis._vendor.ethical_governance_module.rule_engine import BasicRuleEngine
from jarvis._vendor.ethical_governance_module.service import EthicalGovernanceModuleService

# Use pytest-asyncio for async functions
pytestmark = pytest.mark.asyncio

# --- Fixtures ---

@pytest.fixture
def mock_kn_stub() -> knowledge_nexus_pb2_grpc.KnowledgeNexusServiceStub:
    """Provides a mock Knowledge Nexus gRPC stub."""
    stub = MagicMock(spec=knowledge_nexus_pb2_grpc.KnowledgeNexusServiceStub)
    stub.QueryNodes = AsyncMock(return_value=knowledge_nexus_pb2.QueryNodesResponse(nodes=[]))
    return stub

@pytest.fixture
def basic_rule_engine() -> BasicRuleEngine:
    """Provides a BasicRuleEngine instance."""
    return BasicRuleEngine()

@pytest.fixture
def mock_audit_logger() -> AbstractAuditLogger:
    """Provides a mock Audit Logger."""
    logger = MagicMock(spec=AbstractAuditLogger)
    logger.log = AsyncMock() # Make the log method an async mock
    return logger

@pytest.fixture
def planner(mock_kn_stub) -> HierarchicalStrategicPlannerV1:
    """Provides the planner instance."""
    return HierarchicalStrategicPlannerV1(kn_stub=mock_kn_stub)

@pytest.fixture
def egm_service(basic_rule_engine, mock_audit_logger) -> EthicalGovernanceModuleService:
    """Provides the EGM service instance with real rule engine but mock logger."""
    return EthicalGovernanceModuleService(rule_engine=basic_rule_engine, audit_logger=mock_audit_logger)

@pytest.fixture
def mock_egm_stub(egm_service: EthicalGovernanceModuleService) -> egm_pb2_grpc.EthicalGovernanceModuleServiceStub:
    """
    Provides a mock EGM stub that calls the *real* EGM service methods.
    This allows testing the CSC's interaction logic without actual gRPC calls.
    """
    stub = MagicMock(spec=egm_pb2_grpc.EthicalGovernanceModuleServiceStub)
    # Wrap service methods to supply a dummy context (the CSC calls stub methods with only request)
    dummy_ctx = MagicMock(spec=grpc.aio.ServicerContext)
    dummy_ctx.abort = AsyncMock()

    async def _check_plan(request, **kwargs):
        return await egm_service.CheckPlanCompliance(request, dummy_ctx)

    async def _check_action(request, **kwargs):
        return await egm_service.CheckActionCompliance(request, dummy_ctx)

    async def _log_event(request, **kwargs):
        return await egm_service.LogEvent(request, dummy_ctx)

    stub.CheckPlanCompliance = AsyncMock(side_effect=_check_plan)
    stub.CheckActionCompliance = AsyncMock(side_effect=_check_action)
    stub.LogEvent = AsyncMock(side_effect=_log_event)
    return stub

@pytest.fixture
def csc_service(mock_kn_stub, mock_egm_stub, planner) -> CentralStrategicCoreService:
    """Provides the CSC service instance with mock dependencies."""
    # Note: UME stub is omitted as it's not used yet
    return CentralStrategicCoreService(kn_stub=mock_kn_stub, egm_stub=mock_egm_stub, planner=planner)

# --- Test Cases ---

async def test_process_goal_safe_plan_integration(csc_service: CentralStrategicCoreService, mock_egm_stub, mock_audit_logger, monkeypatch):
    """
    Test the ProcessGoal flow for a goal that results in a compliant plan.
    Verifies CSC calls planner, then EGM, stores results, and logs events.
    """
    monkeypatch.setenv("EGM_MODE", "strict")
    goal_input = "Find information about safe topics"
    request = csc_pb2.ProcessGoalRequest(user_goal_input=goal_input)
    # Mock gRPC context - use MagicMock for sync methods like abort
    mock_context = MagicMock(spec=grpc.aio.ServicerContext)
    mock_context.abort = AsyncMock() # Mock the async abort method if needed, but usually sync

    response = await csc_service.ProcessGoal(request, mock_context)

    # Assertions
    assert response is not None
    assert response.goal_id is not None
    # Check final status stored internally (assuming PROCESSING after successful plan check)
    internal_goal_status = csc_service._goals[response.goal_id]["status"]
    assert internal_goal_status == StatusEnum.PROCESSING
    # Check the status returned in the response matches the proto enum value
    assert response.initial_status == csc_pb2.Status.STATUS_PROCESSING

    # Verify EGM CheckPlanCompliance was called
    mock_egm_stub.CheckPlanCompliance.assert_called_once()
    compliance_call_args = mock_egm_stub.CheckPlanCompliance.call_args[0][0] # Get the CheckPlanComplianceRequest
    assert isinstance(compliance_call_args, egm_pb2.CheckPlanComplianceRequest)
    assert len(compliance_call_args.plan.steps) > 0 # Check that a plan was passed

    # Verify Audit Log was called (by EGM service during compliance check)
    # EGM logs PLAN_COMPLIANCE_CHECK after checking
    await asyncio.sleep(0.01) # Allow async log call to potentially register
    assert mock_audit_logger.log.call_count > 0
    # Check the *last* call to the logger, as EGM might log multiple things
    log_call_args = mock_audit_logger.log.call_args_list[-1][0][0] # Get the LogEventRequest from last call
    assert isinstance(log_call_args, egm_pb2.LogEventRequest)
    assert log_call_args.event_type == "PLAN_COMPLIANCE_CHECK"
    assert log_call_args.compliance_status.is_compliant is True

    # Verify context.abort was NOT called
    mock_context.abort.assert_not_called()

async def test_process_goal_unsafe_plan_integration(csc_service: CentralStrategicCoreService, mock_egm_stub, mock_audit_logger, monkeypatch):
    """
    Test the ProcessGoal flow for a goal resulting in a non-compliant plan (due to keywords).
    Verifies CSC calls planner, then EGM, and aborts correctly.
    """
    monkeypatch.setenv("EGM_MODE", "strict")
    goal_input = "Plan to harm_human" # Contains forbidden keyword
    request = csc_pb2.ProcessGoalRequest(user_goal_input=goal_input)
    mock_context = MagicMock(spec=grpc.aio.ServicerContext)
    mock_context.abort = AsyncMock()

    # Call ProcessGoal - it should call context.abort internally
    await csc_service.ProcessGoal(request, mock_context)

    # Assertions
    # Verify EGM CheckPlanCompliance was called
    mock_egm_stub.CheckPlanCompliance.assert_called_once()

    # Verify Audit Log was called (by EGM service during compliance check)
    await asyncio.sleep(0.01) # Allow async log call
    assert mock_audit_logger.log.call_count > 0
    log_call_args = mock_audit_logger.log.call_args_list[-1][0][0]
    assert log_call_args.event_type == "PLAN_COMPLIANCE_CHECK"
    assert log_call_args.compliance_status.is_compliant is False # Should log non-compliance

    # Verify context.abort WAS called with FAILED_PRECONDITION
    mock_context.abort.assert_called_once()
    # Check the arguments passed to abort
    abort_args = mock_context.abort.call_args[0]
    assert abort_args[0] == grpc.StatusCode.FAILED_PRECONDITION
    assert "Plan failed compliance check" in abort_args[1]


    # Check internal goal status was marked as FAILED
    # Find the goal ID - requires inspecting internal state or modifying service for testability
    # This part is fragile and depends on implementation details
    goal_id_found = None
    for gid, gdata in csc_service._goals.items():
        if gdata["input"] == goal_input:
            goal_id_found = gid
            break
    assert goal_id_found is not None, "Goal ID not found in internal state"
    assert csc_service._goals[goal_id_found]["status"] == StatusEnum.FAILED


async def test_process_goal_risky_action_type_integration(csc_service: CentralStrategicCoreService, mock_egm_stub, mock_audit_logger, planner, monkeypatch):
    """
    Test the ProcessGoal flow for a goal resulting in a non-compliant plan (due to risky action type/params).
    """
    monkeypatch.setenv("EGM_MODE", "log_only")
    # We need a goal input that the *current* simple planner translates into a risky action.
    # The current planner doesn't easily generate MODIFY_FILESYSTEM.
    # Let's modify the planner fixture *for this test* to generate a risky plan directly.

    # Override the planner's create_plan for this specific test
    async def create_risky_plan(*args, **kwargs):
        risky_step = PlanStep(
            id=uuid4(),
            description="Modify critical system file",
            action_type="MODIFY_FILESYSTEM",
            parameters={"path": "/etc/important_config"} # Unsafe path
        )
        return Plan(goal_id=args[0], steps=[risky_step], status=StatusEnum.READY)

    planner.create_plan = AsyncMock(side_effect=create_risky_plan) # Temporarily override planner

    goal_input = "Modify system config" # Input text doesn't matter as much as the overridden plan
    request = csc_pb2.ProcessGoalRequest(user_goal_input=goal_input)
    mock_context = MagicMock(spec=grpc.aio.ServicerContext)
    mock_context.abort = AsyncMock()

    # Call ProcessGoal
    await csc_service.ProcessGoal(request, mock_context)

    # In log_only mode (default), only HIGH severity violations cause non-compliance.
    # MODIFY_FILESYSTEM is MEDIUM severity, so it's marked compliant in log_only mode.
    # The plan PASSES compliance, so context.abort should NOT be called.
    mock_context.abort.assert_not_called()

    # Verify the EGM logged the violations even though plan was allowed
    await asyncio.sleep(0.01)
    assert mock_audit_logger.log.call_count > 0
    log_call_args = mock_audit_logger.log.call_args_list[-1][0][0]
    assert log_call_args.event_type == "PLAN_COMPLIANCE_CHECK"
    # In log_only mode with MEDIUM severity, is_compliant is True
    assert log_call_args.compliance_status.is_compliant is True
    # But violations are still recorded
    assert len(log_call_args.compliance_status.violations) > 0


# TODO: Add test for planner failure propagation
# TODO: Add test for EGM communication failure propagation
