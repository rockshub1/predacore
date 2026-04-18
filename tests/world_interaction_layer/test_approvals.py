import pytest

try:
    from predacore._vendor.common.protos import egm_pb2_grpc, wil_pb2
    from predacore._vendor.world_interaction_layer.service import (
        AbstractSandboxManager,
        AbstractToolRegistry,
        WorldInteractionLayerService,
    )
except ImportError:
    pytest.skip("world_interaction_layer not available in _vendor", allow_module_level=True)

from unittest.mock import MagicMock

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _policy_env_defaults(monkeypatch):
    """Ensure approval-mode tests are not affected by global env mutations."""
    monkeypatch.setenv("EGM_MODE", "strict")
    monkeypatch.setenv("APPROVALS_REQUIRED", "1")


@pytest.fixture
def wil_service():
    tool_registry = MagicMock(spec=AbstractToolRegistry)
    sandbox = MagicMock(spec=AbstractSandboxManager)
    egm_stub = MagicMock(spec=egm_pb2_grpc.EthicalGovernanceModuleServiceStub)
    service = WorldInteractionLayerService(tool_registry, sandbox, egm_stub, metrics_registry=None)
    return service


async def test_execute_code_requires_approval(wil_service):
    req = wil_pb2.CodeExecutionRequestMessage(
        request_id="r1",
        code="print('hi')",
    )
    res = await wil_service.ExecuteCode(req, None)
    assert res.status == wil_pb2.InteractionStatus.INTERACTION_STATUS_FAILED
    assert "approval_required" in res.error_message
