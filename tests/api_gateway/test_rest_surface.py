import pytest
from types import SimpleNamespace
try:
    from jarvis._vendor.common import schemas  # test import
except ImportError:
    pytest.skip("Module not vendored", allow_module_level=True)


import pytest
from common.protos import csc_pb2, knowledge_nexus_pb2

from src.api_gateway import main as api_main
from src.user_modeling_engine.service import UserModelingEngineService

pytestmark = pytest.mark.asyncio


class _DummyChannel:
    async def close(self):
        return None


class _FakeCSCStub:
    async def ProcessGoal(self, request):
        return csc_pb2.ProcessGoalResponse(
            goal_id="goal-abc",
            initial_status=csc_pb2.STATUS_PROCESSING,
        )

    async def GetGoalStatus(self, request):
        step = csc_pb2.PlanStepMessage(
            id="step-1",
            description="run generic process",
            action_type="GENERIC_PROCESS",
            status=csc_pb2.STATUS_COMPLETED,
        )
        plan = csc_pb2.PlanMessage(
            id="plan-1",
            goal_id=request.goal_id,
            steps=[step],
            status=csc_pb2.STATUS_COMPLETED,
            confidence=0.9,
            justification="ok",
        )
        goal = csc_pb2.GoalMessage(
            id=request.goal_id,
            status=csc_pb2.STATUS_COMPLETED,
        )
        return csc_pb2.GetGoalStatusResponse(goal=goal, current_plan=plan)


class _FakeKNStub:
    async def SemanticSearch(self, request):
        node = knowledge_nexus_pb2.KnowledgeNodeMessage(id="node-1", labels=["Fact"])
        return knowledge_nexus_pb2.SemanticSearchResponse(
            results=[
                knowledge_nexus_pb2.SemanticSearchResponse.Result(
                    node=node,
                    score=0.91,
                )
            ]
        )

    async def QueryNodes(self, request):
        return knowledge_nexus_pb2.QueryNodesResponse(
            nodes=[knowledge_nexus_pb2.KnowledgeNodeMessage(id="node-q", labels=["Doc"])]
        )

    async def IngestText(self, request):
        return knowledge_nexus_pb2.IngestTextResponse(primary_entity_id="entity-1")


@pytest.fixture
def _grpc_stubs(monkeypatch):
    monkeypatch.setattr(api_main.grpc.aio, "insecure_channel", lambda _: _DummyChannel())
    monkeypatch.setattr(
        api_main.csc_pb2_grpc,
        "CentralStrategicCoreServiceStub",
        lambda _: _FakeCSCStub(),
    )
    monkeypatch.setattr(
        api_main.knowledge_nexus_pb2_grpc,
        "KnowledgeNexusServiceStub",
        lambda _: _FakeKNStub(),
    )
    api_main.TASKS.clear()


async def test_tasks_endpoints_round_trip(_grpc_stubs):
    created = await api_main.create_task(
        {"goal": "ship public beast", "user_context": {"user_id": "u-1"}}
    )
    assert created["goalId"] == "goal-abc"
    assert created["status"] == "STATUS_PROCESSING"

    status = await api_main.get_task(created["taskId"])
    assert status["goalId"] == "goal-abc"
    assert status["status"] == "STATUS_COMPLETED"
    assert status["plan"]["status"] == "STATUS_COMPLETED"

    results = await api_main.get_task_results(created["taskId"])
    assert results["goalId"] == "goal-abc"
    assert results["status"] == "STATUS_COMPLETED"

    intervention = await api_main.intervene_task(created["taskId"], {"action": "pause"})
    assert intervention["ok"] is True
    assert len(intervention["interventions"]) == 1


async def test_knowledge_endpoints_semantic_search_and_ingest(_grpc_stubs):
    searched = await api_main.get_knowledge(query_text="long horizon", top_k=3)
    assert "results" in searched
    assert len(searched["results"]) == 1
    assert searched["results"][0]["node"]["id"] == "node-1"

    ingested = await api_main.post_knowledge(
        {
            "text": "Prometheus note",
            "source": "unit-test",
            "layer": "runtime",
            "metadata": {"tag": "test"},
        }
    )
    assert ingested["primaryEntityId"] == "entity-1"


async def test_user_model_endpoints_round_trip(tmp_path, _grpc_stubs, monkeypatch):
    monkeypatch.setattr(
        api_main,
        "UME",
        UserModelingEngineService(data_path=str(tmp_path / "ume")),
    )
    request = SimpleNamespace(state=SimpleNamespace(user={"sub": "user-42"}))

    updated = await api_main.put_user_model(
        {
            "preferences": {"risk": "yolo"},
            "goals": ["launch"],
            "knowledge_areas": {"python": "advanced"},
        },
        request=request,
    )
    assert updated["user_id"] == "user-42"
    assert updated["preferences"]["risk"] == "yolo"
    assert updated["goals"] == ["launch"]

    profile = await api_main.get_user_model(request=request)
    assert profile["user_id"] == "user-42"
    assert profile["knowledge_areas"]["python"] == "advanced"
