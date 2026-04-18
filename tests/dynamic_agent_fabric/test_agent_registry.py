"""
Unit tests for the DAF Agent Registry implementations.
"""
from uuid import uuid4

import pytest
from predacore._vendor.common.protos import daf_pb2  # For AgentStatus enum

from predacore.agents.daf.agent_registry import (
    ActiveAgentInstanceRegistry,
    StaticAgentTypeRegistry,
)

# --- Fixtures ---

@pytest.fixture
def type_registry() -> StaticAgentTypeRegistry:
    """Provides a StaticAgentTypeRegistry instance."""
    return StaticAgentTypeRegistry()

@pytest.fixture
def instance_registry() -> ActiveAgentInstanceRegistry:
    """Provides an ActiveAgentInstanceRegistry instance."""
    return ActiveAgentInstanceRegistry()

# --- Tests for StaticAgentTypeRegistry ---

def test_list_agent_types(type_registry: StaticAgentTypeRegistry):
    """Test listing all agent types."""
    agent_types = type_registry.list_agent_types()
    assert len(agent_types) >= 15
    agent_ids = {a["agent_type_id"] for a in agent_types}
    expected_ids = {
        "web_searcher", "python_executor", "web_scraper", "weather_fetcher",
        "wiki_summarizer", "advanced_scraper", "python_lint_agent",
        "python_formatter_agent", "translator_agent", "doc_summarizer_agent",
        "email_agent", "slack_bot_agent", "discord_bot_agent",
        "pdf_parser_agent", "image_analysis_agent", "calendar_agent"
    }
    assert expected_ids.issubset(agent_ids)

def test_get_agent_type_exists(type_registry: StaticAgentTypeRegistry):
    """Test retrieving an existing agent type."""
    agent_type = type_registry.get_agent_type("slack_bot_agent")
    assert agent_type is not None
    assert agent_type["agent_type_id"] == "slack_bot_agent"
    assert "slack_bot" in agent_type["required_tools"]

def test_get_agent_type_not_exists(type_registry: StaticAgentTypeRegistry):
    """Test retrieving a non-existent agent type."""
    agent_type = type_registry.get_agent_type("nonexistent_agent")
    assert agent_type is None

def test_find_agent_for_capability_exact_match(type_registry: StaticAgentTypeRegistry):
    """Test finding an agent by an exactly matching capability."""
    agent_id = type_registry.find_agent_for_capability("send_email")
    assert agent_id == "email_agent"

def test_find_agent_for_capability_case_insensitive(type_registry: StaticAgentTypeRegistry):
    """Test finding an agent by capability, ignoring case."""
    agent_id = type_registry.find_agent_for_capability("Summarize_Document")
    assert agent_id == "doc_summarizer_agent"

def test_find_agent_for_capability_no_match(type_registry: StaticAgentTypeRegistry):
    """Test finding an agent when no capability matches."""
    agent_id = type_registry.find_agent_for_capability("make_coffee")
    assert agent_id is None

# --- Tests for ActiveAgentInstanceRegistry ---
# Module-level pytestmark would also tag the sync tests above, so each async
# test is marked individually instead.


@pytest.mark.asyncio
async def test_register_instance(instance_registry: ActiveAgentInstanceRegistry):
    """Test registering a new agent instance."""
    instance_id = str(uuid4())
    type_id = "test_agent"
    config = {"key": "value"}
    instance_info = await instance_registry.register_instance(instance_id, type_id, config)

    assert instance_info["instance_id"] == instance_id
    assert instance_info["type_id"] == type_id
    assert instance_info["status"] == daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_IDLE
    assert instance_info["config"] == config
    assert instance_info["current_task_info"] is None

    # Verify it's stored
    stored_instance = await instance_registry.get_instance(instance_id)
    assert stored_instance == instance_info

@pytest.mark.asyncio
async def test_register_existing_instance(instance_registry: ActiveAgentInstanceRegistry):
    """Test registering an instance that already exists."""
    instance_id = str(uuid4())
    await instance_registry.register_instance(instance_id, "type1", {})
    # Register again
    instance_info = await instance_registry.register_instance(instance_id, "type2", {"new": "config"})
    # Should return the original registration
    assert instance_info["type_id"] == "type1"
    assert instance_info["config"] == {}

@pytest.mark.asyncio
async def test_unregister_instance(instance_registry: ActiveAgentInstanceRegistry):
    """Test unregistering an existing instance."""
    instance_id = str(uuid4())
    await instance_registry.register_instance(instance_id, "test", {})
    assert await instance_registry.get_instance(instance_id) is not None
    unregistered = await instance_registry.unregister_instance(instance_id)
    assert unregistered is True
    assert await instance_registry.get_instance(instance_id) is None

@pytest.mark.asyncio
async def test_unregister_nonexistent_instance(instance_registry: ActiveAgentInstanceRegistry):
    """Test unregistering an instance that doesn't exist."""
    unregistered = await instance_registry.unregister_instance(str(uuid4()))
    assert unregistered is False

@pytest.mark.asyncio
async def test_list_instances(instance_registry: ActiveAgentInstanceRegistry):
    """Test listing active instances, optionally filtered by type."""
    id1 = str(uuid4())
    id2 = str(uuid4())
    id3 = str(uuid4())
    await instance_registry.register_instance(id1, "typeA", {})
    await instance_registry.register_instance(id2, "typeB", {})
    await instance_registry.register_instance(id3, "typeA", {})

    all_instances = await instance_registry.list_instances()
    assert len(all_instances) == 3
    instance_ids = {inst["instance_id"] for inst in all_instances}
    assert instance_ids == {id1, id2, id3}

    typeA_instances = await instance_registry.list_instances(type_id="typeA")
    assert len(typeA_instances) == 2
    typeA_ids = {inst["instance_id"] for inst in typeA_instances}
    assert typeA_ids == {id1, id3}

    typeC_instances = await instance_registry.list_instances(type_id="typeC")
    assert len(typeC_instances) == 0

@pytest.mark.asyncio
async def test_update_instance_status(instance_registry: ActiveAgentInstanceRegistry):
    """Test updating the status and task info of an instance."""
    instance_id = str(uuid4())
    await instance_registry.register_instance(instance_id, "test", {})
    instance = await instance_registry.get_instance(instance_id)
    assert instance["status"] == daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_IDLE

    task_info = {"task_id": "task123"}
    updated = await instance_registry.update_instance_status(
        instance_id, daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_BUSY, task_info
    )
    assert updated is True
    instance = await instance_registry.get_instance(instance_id)
    assert instance["status"] == daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_BUSY
    assert instance["current_task_info"] == task_info

    updated_nonexistent = await instance_registry.update_instance_status(
        "nonexistent", daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_ERROR
    )
    assert updated_nonexistent is False

@pytest.mark.asyncio
async def test_find_idle_instance(instance_registry: ActiveAgentInstanceRegistry):
    """Test finding an idle instance of a specific type."""
    id_idle_A = str(uuid4())
    id_busy_A = str(uuid4())
    id_idle_B = str(uuid4())
    await instance_registry.register_instance(id_idle_A, "typeA", {})
    await instance_registry.register_instance(id_busy_A, "typeA", {})
    await instance_registry.register_instance(id_idle_B, "typeB", {})
    await instance_registry.update_instance_status(id_busy_A, daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_BUSY)

    found_A = await instance_registry.find_idle_instance("typeA")
    assert found_A == id_idle_A

    found_B = await instance_registry.find_idle_instance("typeB")
    assert found_B == id_idle_B

    found_C = await instance_registry.find_idle_instance("typeC")
    assert found_C is None
