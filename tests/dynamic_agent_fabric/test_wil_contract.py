import pytest

try:
    from jarvis.agents.daf.agent_registry import StaticAgentTypeRegistry
except ImportError:
    pytest.skip("jarvis.agents.daf.agent_registry not available", allow_module_level=True)

try:
    from jarvis._vendor.world_interaction_layer.tool_registry import SimpleToolRegistry
except ImportError:
    pytest.skip("world_interaction_layer.tool_registry not available in _vendor", allow_module_level=True)


def test_agent_required_tools_exist_in_wil_registry():
    type_registry = StaticAgentTypeRegistry()
    tool_registry = SimpleToolRegistry()
    available_tool_ids = {tool["tool_id"] for tool in tool_registry.list_tools()}

    missing_by_agent = {}
    for agent in type_registry.list_agent_types():
        required = [
            tool_id
            for tool_id in agent.get("required_tools", [])
            if tool_id not in available_tool_ids
        ]
        if required:
            missing_by_agent[agent["agent_type_id"]] = required

    assert not missing_by_agent, (
        "DAF->WIL tool contract mismatch. Missing required tools: "
        f"{missing_by_agent}"
    )
