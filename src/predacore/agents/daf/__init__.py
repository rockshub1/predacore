"""
Dynamic Agent Fabric (DAF) package exports.

Keep imports lazy so lightweight modules like the task store or registries do
not force-load the gRPC service stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AbstractAgentRegistry": ".agent_registry",
    "AbstractAgentTypeRegistry": ".agent_registry",
    "AbstractAgentInstanceRegistry": ".agent_registry",
    "ActiveAgentInstanceRegistry": ".agent_registry",
    "SimpleAgentRegistry": ".agent_registry",
    "StaticAgentTypeRegistry": ".agent_registry",
    "health_status": ".health",
    "DynamicAgentFabricControllerService": ".service",
    "AbstractTaskStore": ".task_store",
    "MemoryTaskStore": ".task_store",
    "RedisTaskStore": ".task_store",
    "get_task_store": ".task_store",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value

