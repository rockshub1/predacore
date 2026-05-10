"""Runners — pluggable execution backends for one AgentSpec."""
from .base import AgentResult, RunContext, Runner
from .daf import DAFRunner
from .in_process import InProcessRunner

__all__ = [
    "AgentResult",
    "DAFRunner",
    "InProcessRunner",
    "RunContext",
    "Runner",
]
