"""Patterns — Anthropic's six + Self-MoA. Each is a strategy class."""
from .autonomous import AutonomousPattern
from .base import Pattern, PatternResult
from .evaluator_optimizer import EvaluatorOptimizerPattern
from .orchestrator_workers import OrchestratorWorkersPattern
from .parallelize import ParallelizePattern
from .prompt_chain import ChainStep, PromptChainPattern
from .routing import Route, RoutingPattern
from .self_moa import SelfMoAPattern

__all__ = [
    "AutonomousPattern",
    "ChainStep",
    "EvaluatorOptimizerPattern",
    "OrchestratorWorkersPattern",
    "ParallelizePattern",
    "Pattern",
    "PatternResult",
    "PromptChainPattern",
    "Route",
    "RoutingPattern",
    "SelfMoAPattern",
]
