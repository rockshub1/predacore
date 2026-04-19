"""
MCTS Planner Enhancements — companion module for ABMCTSPlanner.

Provides a proper Monte-Carlo Tree Search implementation with:
- UCB1 node selection
- Expansion with branching
- Simulation (rollout)
- Backpropagation

This module is a drop-in enhancement for the existing ABMCTSPlanner
in planner_mcts.py. It provides the search tree data structures and
algorithms without modifying the existing 583-line module.
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SearchConfig:
    """Configuration for the MCTS search."""

    exploration_constant: float = 1.414  # UCB1 √2
    max_iterations: int = 100
    max_depth: int = 10
    time_budget_seconds: float = 5.0
    branch_factor: int = 3  # Max children per node
    min_visits_for_expansion: int = 1
    discount_factor: float = 0.95  # Future reward discount


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------


@dataclass
class MCTSNode:
    """A node in the MCTS search tree."""

    node_id: str = ""
    state: Any = None  # Planner state (e.g., partial plan)
    action: str = ""  # Action that led to this node
    parent: MCTSNode | None = None
    children: list[MCTSNode] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    depth: int = 0
    is_terminal: bool = False

    def __post_init__(self) -> None:
        if not self.node_id:
            self.node_id = uuid4().hex[:12]

    @property
    def value(self) -> float:
        """Average value of this node."""
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def ucb1(self, c: float = 1.414) -> float:
        """Upper Confidence Bound 1 score."""
        if self.visits == 0:
            return float("inf")
        exploitation = self.value
        parent_visits = self.parent.visits if self.parent else 1
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    @property
    def is_fully_expanded(self) -> bool:
        """Check if all possible children have been generated."""
        # Heuristic: consider fully expanded if we have branch_factor children
        branch_factor = getattr(self, '_branch_factor', 3)
        return len(self.children) >= branch_factor

    def set_branch_factor(self, factor: int) -> None:
        """Set the branch factor for expansion check."""
        self._branch_factor = factor

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def best_child(self, c: float = 1.414) -> MCTSNode | None:
        """Return child with highest UCB1 score."""
        if not self.children:
            return None
        return max(self.children, key=lambda n: n.ucb1(c))

    def add_child(
        self,
        state: Any,
        action: str = "",
    ) -> MCTSNode:
        """Create and add a child node."""
        child = MCTSNode(
            state=state,
            action=action,
            parent=self,
            depth=self.depth + 1,
        )
        self.children.append(child)
        return child


# ---------------------------------------------------------------------------
# MCTS Tree
# ---------------------------------------------------------------------------


class MCTSTree:
    """
    Full Monte-Carlo Tree Search implementation.

    Usage:
        tree = MCTSTree(config)
        tree.set_root(initial_state)
        tree.search(expand_fn, simulate_fn)
        best = tree.best_action()
    """

    def __init__(
        self,
        config: SearchConfig | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self.config = config or SearchConfig()
        self._root: MCTSNode | None = None
        self._log = log or logger
        self._iterations_run = 0

    def set_root(self, state: Any) -> MCTSNode:
        """Initialize the tree with a root state."""
        self._root = MCTSNode(state=state, depth=0)
        self._root.set_branch_factor(self.config.branch_factor)
        self._iterations_run = 0
        return self._root

    @property
    def root(self) -> MCTSNode | None:
        return self._root

    @property
    def iterations_run(self) -> int:
        return self._iterations_run

    # -- Main search loop ---------------------------------------------------

    def search(
        self,
        expand_fn: Callable[[Any], list[tuple[Any, str]]],
        simulate_fn: Callable[[Any], float],
    ) -> MCTSNode:
        """
        Run MCTS for the configured number of iterations or time budget.

        Args:
            expand_fn: Given a state, returns list of (child_state, action) pairs
            simulate_fn: Given a state, returns a value estimate (0.0–1.0)

        Returns:
            The root node (with tree built).
        """
        if not self._root:
            raise ValueError("Call set_root() before search()")

        c = self.config
        t0 = time.monotonic()

        for i in range(c.max_iterations):
            # Time budget check
            if (time.monotonic() - t0) > c.time_budget_seconds:
                self._log.debug("MCTS: time budget exhausted at iteration %d", i)
                break

            # 1. SELECT
            node = self._select(self._root)

            # 2. EXPAND
            if not node.is_terminal and node.depth < c.max_depth:
                node = self._expand(node, expand_fn)

            # 3. SIMULATE
            value = simulate_fn(node.state)

            # 4. BACKPROPAGATE
            self._backpropagate(node, value)

            self._iterations_run = i + 1

        return self._root

    # -- MCTS phases --------------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCB1 tree policy."""
        c = self.config.exploration_constant
        while not node.is_leaf and not node.is_terminal:
            if not node.is_fully_expanded:
                return node  # Expand this node
            best = node.best_child(c)
            if best is None:
                return node
            node = best
        return node

    def _expand(
        self,
        node: MCTSNode,
        expand_fn: Callable[[Any], list[tuple[Any, str]]],
    ) -> MCTSNode:
        """Expand a node by generating children."""
        if node.visits < self.config.min_visits_for_expansion:
            return node

        children_specs = expand_fn(node.state)
        if not children_specs:
            node.is_terminal = True
            return node

        # Limit to branch factor
        children_specs = children_specs[: self.config.branch_factor]
        for state, action in children_specs:
            node.add_child(state=state, action=action)

        # Return first new child for simulation
        return node.children[-1] if node.children else node

    @staticmethod
    def _backpropagate(node: MCTSNode, value: float) -> None:
        """Propagate the simulation value back up the tree."""
        current: MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent

    # -- Result extraction --------------------------------------------------

    def best_action(self) -> str | None:
        """Return the action with the most visits from the root."""
        if not self._root or not self._root.children:
            return None
        best = max(self._root.children, key=lambda n: n.visits)
        return best.action

    def best_child_node(self) -> MCTSNode | None:
        """Return the best child of root by visits."""
        if not self._root or not self._root.children:
            return None
        return max(self._root.children, key=lambda n: n.visits)

    def get_statistics(self) -> dict[str, Any]:
        """Return tree statistics."""
        total_nodes = self._count_nodes(self._root) if self._root else 0
        max_depth = self._max_depth(self._root) if self._root else 0
        return {
            "iterations": self._iterations_run,
            "total_nodes": total_nodes,
            "max_depth": max_depth,
            "root_visits": self._root.visits if self._root else 0,
            "root_value": self._root.value if self._root else 0,
            "children_count": len(self._root.children) if self._root else 0,
        }

    def _count_nodes(self, node: MCTSNode | None) -> int:
        if node is None:
            return 0
        return 1 + sum(self._count_nodes(c) for c in node.children)

    def _max_depth(self, node: MCTSNode | None) -> int:
        if node is None or not node.children:
            return node.depth if node else 0
        return max(self._max_depth(c) for c in node.children)


# ---------------------------------------------------------------------------
# Plan ranker
# ---------------------------------------------------------------------------


@dataclass
class PlanCandidate:
    """A candidate plan with multi-criteria scores."""

    plan_id: str = ""
    plan_data: Any = None
    scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.plan_id:
            self.plan_id = uuid4().hex[:8]

    @property
    def overall_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)


class PlanRanker:
    """
    Ranks plan candidates using multi-criteria scoring.

    Criteria:
    - cost: computational/API cost estimate
    - risk: EGM risk assessment
    - latency: estimated execution time
    - quality: expected output quality
    """

    DEFAULT_WEIGHTS = {
        "cost": 0.2,
        "risk": 0.3,
        "latency": 0.2,
        "quality": 0.3,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.weights = weights or self.DEFAULT_WEIGHTS

    def rank(
        self,
        candidates: list[PlanCandidate],
    ) -> list[PlanCandidate]:
        """Rank candidates by weighted overall score (highest first)."""
        for candidate in candidates:
            candidate.scores["weighted"] = self._weighted_score(candidate.scores)
        return sorted(
            candidates, key=lambda c: c.scores.get("weighted", 0), reverse=True
        )

    def _weighted_score(self, scores: dict[str, float]) -> float:
        total = 0.0
        weight_sum = 0.0
        for criterion, weight in self.weights.items():
            if criterion in scores:
                total += scores[criterion] * weight
                weight_sum += weight
        return total / weight_sum if weight_sum > 0 else 0.0

    def compare(
        self,
        plan_a: PlanCandidate,
        plan_b: PlanCandidate,
    ) -> str:
        """Return which plan is better: 'a', 'b', or 'tie'."""
        sa = self._weighted_score(plan_a.scores)
        sb = self._weighted_score(plan_b.scores)
        if abs(sa - sb) < 0.01:
            return "tie"
        return "a" if sa > sb else "b"
