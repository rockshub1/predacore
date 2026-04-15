"""
Tests for Phase 4 — Differentiation Features.

Covers all 6 modules:
1. FAISSVectorIndex + HashEmbedding
2. PersistentAuditStore
3. AgentCollaboration (fan-out, pipeline, consensus, supervisor)
4. SWE-bench EvalRunner + EvalScorer
5. ALE-bench AgentEvalRunner + AgentScorer
6. MCTS Tree + PlanRanker
"""
import asyncio
import os
import tempfile
import unittest


def _run(coro):
    """Helper to run async code in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ===== 1. FAISS Vector Index ================================================

class TestFAISSVectorIndex(unittest.TestCase):

    def test_add_and_search(self):
        from jarvis._vendor.knowledge_nexus.faiss_vector_index import FAISSVectorIndex

        idx = FAISSVectorIndex(dimensions=4)
        _run(idx.add("a", [1.0, 0.0, 0.0, 0.0]))
        _run(idx.add("b", [0.0, 1.0, 0.0, 0.0]))
        _run(idx.add("c", [1.0, 0.1, 0.0, 0.0]))

        results = _run(idx.search_similar([1.0, 0.0, 0.0, 0.0], top_k=2))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "a")  # Exact match first
        self.assertGreater(results[0][1], 0.9)

    def test_dimension_mismatch(self):
        from jarvis._vendor.knowledge_nexus.faiss_vector_index import FAISSVectorIndex

        idx = FAISSVectorIndex(dimensions=4)
        with self.assertRaises(ValueError):
            _run(idx.add("x", [1.0, 0.0]))

    def test_remove(self):
        from jarvis._vendor.knowledge_nexus.faiss_vector_index import FAISSVectorIndex

        idx = FAISSVectorIndex(dimensions=2)
        _run(idx.add("a", [1.0, 0.0]))
        self.assertEqual(idx.size, 1)
        _run(idx.remove("a"))
        self.assertEqual(idx.size, 0)

    def test_layer_filtering(self):
        from jarvis._vendor.knowledge_nexus.faiss_vector_index import FAISSVectorIndex

        idx = FAISSVectorIndex(dimensions=2)
        _run(idx.add("a", [1.0, 0.0], {"layers": ["code"]}))
        _run(idx.add("b", [0.9, 0.1], {"layers": ["docs"]}))

        results = _run(idx.search_similar([1.0, 0.0], layers={"code"}))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "a")

    def test_persistence(self):
        from jarvis._vendor.knowledge_nexus.faiss_vector_index import FAISSVectorIndex

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            idx = FAISSVectorIndex(dimensions=3)
            _run(idx.add("x", [1.0, 0.5, 0.0]))
            _run(idx.add("y", [0.0, 1.0, 0.5]))
            _run(idx.save_to_disk(path))

            loaded = _run(FAISSVectorIndex.load_from_disk(path))
            self.assertEqual(loaded.size, 2)
            results = _run(loaded.search_similar([1.0, 0.5, 0.0], top_k=1))
            self.assertEqual(results[0][0], "x")
        finally:
            os.unlink(path)

    def test_hash_embedding(self):
        from jarvis._vendor.knowledge_nexus.faiss_vector_index import HashEmbedding

        emb = HashEmbedding(dims=64)
        v1 = emb.embed("hello world")
        v2 = emb.embed("hello world")
        v3 = emb.embed("goodbye moon")

        self.assertEqual(len(v1), 64)
        self.assertEqual(v1, v2)  # Deterministic
        self.assertNotEqual(v1, v3)

    def test_hash_embedding_normalized(self):
        import math

        from jarvis._vendor.knowledge_nexus.faiss_vector_index import HashEmbedding

        emb = HashEmbedding(dims=32)
        v = emb.embed("test text")
        norm = math.sqrt(sum(x * x for x in v))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_metadata(self):
        from jarvis._vendor.knowledge_nexus.faiss_vector_index import FAISSVectorIndex

        idx = FAISSVectorIndex(dimensions=2)
        _run(idx.add("a", [1.0, 0.0], {"label": "test"}))
        meta = idx.get_metadata("a")
        self.assertEqual(meta["label"], "test")

    def test_update(self):
        from jarvis._vendor.knowledge_nexus.faiss_vector_index import FAISSVectorIndex

        idx = FAISSVectorIndex(dimensions=2)
        _run(idx.add("a", [1.0, 0.0]))
        _run(idx.update("a", [0.0, 1.0]))
        results = _run(idx.search_similar([0.0, 1.0], top_k=1))
        self.assertEqual(results[0][0], "a")
        self.assertGreater(results[0][1], 0.9)


# ===== 2. Persistent Audit Store ============================================

class TestPersistentAuditStore(unittest.TestCase):

    def setUp(self):
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmpfile.close()
        self.db_path = self._tmpfile.name

    def tearDown(self):
        os.unlink(self.db_path)

    def _store(self):
        from jarvis._vendor.ethical_governance_module.persistent_audit import PersistentAuditStore
        return PersistentAuditStore(db_path=self.db_path)

    def test_log_and_query(self):
        store = self._store()
        eid = store.log_decision(
            component="planner",
            event_type="plan_check",
            is_compliant=True,
            decision="ALLOW",
        )
        self.assertGreater(eid, 0)
        entries = store.query(component="planner")
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0].is_compliant)
        store.close()

    def test_query_violations(self):
        store = self._store()
        store.log_decision("a", "check", True)
        store.log_decision("b", "check", False, severity="WARNING")
        store.log_decision("c", "check", False, severity="CRITICAL")

        violations = store.query(is_compliant=False)
        self.assertEqual(len(violations), 2)
        store.close()

    def test_statistics(self):
        store = self._store()
        store.log_decision("a", "check", True)
        store.log_decision("b", "check", True)
        store.log_decision("c", "check", False, principle="safety")

        stats = store.get_statistics()
        self.assertEqual(stats["total_entries"], 3)
        self.assertEqual(stats["violations"], 1)
        self.assertAlmostEqual(stats["compliance_rate"], 66.7, places=0)
        self.assertEqual(stats["violations_by_principle"]["safety"], 1)
        store.close()

    def test_export_report(self):
        store = self._store()
        store.log_decision("x", "check", False, justification="blocked")
        report = store.export_report()
        self.assertIn("EGM Audit Report", report)
        self.assertIn("blocked", report)
        store.close()

    def test_query_by_severity(self):
        store = self._store()
        store.log_decision("a", "check", True, severity="INFO")
        store.log_decision("b", "check", False, severity="CRITICAL")

        critical = store.query(severity="CRITICAL")
        self.assertEqual(len(critical), 1)
        self.assertEqual(critical[0].severity, "CRITICAL")
        store.close()

    def test_entry_count(self):
        store = self._store()
        self.assertEqual(store.entry_count, 0)
        store.log_decision("a", "check", True)
        store.log_decision("b", "check", True)
        self.assertEqual(store.entry_count, 2)
        store.close()


# ===== 3. Agent Collaboration ===============================================

class TestAgentCollaboration(unittest.TestCase):

    def test_fan_out(self):
        from jarvis.agents.collaboration import (
            AgentSpec,
            AgentTeam,
            TaskPayload,
        )

        team = AgentTeam()

        async def agent_a(task: TaskPayload):
            return "answer_a"

        async def agent_b(task: TaskPayload):
            return "answer_b"

        team.add_agent(AgentSpec(agent_id="a"), agent_a)
        team.add_agent(AgentSpec(agent_id="b"), agent_b)

        result = _run(team.fan_out(TaskPayload(prompt="test")))
        self.assertEqual(result.success_count, 2)
        self.assertIn("answer_a", result.final_output)
        self.assertIn("answer_b", result.final_output)

    def test_pipeline(self):
        from jarvis.agents.collaboration import (
            AgentSpec,
            AgentTeam,
            TaskPayload,
        )

        team = AgentTeam()

        async def upper(task: TaskPayload):
            return task.prompt.upper()

        async def add_excl(task: TaskPayload):
            return task.prompt + "!"

        team.add_agent(AgentSpec(agent_id="upper"), upper)
        team.add_agent(AgentSpec(agent_id="excl"), add_excl)

        result = _run(team.pipeline(
            TaskPayload(prompt="hello"),
            order=["upper", "excl"],
        ))
        self.assertEqual(result.final_output, "HELLO!")

    def test_consensus(self):
        from jarvis.agents.collaboration import (
            AgentSpec,
            AgentTeam,
            TaskPayload,
        )

        team = AgentTeam()

        async def yes_agent(task: TaskPayload):
            return "yes"

        async def no_agent(task: TaskPayload):
            return "no"

        team.add_agent(AgentSpec(agent_id="a1"), yes_agent)
        team.add_agent(AgentSpec(agent_id="a2"), yes_agent)
        team.add_agent(AgentSpec(agent_id="a3"), no_agent)

        result = _run(team.consensus(TaskPayload(prompt="vote"), quorum=0.5))
        self.assertTrue(result.consensus_reached)
        self.assertEqual(result.final_output, "yes")

    def test_consensus_not_reached(self):
        from jarvis.agents.collaboration import (
            AgentSpec,
            AgentTeam,
            TaskPayload,
        )

        team = AgentTeam()

        async def unique_a(t): return "alpha"
        async def unique_b(t): return "beta"
        async def unique_c(t): return "gamma"

        team.add_agent(AgentSpec(agent_id="a"), unique_a)
        team.add_agent(AgentSpec(agent_id="b"), unique_b)
        team.add_agent(AgentSpec(agent_id="c"), unique_c)

        result = _run(team.consensus(TaskPayload(prompt="x"), quorum=0.9))
        self.assertFalse(result.consensus_reached)

    def test_supervisor(self):
        from jarvis.agents.collaboration import (
            AgentSpec,
            AgentTeam,
            TaskPayload,
        )

        team = AgentTeam()

        async def worker(task: TaskPayload):
            return "draft answer"

        async def reviewer(task: TaskPayload):
            return "reviewed: " + task.prompt.split("\n")[-1]

        team.add_agent(AgentSpec(agent_id="worker"), worker)
        team.add_agent(AgentSpec(agent_id="reviewer"), reviewer)

        result = _run(team.supervise(
            TaskPayload(prompt="question"),
            worker_id="worker",
            supervisor_id="reviewer",
        ))
        self.assertIn("reviewed:", str(result.final_output))
        self.assertEqual(len(result.results), 2)

    def test_error_handling(self):
        from jarvis.agents.collaboration import (
            AgentSpec,
            AgentTeam,
            TaskPayload,
        )

        team = AgentTeam()

        async def fail_agent(task: TaskPayload):
            raise RuntimeError("broken")

        async def ok_agent(task: TaskPayload):
            return "ok"

        team.add_agent(AgentSpec(agent_id="fail"), fail_agent)
        team.add_agent(AgentSpec(agent_id="ok"), ok_agent)

        result = _run(team.fan_out(TaskPayload(prompt="test")))
        self.assertEqual(result.success_count, 1)
        self.assertEqual(result.error_count, 1)

    def test_result_aggregator(self):
        from jarvis.agents.collaboration import AgentResult, ResultAggregator

        results = [
            AgentResult(agent_id="a", output="yes", success=True),
            AgentResult(agent_id="b", output="yes", success=True),
            AgentResult(agent_id="c", output="no", success=True),
        ]
        self.assertEqual(ResultAggregator.majority_vote(results), "yes")

    def test_task_distributor(self):
        from jarvis.agents.collaboration import (
            AgentSpec,
            TaskDistributor,
            TaskPayload,
        )

        agents = [AgentSpec(agent_id="a"), AgentSpec(agent_id="b")]
        task = TaskPayload(prompt="test task")
        assignments = TaskDistributor.replicate(task, agents)
        self.assertEqual(len(assignments), 2)


# ===== 4. SWE-bench Eval ===================================================

class TestSWEBenchEval(unittest.TestCase):

    def test_exact_match_scorer(self):
        from jarvis.evals.swe_bench import EvalScorer

        self.assertEqual(EvalScorer.exact_match("hello", "hello"), 1.0)
        self.assertEqual(EvalScorer.exact_match("hello", "world"), 0.0)

    def test_fuzzy_match_scorer(self):
        from jarvis.evals.swe_bench import EvalScorer

        score = EvalScorer.fuzzy_match("hello world", "hello world!")
        self.assertGreater(score, 0.8)

    def test_line_match_scorer(self):
        from jarvis.evals.swe_bench import EvalScorer

        output = "line1\nline2\nline3"
        expected = "line1\nline2"
        score = EvalScorer.line_match(output, expected)
        self.assertEqual(score, 1.0)

    def test_diff_summary(self):
        from jarvis.evals.swe_bench import EvalScorer

        diff = EvalScorer.diff_summary("hello", "world")
        self.assertIn("---", diff)

    def test_eval_runner_pass(self):
        from jarvis.evals.swe_bench import EvalRunner, EvalTask

        async def mock_agent(prompt: str) -> str:
            return "def fizzbuzz(n): pass"

        runner = EvalRunner(scoring_method="fuzzy", pass_threshold=0.3)
        task = EvalTask(
            task_id="test1",
            prompt="Write fizzbuzz",
            expected_output="def fizzbuzz(n): pass",
        )
        result = _run(runner.run_task(task, mock_agent))
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.score, 1.0)

    def test_eval_runner_fail(self):
        from jarvis.evals.swe_bench import EvalRunner, EvalTask

        async def bad_agent(prompt: str) -> str:
            return "completely wrong"

        runner = EvalRunner(scoring_method="exact", pass_threshold=1.0)
        task = EvalTask(
            task_id="test2",
            prompt="Write fizzbuzz",
            expected_output="def fizzbuzz(n): pass",
        )
        result = _run(runner.run_task(task, bad_agent))
        self.assertFalse(result.passed)

    def test_eval_suite(self):
        from jarvis.evals.swe_bench import EvalRunner, EvalTask

        async def echo_agent(prompt: str) -> str:
            return prompt

        runner = EvalRunner(scoring_method="fuzzy", pass_threshold=0.5)
        tasks = [
            EvalTask(task_id="t1", prompt="hello", expected_output="hello"),
            EvalTask(task_id="t2", prompt="world", expected_output="world"),
        ]
        report = _run(runner.run_suite(tasks, echo_agent, suite_name="test"))
        self.assertEqual(report.total, 2)
        self.assertEqual(report.passed, 2)
        self.assertAlmostEqual(report.pass_rate, 1.0)

    def test_eval_report_to_dict(self):
        from jarvis.evals.swe_bench import EvalReport

        report = EvalReport(suite_name="x")
        d = report.to_dict()
        self.assertEqual(d["suite_name"], "x")
        self.assertEqual(d["total"], 0)

    def test_sample_tasks_exist(self):
        from jarvis.evals.swe_bench import SAMPLE_TASKS

        self.assertGreaterEqual(len(SAMPLE_TASKS), 3)
        self.assertEqual(SAMPLE_TASKS[0].task_id, "fizzbuzz")


# ===== 5. ALE-bench Eval ===================================================

class TestALEBenchEval(unittest.TestCase):

    def test_tool_accuracy(self):
        from jarvis.evals.ale_bench import AgentScorer

        score = AgentScorer.tool_accuracy(
            used=["create_file", "read_file"],
            expected=["create_file", "read_file"],
        )
        self.assertEqual(score, 1.0)

    def test_tool_accuracy_partial(self):
        from jarvis.evals.ale_bench import AgentScorer

        score = AgentScorer.tool_accuracy(
            used=["create_file"],
            expected=["create_file", "read_file"],
        )
        self.assertEqual(score, 0.5)

    def test_plan_quality_ideal(self):
        from jarvis.evals.ale_bench import AgentAction, AgentScorer

        actions = [AgentAction(step=i) for i in range(3)]
        score = AgentScorer.plan_quality(actions, max_steps=10)
        self.assertEqual(score, 1.0)

    def test_plan_quality_at_max(self):
        from jarvis.evals.ale_bench import AgentAction, AgentScorer

        actions = [AgentAction(step=i) for i in range(10)]
        score = AgentScorer.plan_quality(actions, max_steps=10)
        self.assertEqual(score, 0.0)

    def test_error_recovery_no_errors(self):
        from jarvis.evals.ale_bench import AgentAction, AgentScorer

        actions = [AgentAction(step=i) for i in range(3)]
        self.assertEqual(AgentScorer.error_recovery(actions), 1.0)

    def test_error_recovery_with_recovery(self):
        from jarvis.evals.ale_bench import AgentAction, AgentScorer

        actions = [
            AgentAction(step=0, error="oops"),    # Error
            AgentAction(step=1),                    # Recovery
        ]
        self.assertEqual(AgentScorer.error_recovery(actions), 1.0)

    def test_outcome_match(self):
        from jarvis.evals.ale_bench import AgentScorer

        score = AgentScorer.outcome_match(
            "Paris is the capital of France",
            "Paris capital France",
        )
        self.assertEqual(score, 1.0)

    def test_scenario_runner(self):
        from jarvis.evals.ale_bench import (
            AgentAction,
            AgentEvalRunner,
            AgentScenario,
            CapabilityCategory,
        )

        async def mock_agent(prompt):
            return (
                [AgentAction(step=0, tool_name="create_file")],
                "file created successfully",
            )

        runner = AgentEvalRunner(pass_threshold=0.5)
        scenario = AgentScenario(
            scenario_id="test",
            category=CapabilityCategory.TOOL_USE,
            prompt="Create a file",
            expected_tools=["create_file"],
            expected_outcome="file created successfully",
            max_steps=5,
        )
        result = _run(runner.run_scenario(scenario, mock_agent))
        self.assertTrue(result.passed)
        self.assertGreater(result.scores["overall"], 0.5)

    def test_scenario_suite(self):
        from jarvis.evals.ale_bench import (
            AgentAction,
            AgentEvalRunner,
            AgentScenario,
            CapabilityCategory,
        )

        async def mock_agent(prompt):
            return ([AgentAction(step=0)], "done")

        runner = AgentEvalRunner(pass_threshold=0.3)
        scenarios = [
            AgentScenario(scenario_id="s1", prompt="test", expected_outcome="done",
                          category=CapabilityCategory.REASONING),
        ]
        report = _run(runner.run_suite(scenarios, mock_agent))
        self.assertEqual(report.total, 1)
        d = report.to_dict()
        self.assertIn("by_category", d)

    def test_sample_scenarios_exist(self):
        from jarvis.evals.ale_bench import SAMPLE_SCENARIOS

        self.assertGreaterEqual(len(SAMPLE_SCENARIOS), 4)


# ===== 6. MCTS Planner Enhancements ========================================

class TestMCTSEnhancements(unittest.TestCase):

    def test_mcts_node_ucb1(self):
        from jarvis._vendor.core_strategic_engine.planner_enhancements import MCTSNode

        parent = MCTSNode(state="root", visits=10, total_value=5.0)
        child = MCTSNode(state="child", parent=parent, visits=3, total_value=1.5)
        parent.children.append(child)

        ucb = child.ucb1(c=1.414)
        self.assertGreater(ucb, 0)

    def test_mcts_node_unvisited_infinite(self):
        from jarvis._vendor.core_strategic_engine.planner_enhancements import MCTSNode

        node = MCTSNode(state="new", visits=0)
        self.assertEqual(node.ucb1(), float("inf"))

    def test_mcts_node_add_child(self):
        from jarvis._vendor.core_strategic_engine.planner_enhancements import MCTSNode

        root = MCTSNode(state="root")
        child = root.add_child(state="child", action="move_a")
        self.assertEqual(len(root.children), 1)
        self.assertEqual(child.parent, root)
        self.assertEqual(child.depth, 1)

    def test_mcts_tree_search(self):
        from jarvis._vendor.core_strategic_engine.planner_enhancements import (
            MCTSTree,
            SearchConfig,
        )

        config = SearchConfig(
            max_iterations=50,
            time_budget_seconds=2.0,
            branch_factor=2,
        )
        tree = MCTSTree(config=config)
        tree.set_root("initial_state")

        def expand_fn(state):
            if state == "initial_state":
                return [("state_a", "action_a"), ("state_b", "action_b")]
            return []

        def simulate_fn(state):
            if state == "state_a":
                return 0.8
            return 0.2

        tree.search(expand_fn, simulate_fn)

        stats = tree.get_statistics()
        self.assertGreater(stats["iterations"], 0)
        self.assertGreater(stats["total_nodes"], 1)
        best = tree.best_action()
        self.assertEqual(best, "action_a")

    def test_mcts_tree_best_child(self):
        from jarvis._vendor.core_strategic_engine.planner_enhancements import MCTSTree

        tree = MCTSTree()
        root = tree.set_root("root")

        child_a = root.add_child("a", "go_a")
        child_a.visits = 10
        child_a.total_value = 8.0

        child_b = root.add_child("b", "go_b")
        child_b.visits = 10
        child_b.total_value = 3.0

        best = tree.best_child_node()
        self.assertEqual(best.action, "go_a")

    def test_plan_ranker(self):
        from jarvis._vendor.core_strategic_engine.planner_enhancements import (
            PlanCandidate,
            PlanRanker,
        )

        ranker = PlanRanker()
        candidates = [
            PlanCandidate(
                plan_id="plan_a",
                scores={"cost": 0.9, "risk": 0.8, "latency": 0.7, "quality": 0.9},
            ),
            PlanCandidate(
                plan_id="plan_b",
                scores={"cost": 0.5, "risk": 0.3, "latency": 0.4, "quality": 0.5},
            ),
        ]
        ranked = ranker.rank(candidates)
        self.assertEqual(ranked[0].plan_id, "plan_a")

    def test_plan_ranker_compare(self):
        from jarvis._vendor.core_strategic_engine.planner_enhancements import (
            PlanCandidate,
            PlanRanker,
        )

        ranker = PlanRanker()
        a = PlanCandidate(scores={"cost": 0.9, "quality": 0.9})
        b = PlanCandidate(scores={"cost": 0.1, "quality": 0.1})
        result = ranker.compare(a, b)
        self.assertEqual(result, "a")

    def test_plan_candidate_overall(self):
        from jarvis._vendor.core_strategic_engine.planner_enhancements import PlanCandidate

        p = PlanCandidate(scores={"a": 0.8, "b": 0.6})
        self.assertAlmostEqual(p.overall_score, 0.7)

    def test_search_config_defaults(self):
        from jarvis._vendor.core_strategic_engine.planner_enhancements import SearchConfig

        sc = SearchConfig()
        self.assertAlmostEqual(sc.exploration_constant, 1.414)
        self.assertEqual(sc.max_iterations, 100)
        self.assertEqual(sc.branch_factor, 3)

    def test_mcts_backpropagation(self):
        from jarvis._vendor.core_strategic_engine.planner_enhancements import MCTSNode, MCTSTree

        root = MCTSNode(state="root")
        child = root.add_child("child", "act")
        grandchild = child.add_child("gc", "act2")

        MCTSTree._backpropagate(grandchild, 0.7)
        self.assertEqual(grandchild.visits, 1)
        self.assertEqual(child.visits, 1)
        self.assertEqual(root.visits, 1)
        self.assertAlmostEqual(grandchild.value, 0.7)


if __name__ == "__main__":
    unittest.main()
