"""
Prototype AB-MCTS planner for CSC.

Scaffold of a planner that can combine HTN structure with MCTS-style expansion
guided by GPT-5 and EGM gating. For now, it defers to the existing HTN planner
but provides a slot for adding deep search over alternatives at key choice points.
"""
from __future__ import annotations

import asyncio
import hashlib as _hashlib
import json as _json
import logging
import logging as _logging
import math
import os
from collections import OrderedDict
from typing import Any, Optional
from uuid import UUID, uuid4

from predacore._vendor.common.llm import default_params, get_default_llm_client
from predacore._vendor.common.logging_utils import log_json
from predacore._vendor.common.models import Plan, PlanStep, StatusEnum
from predacore._vendor.common.protos import egm_pb2, egm_pb2_grpc
from google.protobuf.struct_pb2 import Struct
from prometheus_client import Counter, Histogram

from .plan_cache import PlanMotifStore
from .planner import HierarchicalStrategicPlannerV1

try:
    from .planner_enhancements import PlanCandidate, PlanRanker
    _PLAN_RANKER_AVAILABLE = True
except Exception:  # noqa: BLE001 — graceful degradation
    _PLAN_RANKER_AVAILABLE = False


class ABMCTSPlanner:
    def __init__(
        self,
        kn_stub,
        egm_stub: egm_pb2_grpc.EthicalGovernanceModuleServiceStub | None = None,
        logger: logging.Logger | None = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        # Reuse HTN as a baseline; augment with MCTS expansions later
        self.baseline = HierarchicalStrategicPlannerV1(
            kn_stub=kn_stub, logger=self.logger
        )
        self.logger.info(
            "ABMCTSPlanner initialized (prototype; defers to HTN baseline)."
        )
        try:
            self.llm = get_default_llm_client(logger=self.logger)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"LLM client not initialized for MCTS scoring: {e}")
            self.llm = None
        self.egm_stub = egm_stub
        # Tunables
        try:
            self.max_depth = int(os.getenv("MCTS_MAX_DEPTH", "3"))
        except (ValueError, TypeError):
            self.max_depth = 3
        try:
            self.branch_factor = int(os.getenv("MCTS_BRANCHES", "3"))
        except (ValueError, TypeError):
            self.branch_factor = 3
        try:
            self.budget = int(os.getenv("MCTS_BUDGET", "16"))  # node expansions total
        except (ValueError, TypeError):
            self.budget = 16
        try:
            self.c_puct = float(os.getenv("MCTS_C_PUCT", "1.2"))
        except (ValueError, TypeError):
            self.c_puct = 1.2
        try:
            self.repair_radius = int(os.getenv("MCTS_REPAIR_RADIUS", "1"))
        except (ValueError, TypeError):
            self.repair_radius = 1
        try:
            self.parallel = max(1, int(os.getenv("MCTS_PARALLEL", "4")))
        except (ValueError, TypeError):
            self.parallel = 4
        try:
            self._score_ttl = max(0, int(os.getenv("MCTS_SCORE_TTL_SEC", "0")))
        except (ValueError, TypeError):
            self._score_ttl = 0
        # Score memoization (LRU)
        self._score_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._score_cache_ts: dict[str, float] = {}
        try:
            self._score_cache_max = int(os.getenv("MCTS_SCORE_CACHE_MAX", "2000"))
        except (ValueError, TypeError):
            self._score_cache_max = 2000
        # Domain risk priors
        self._low_risk_domains = (
            set((os.getenv("MCTS_LOW_RISK_DOMAINS", "") or "").lower().split(","))
            if os.getenv("MCTS_LOW_RISK_DOMAINS")
            else set()
        )
        self._high_risk_domains = (
            set((os.getenv("MCTS_HIGH_RISK_DOMAINS", "") or "").lower().split(","))
            if os.getenv("MCTS_HIGH_RISK_DOMAINS")
            else set()
        )
        # Weights for multi-objective scoring
        try:
            self.w_alpha = float(os.getenv("MCTS_W_ALPHA", "1.0"))
            self.w_beta = float(os.getenv("MCTS_W_BETA", "0.2"))
            self.w_gamma = float(os.getenv("MCTS_W_GAMMA", "0.1"))
        except (ValueError, TypeError):
            self.w_alpha, self.w_beta, self.w_gamma = 1.0, 0.2, 0.1
        # Plan motif cache
        self.plan_store = PlanMotifStore()
        # Metrics
        try:
            self._m_exp = Counter(
                "csc_mcts_expansions_total", "AB-MCTS node expansions"
            )
            self._h_best = Histogram(
                "csc_mcts_best_score", "AB-MCTS best score per improvement"
            )
            self._h_depth = Histogram("csc_mcts_depth", "AB-MCTS depths visited")
            self._h_branch = Histogram(
                "csc_mcts_branch_factor", "AB-MCTS children per expansion"
            )
        except Exception:  # noqa: BLE001
            self._m_exp = None
            self._h_best = None
            self._h_depth = None
            self._h_branch = None

    async def create_plan(
        self, goal_id: UUID, goal_input: str, user_context: dict[str, Any]
    ) -> Plan | None:
        # Step 1: Generate a baseline plan via HTN
        plan = await self.baseline.create_plan(goal_id, goal_input, user_context)
        if plan is None:
            return plan
        # Step 2: Score the plan with a quick self-consistency heuristic via GPT-5 or heuristics
        score = 0.5  # Default baseline score if scoring fails
        try:
            # Build a compact representation of the plan
            steps_brief = [
                f"{i+1}. {s.action_type}: {s.description[:120]}"
                for i, s in enumerate(plan.steps)
            ]
            score, just = await self._score_steps(goal_input, steps_brief)
            plan.justification = f"Initial score={score:.2f}: {just[:300]}"
            try:
                log_json(
                    self.logger,
                    _logging.INFO,
                    "csc.plan.score",
                    goal_id=str(goal_id),
                    score=round(score, 3),
                )
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            self.logger.debug(f"MCTS scoring failed: {e}")

        # Step 3: Run a bounded AB-MCTS-style expansion at key choice points
        try:
            improved = await self._mcts_improve(goal_id, goal_input, user_context, plan)
            if improved is not None:
                plan = improved
        except Exception as e:  # noqa: BLE001
            self.logger.debug(f"MCTS search failed (using baseline): {e}")

        # Step 4: Propose alternative branches and choose the best (fallback/simple alternate generation)
        try:
            k = self.branch_factor
            prompt = {
                "instruction": "Propose alternative step sequences to achieve the goal, focusing on efficiency and reliability. Output JSON: {candidates: [ {steps: [ {action_type, description, parameters} ]}]}",
                "goal": goal_input,
                "baseline_steps": steps_brief,
            }
            msgs = [
                {
                    "role": "system",
                    "content": "You generate alternative plans as strict JSON only.",
                },
                {"role": "user", "content": _json.dumps(prompt, ensure_ascii=False)},
            ]
            alt_raw = await self.llm.generate(
                msgs, params=default_params(temperature=0.4, max_tokens=800)
            )
            try:
                alt = _json.loads(alt_raw)
                cands: list[dict[str, Any]] = (alt.get("candidates") or [])[:k]
            except (_json.JSONDecodeError, ValueError):
                cands = []

            best = plan
            best_score = 0.0
            egm_mode = os.getenv("EGM_MODE", "off").lower()
            for cand in cands:
                steps_json = cand.get("steps") or []
                if not isinstance(steps_json, list):
                    continue
                # Build a Plan from candidate
                cand_steps: list[PlanStep] = []
                for s in steps_json:
                    try:
                        cand_steps.append(
                            PlanStep(
                                id=uuid4(),
                                description=str(s.get("description", ""))[:300],
                                action_type=str(
                                    s.get("action_type", "GENERIC_PROCESS")
                                )[:64],
                                parameters=s.get("parameters") or {},
                                status=StatusEnum.PENDING,
                            )
                        )
                    except Exception:  # noqa: BLE001
                        continue
                if not cand_steps:
                    continue
                cand_plan = Plan(
                    id=uuid4(),
                    goal_id=goal_id,
                    steps=cand_steps,
                    status=StatusEnum.READY,
                )
                # EGM compliance check if available
                compliant = True
                if self.egm_stub is not None:
                    try:
                        # Light check by sending a summary of the candidate plan
                        desc = Struct()
                        desc.update(
                            {
                                "action_type": "PLAN_CANDIDATE",
                                "summary": "; ".join(
                                    [s.description for s in cand_steps]
                                )[:1500],
                            }
                        )
                        req = egm_pb2.CheckActionComplianceRequest(
                            action_description=desc
                        )
                        res = await self.egm_stub.CheckActionCompliance(req)
                        if not res.is_compliant and egm_mode == "strict":
                            compliant = False
                    except Exception:  # noqa: BLE001
                        pass
                if not compliant:
                    continue
                # Score candidate
                try:
                    c_brief = [
                        f"{i+1}. {s.action_type}: {s.description[:120]}"
                        for i, s in enumerate(cand_steps)
                    ]
                    _sys_prompt = (
                        "You are an expert planning evaluator. Score the plan for the goal on a 0.0-1.0 scale; "
                        "higher is better for completeness, safety, and efficiency. Output JSON: {score: number, justification: string}."
                    )
                    msgs2 = [
                        {"role": "system", "content": _sys_prompt},
                        {
                            "role": "user",
                            "content": _json.dumps(
                                {"goal": goal_input, "steps": c_brief},
                                ensure_ascii=False,
                            ),
                        },
                    ]
                    score_raw = await self.llm.generate(
                        msgs2, params=default_params(temperature=0.2, max_tokens=200)
                    )
                    pr = _json.loads(score_raw)
                    sc = float(pr.get("score", 0.0))
                except Exception:  # noqa: BLE001
                    sc = 0.0
                if sc > best_score:
                    best = cand_plan
                    best_score = sc

            # Optional multi-criteria re-ranking via PlanRanker
            if _PLAN_RANKER_AVAILABLE and cands:
                try:
                    _ranker = PlanRanker()
                    _rank_cands: list[PlanCandidate] = []
                    # Include baseline plan as a candidate
                    _rank_cands.append(PlanCandidate(
                        plan_id="baseline", plan_data=plan,
                        scores={"quality": score if isinstance(score, (int, float)) else 0.5},
                    ))
                    for idx_c, cand in enumerate(cands):
                        _rank_cands.append(PlanCandidate(
                            plan_id=f"alt_{idx_c}", plan_data=cand,
                            scores={"quality": float(cand.get("_score", 0.0))},
                        ))
                    ranked = _ranker.rank(_rank_cands)
                    if ranked and ranked[0].plan_data is not plan:
                        self.logger.debug(
                            "PlanRanker re-ranked: top=%s weighted=%.3f",
                            ranked[0].plan_id,
                            ranked[0].scores.get("weighted", 0),
                        )
                except Exception as _rank_exc:  # noqa: BLE001
                    self.logger.debug("PlanRanker failed (using default): %s", _rank_exc)

            if best is not plan:
                best.justification = (
                    best.justification or ""
                ) + f"\nSelected by MCTS scoring {best_score:.2f}"
                plan = best
        except Exception as e:  # noqa: BLE001
            self.logger.debug(f"MCTS alternate generation failed: {e}")
        return plan

    # --- Internal helpers ---
    def _score_key(self, goal: str, steps_brief: list[str]) -> str:
        raw = (goal.strip() + "||" + "\n".join(steps_brief)).encode(
            "utf-8", errors="ignore"
        )
        return _hashlib.sha1(raw).hexdigest()

    async def _score_steps(
        self, goal: str, steps_brief: list[str]
    ) -> tuple[float, str]:
        """LLM- or heuristic-scored value for a sequence of steps. Returns (score, justification)."""
        # Memoization
        key = self._score_key(goal, steps_brief)
        if key in self._score_cache:
            s, j = self._score_cache.pop(key)
            # move to end (most-recent)
            self._score_cache[key] = (s, j)
            return s, j
        if self.llm is None:
            g = goal.lower()
            score = 0.2
            if any(k in g for k in ("edinet", "filing", "financial")):
                score += 0.3
            if any(k in g for k in ("scrape", "extract", "example.com")):
                score += 0.3
            if any(k in g for k in ("summarize", "answer", "rag")):
                score += 0.2
            s, j = min(1.0, score), "heuristic"
            # cache
            self._score_cache[key] = (s, j)
            if len(self._score_cache) > self._score_cache_max:
                self._score_cache.popitem(last=False)
            return s, j
        sys = (
            "You are an expert planning evaluator. Score the plan for the goal on a 0.0-1.0 scale; "
            "higher is better for completeness, safety, and efficiency. Output JSON: {score: number, justification: string}."
        )
        usr = {"goal": goal, "steps": steps_brief}
        msgs = [
            {"role": "system", "content": sys},
            {"role": "user", "content": _json.dumps(usr, ensure_ascii=False)},
        ]
        try:
            content = await self.llm.generate(
                msgs, params=default_params(temperature=0.2, max_tokens=250)
            )
            pr = _json.loads(content)
            s, j = float(pr.get("score", 0.0)), str(pr.get("justification", ""))
            self._score_cache[key] = (s, j)
            if len(self._score_cache) > self._score_cache_max:
                self._score_cache.popitem(last=False)
            return s, j
        except Exception:  # noqa: BLE001
            return 0.0, "score_failed"

    def _brief(self, steps: list[PlanStep]) -> list[str]:
        return [
            f"{i+1}. {s.action_type}: {s.description[:120]}"
            for i, s in enumerate(steps)
        ]

    def _is_choice_point(self, step: PlanStep) -> bool:
        # Heuristic: generic or underspecified steps are choice points
        if step.action_type in ("GENERIC_PROCESS", "SUMMARIZE_DATA"):
            return True
        if not step.parameters or len(step.parameters) == 0:
            return True
        return False

    def _segment_steps_by_intent(self, steps: list[PlanStep]) -> list[tuple[int, int]]:
        """Segment steps into intent-coherent spans using action_type/tool_id hints.
        Returns list of [start, end) index windows.
        """

        def cat(st: PlanStep) -> str:
            at = (st.action_type or "").upper()
            p = st.parameters or {}
            tool = str(p.get("tool_id", "")).lower()
            if "SUMMARIZE" in at:
                return "SUMMARIZE"
            if tool in ("rag_embed", "rag_retrieve", "rag_answer"):
                return "RAG"
            if tool in ("selector_extract", "browser_automation"):
                return "SCRAPE"
            if tool in ("edinet_fetch",):
                return "EDINET"
            if "KN" in at or "QUERY_KN" in at or "ADD_RELATION" in at:
                return "KN"
            if tool in ("python_sandbox", "node_executor", "go_executor"):
                return "CODE"
            return "GEN"

        segs: list[tuple[int, int]] = []
        if not steps:
            return segs
        cur_cat = cat(steps[0])
        start = 0
        for i in range(1, len(steps)):
            c = cat(steps[i])
            if c != cur_cat:
                segs.append((start, i))
                start = i
                cur_cat = c
        segs.append((start, len(steps)))
        return segs

    async def _expand(
        self, goal: str, base_steps: list[PlanStep]
    ) -> list[list[PlanStep]]:
        """Propose up to branch_factor alternative next-step sequences (local improvements)."""
        if self.llm is None:
            return self._heuristic_expand(goal, base_steps)
        steps_brief = self._brief(base_steps)
        prompt = {
            "instruction": "Improve the plan locally: rewrite, insert, or remove 1-3 steps to increase reliability/efficiency. Output JSON: {candidates: [ {steps: [ {action_type, description, parameters} ]} ]}",
            "goal": goal,
            "current_steps": steps_brief,
        }
        msgs = [
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": _json.dumps(prompt, ensure_ascii=False)},
        ]
        try:
            raw = await self.llm.generate(
                msgs, params=default_params(temperature=0.4, max_tokens=700)
            )
            obj = _json.loads(raw)
            cands = (obj.get("candidates") or [])[: self.branch_factor]
        except Exception:  # noqa: BLE001
            cands = []
        out: list[list[PlanStep]] = []
        for cand in cands:
            steps_json = cand.get("steps") or []
            seq: list[PlanStep] = []
            for s in steps_json:
                try:
                    seq.append(
                        PlanStep(
                            id=uuid4(),
                            description=str(s.get("description", ""))[:300],
                            action_type=str(s.get("action_type", "GENERIC_PROCESS"))[
                                :64
                            ],
                            parameters=s.get("parameters") or {},
                            status=StatusEnum.PENDING,
                        )
                    )
                except Exception:  # noqa: BLE001
                    continue
            if seq:
                out.append(seq)
        return out

    def _heuristic_expand(
        self, goal: str, base_steps: list[PlanStep]
    ) -> list[list[PlanStep]]:
        g = goal.lower()
        outs: list[list[PlanStep]] = []
        if any(k in g for k in ("edinet", "filing", "financial")):
            outs.append(
                [
                    PlanStep(
                        description="Fetch EDINET document",
                        action_type="INVOKE_TOOL",
                        parameters={
                            "tool_id": "edinet_fetch",
                            "url": "https://disclosure.edinet-fsa.go.jp/",
                        },
                    ),
                    PlanStep(
                        description="Embed content for retrieval",
                        action_type="INVOKE_TOOL",
                        parameters={
                            "tool_id": "rag_embed",
                            "namespace": "edinet",
                            "id": "doc1",
                            "text": "${fetched_text}",
                        },
                    ),
                    PlanStep(
                        description="Answer question with citations",
                        action_type="INVOKE_TOOL",
                        parameters={
                            "tool_id": "rag_answer",
                            "question": "Summarize key financials",
                            "context": [],
                        },
                    ),
                ]
            )
        if any(k in g for k in ("scrape", "extract", "example.com")):
            outs.append(
                [
                    PlanStep(
                        description="Extract items by CSS",
                        action_type="INVOKE_TOOL",
                        parameters={
                            "tool_id": "selector_extract",
                            "url": "https://example.com",
                            "selectors": [{"name": "title", "selector": "h1"}],
                            "max_pages": 1,
                            "export_format": "json",
                        },
                    ),
                    PlanStep(
                        description="Summarize extracted content",
                        action_type="SUMMARIZE_DATA",
                        parameters={"target_entity": "content"},
                    ),
                ]
            )
        if any(k in g for k in ("summarize", "rag", "answer")):
            outs.append(
                [
                    PlanStep(
                        description="Embed provided text into memory",
                        action_type="INVOKE_TOOL",
                        parameters={
                            "tool_id": "rag_embed",
                            "namespace": "default",
                            "id": "item1",
                            "text": "${input_text}",
                        },
                    ),
                    PlanStep(
                        description="Answer with citations",
                        action_type="INVOKE_TOOL",
                        parameters={
                            "tool_id": "rag_answer",
                            "question": "Provide a concise answer",
                            "context": [],
                        },
                    ),
                ]
            )
        return outs[: self.branch_factor]

    async def _egm_ok(self, steps: list[PlanStep]) -> bool:
        if self.egm_stub is None:
            return True
        try:
            desc = Struct()
            desc.update(
                {
                    "action_type": "PLAN_CANDIDATE",
                    "summary": "; ".join([s.description for s in steps])[:1500],
                }
            )
            req = egm_pb2.CheckActionComplianceRequest(action_description=desc)
            res = await self.egm_stub.CheckActionCompliance(req)
            egm_mode = os.getenv("EGM_MODE", "off").lower()
            return bool(res.is_compliant or egm_mode != "strict")
        except Exception:  # noqa: BLE001
            return True

    async def _mcts_improve(
        self, goal_id: UUID, goal_input: str, user_context: dict[str, Any], plan: Plan
    ) -> Plan | None:
        """Bounded AB-MCTS-like search that locally improves the plan.
        Returns improved Plan or None.
        """
        root_steps = plan.steps
        best_steps = root_steps
        best_score, _ = await self._score_steps(goal_input, self._brief(root_steps))
        expansions = 0
        # pUCT-like frontier nodes
        frontier_nodes: list[dict[str, Any]] = [
            {
                "steps": root_steps,
                "depth": 0,
                "n": 0,
                "q": best_score,
                "p": max(0.0, min(1.0, best_score)),
            }
        ]
        total_n = 0
        while frontier_nodes and expansions < self.budget:
            # Select by pUCT
            best_i = 0
            best_ucb = -1e9
            for i, nd in enumerate(frontier_nodes):
                n = nd["n"]
                q = nd["q"]
                p = max(0.0, min(1.0, float(nd.get("p", 0.5))))
                u = q + self.c_puct * p * math.sqrt(max(1, total_n)) / (1 + n)
                if u > best_ucb:
                    best_ucb = u
                    best_i = i
            nd = frontier_nodes.pop(best_i)
            depth = nd["depth"]
            steps_seq: list[PlanStep] = nd["steps"]
            if depth >= self.max_depth:
                continue
            # Localized repair windows (batch across 1..N segments)
            cp_idx = next(
                (i for i, s in enumerate(steps_seq) if self._is_choice_point(s)), None
            )
            if cp_idx is None:
                cp_idx = len(steps_seq) // 2
            segs = self._segment_steps_by_intent(steps_seq)
            # Choose primary segment containing cp_idx
            seg_w0, seg_w1 = 0, len(steps_seq)
            for a, b in segs:
                if a <= cp_idx < b:
                    seg_w0, seg_w1 = a, b
                    break
            windows = [
                (
                    max(0, seg_w0 - self.repair_radius),
                    min(len(steps_seq), seg_w1 + self.repair_radius),
                )
            ]
            # Optionally add one neighbor segment to diversify
            try:
                idx = segs.index((seg_w0, seg_w1)) if (seg_w0, seg_w1) in segs else -1
                if idx >= 0:
                    if idx + 1 < len(segs):
                        a, b = segs[idx + 1]
                        windows.append(
                            (
                                max(0, a - self.repair_radius),
                                min(len(steps_seq), b + self.repair_radius),
                            )
                        )
                    elif idx - 1 >= 0:
                        a, b = segs[idx - 1]
                        windows.append(
                            (
                                max(0, a - self.repair_radius),
                                min(len(steps_seq), b + self.repair_radius),
                            )
                        )
            except ValueError:
                pass
            # Expand windows and aggregate children
            children: list[list[PlanStep]] = []
            for w0, w1 in windows:
                children.extend(
                    await self._expand_window(goal_input, steps_seq, w0, w1)
                )
            # Seed with motif-derived candidates using the first (primary) window
            _primary_w0, _primary_w1 = windows[0] if windows else (0, len(steps_seq))
            motif_variants = self._motif_variants(goal_input, steps_seq, _primary_w0, _primary_w1)
            if motif_variants:
                children.extend(motif_variants)
            expansions += 1
            try:
                if self._m_exp is not None:
                    self._m_exp.inc()
                if self._h_depth is not None:
                    self._h_depth.observe(depth)
                if self._h_branch is not None:
                    self._h_branch.observe(len(children))
            except Exception:  # noqa: BLE001
                pass
            nd["n"] += 1
            total_n += 1
            # Score children (EGM-gated) and enqueue (bounded parallelism)
            child_scores: list[tuple[float, list[PlanStep]]] = []
            egm_ok_children: list[list[PlanStep]] = []
            for child in children:
                if await self._egm_ok(child):
                    egm_ok_children.append(child)
            if egm_ok_children:
                sem = asyncio.Semaphore(self.parallel)

                async def _score_one(
                    ch: list[PlanStep],
                ) -> tuple[float, list[PlanStep]]:
                    async with sem:
                        llm_sc, _ = await self._score_steps(goal_input, self._brief(ch))
                        rk, cs = self._risk_cost(ch)
                        sc = max(
                            0.0,
                            min(
                                1.0,
                                self.w_alpha * llm_sc
                                - self.w_beta * rk
                                - self.w_gamma * cs,
                            ),
                        )
                        return sc, ch

                scored = await asyncio.gather(
                    *[_score_one(ch) for ch in egm_ok_children]
                )
                child_scores.extend(scored)
                top_sc, top_child = max(scored, key=lambda x: x[0])
                if top_sc > best_score:
                    best_score = top_sc
                    best_steps = top_child
                    try:
                        log_json(
                            self.logger,
                            _logging.INFO,
                            "csc.plan.mcts_improve",
                            goal_id=str(goal_id),
                            depth=depth,
                            best_score=round(best_score, 3),
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        if self._h_best is not None:
                            self._h_best.observe(best_score)
                    except Exception:  # noqa: BLE001
                        pass
            if not child_scores:
                continue
            # Update node value toward best child
            nd["q"] = max(nd["q"], max(cs for cs, _ in child_scores))
            # Keep top-k children
            child_scores.sort(key=lambda x: x[0], reverse=True)
            keep = max(1, self.branch_factor // 2)
            for sc, child in child_scores[:keep]:
                frontier_nodes.append(
                    {
                        "steps": child,
                        "depth": depth + 1,
                        "n": 0,
                        "q": sc,
                        "p": max(0.0, min(1.0, sc)),
                    }
                )
        if best_steps is not root_steps:
            improved = Plan(
                id=uuid4(), goal_id=goal_id, steps=best_steps, status=StatusEnum.READY
            )
            # Attach step-level justifications (generic heuristics) for transparency
            reasons = []
            for i, st in enumerate(best_steps, 1):
                reasons.append(f"{i}. {st.action_type} — {self._step_reason(st)}")
            improved.justification = (
                (plan.justification or "")
                + f"\nMCTS trace: expansions={expansions}, best_score={best_score:.2f}\nReasons:\n"
                + "\n".join(reasons)
            )
            return improved
        return None

    def _step_reason(self, st: PlanStep) -> str:
        at = st.action_type.upper()
        p = _json.dumps(st.parameters, ensure_ascii=False)
        if "SELECTOR" in at:
            return "Typed scraping for structured extraction and pagination"
        if "BROWSER" in at or "AUTOMATION" in at:
            return "Automates navigation across dynamic web flows"
        if "RAG" in p.lower():
            return "Grounded retrieval with citations to reduce hallucination"
        if "SUMMARIZE" in at:
            return "Condenses content to essentials for downstream use"
        if "EDINET" in p.lower():
            return "Finance-grade regulatory source under governance"
        return "Improves efficiency/reliability vs. baseline"

    async def _expand_window(
        self, goal: str, steps_seq: list[PlanStep], w0: int, w1: int
    ) -> list[list[PlanStep]]:
        """Expand only a window of steps; returns full sequences with window replaced."""
        prefix = steps_seq[:w0]
        window = steps_seq[w0:w1]
        suffix = steps_seq[w1:]
        variants = await self._expand(goal, window)
        out: list[list[PlanStep]] = []
        for var in variants:
            out.append(prefix + var + suffix)
        if not out:
            # fallback: expand entire sequence
            all_vars = await self._expand(goal, steps_seq)
            out.extend(all_vars)
        return out

    def _motif_variants(
        self, goal: str, steps_seq: list[PlanStep], w0: int, w1: int
    ) -> list[list[PlanStep]]:
        candidates: list[list[PlanStep]] = []
        try:
            motif_steps_list = self.plan_store.retrieve(goal, top_k=self.branch_factor)
            for motif_steps in motif_steps_list:
                seq: list[PlanStep] = []
                for s in motif_steps:
                    try:
                        seq.append(
                            PlanStep(
                                description=str(s.get("description", ""))[:300],
                                action_type=str(
                                    s.get("action_type", "GENERIC_PROCESS")
                                )[:64],
                                parameters=s.get("parameters") or {},
                            )
                        )
                    except Exception:  # noqa: BLE001
                        continue
                if not seq:
                    continue
                prefix = steps_seq[:w0]
                suffix = steps_seq[w1:]
                candidates.append(prefix + seq + suffix)
        except Exception:  # noqa: BLE001
            return []
        return candidates

    def _risk_cost(self, steps: list[PlanStep]) -> tuple[float, float]:
        # Optionally configurable weights via env
        def _f(env_key: str, default: float) -> float:
            try:
                return float(os.getenv(env_key, str(default)))
            except (ValueError, TypeError):
                return default

        w_code_risk = _f("MCTS_RISK_CODE", 0.2)
        w_code_net = _f("MCTS_RISK_CODE_NET", 0.2)
        w_browser_risk = _f("MCTS_RISK_BROWSER", 0.1)
        w_scrape_cost = _f("MCTS_COST_SCRAPE", 0.1)
        w_rag_cost = _f("MCTS_COST_RAG", 0.05)
        w_browser_cost = _f("MCTS_COST_BROWSER", 0.3)
        w_code_cost = _f("MCTS_COST_CODE", 0.2)
        risk = 0.0
        cost = 0.0
        for s in steps:
            tool = str((s.parameters or {}).get("tool_id", "")).lower()
            url = (
                (s.parameters or {}).get("url")
                or (s.parameters or {}).get("parameters", {}).get("url")
                if isinstance((s.parameters or {}).get("parameters"), dict)
                else None
            )
            domain = ""
            if isinstance(url, str) and "://" in url:
                try:
                    from urllib.parse import urlparse

                    domain = urlparse(url).netloc.lower()
                except (ImportError, ValueError):
                    domain = ""
            if tool in ("python_sandbox", "node_executor", "go_executor"):
                risk += w_code_risk
                if (s.parameters or {}).get("allow_network"):
                    risk += w_code_net
                cost += w_code_cost
            elif tool in ("browser_automation",):
                risk += w_browser_risk
                cost += w_browser_cost
            elif tool in (
                "selector_extract",
                "basic_web_scraper",
                "advanced_web_scraper",
            ):
                cost += w_scrape_cost
            elif tool in ("rag_embed", "rag_retrieve", "rag_answer"):
                cost += w_rag_cost
            # Domain priors
            if domain:
                if self._high_risk_domains and domain in self._high_risk_domains:
                    risk += 0.1
                if self._low_risk_domains and domain in self._low_risk_domains:
                    risk = max(0.0, risk - 0.05)
        return min(1.0, risk), min(1.0, cost)
