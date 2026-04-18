"""
Deterministic hallucination benchmark harness for PredaCore releases.

This harness uses fixed prompt cases from hallucination_cases.json and produces
a stable scorecard so quality can be compared release-by-release.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

DEFAULT_CASES_PATH = Path(__file__).with_name("hallucination_cases.json")
UNKNOWN_MARKER_RE = re.compile(
    r"\b(not provided|unknown|not enough information|cannot determine|can't determine)\b"
)


@dataclass
class HallucinationCase:
    case_id: str
    prompt: str
    expected_facts: list[str] = field(default_factory=list)
    required_phrases: list[str] = field(default_factory=list)
    forbidden_phrases: list[str] = field(default_factory=list)
    allow_unknown: bool = False


@dataclass
class HallucinationCaseResult:
    case_id: str
    score: float
    passed: bool
    latency_ms: float
    response: str = ""
    missing_required: list[str] = field(default_factory=list)
    forbidden_hits: list[str] = field(default_factory=list)
    fact_coverage: float = 0.0
    notes: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class HallucinationScorecard:
    suite_name: str
    results: list[HallucinationCaseResult] = field(default_factory=list)
    pass_threshold: float = 0.70
    generated_at_unix: float = field(default_factory=time.time)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100.0) if self.total else 0.0

    @property
    def avg_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "generated_at_unix": self.generated_at_unix,
            "pass_threshold": self.pass_threshold,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 2),
            "avg_score": round(self.avg_score, 4),
            "results": [
                {
                    "case_id": r.case_id,
                    "score": round(r.score, 4),
                    "passed": r.passed,
                    "latency_ms": round(r.latency_ms, 2),
                    "fact_coverage": round(r.fact_coverage, 4),
                    "missing_required": r.missing_required,
                    "forbidden_hits": r.forbidden_hits,
                    "notes": r.notes,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


AgentFn = Callable[[str], Awaitable[str]]


def load_fixed_cases(path: Optional[Path] = None) -> list[HallucinationCase]:
    """Load the fixed benchmark case list from JSON."""
    cases_path = path or DEFAULT_CASES_PATH
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    return [
        HallucinationCase(
            case_id=str(item["case_id"]),
            prompt=str(item["prompt"]),
            expected_facts=[str(x) for x in item.get("expected_facts", [])],
            required_phrases=[str(x) for x in item.get("required_phrases", [])],
            forbidden_phrases=[str(x) for x in item.get("forbidden_phrases", [])],
            allow_unknown=bool(item.get("allow_unknown", False)),
        )
        for item in data
    ]


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _fact_overlap(response: str, fact: str) -> float:
    """Token-overlap score for one expected fact phrase."""
    fact_tokens = _token_set(fact)
    if not fact_tokens:
        return 1.0
    response_tokens = _token_set(response)
    if not response_tokens:
        return 0.0
    overlap = fact_tokens.intersection(response_tokens)
    return len(overlap) / len(fact_tokens)


def score_response(
    case: HallucinationCase,
    response: str,
    pass_threshold: float = 0.70,
) -> HallucinationCaseResult:
    """Score one response against a fixed hallucination benchmark case."""
    normalized = _normalize(response)
    missing_required = [
        phrase
        for phrase in case.required_phrases
        if _normalize(phrase) not in normalized
    ]
    forbidden_hits = [
        phrase
        for phrase in case.forbidden_phrases
        if _normalize(phrase) in normalized
    ]

    if case.expected_facts:
        overlaps = [_fact_overlap(normalized, fact) for fact in case.expected_facts]
        fact_coverage = sum(overlaps) / len(overlaps)
    else:
        fact_coverage = 1.0

    required_score = (
        1.0 - (len(missing_required) / len(case.required_phrases))
        if case.required_phrases
        else 1.0
    )
    forbidden_penalty = (
        len(forbidden_hits) / len(case.forbidden_phrases)
        if case.forbidden_phrases
        else 0.0
    )

    unknown_score = 1.0
    if case.allow_unknown:
        unknown_score = 1.0 if UNKNOWN_MARKER_RE.search(normalized) else 0.0

    base_score = (
        0.55 * fact_coverage
        + 0.25 * required_score
        + 0.20 * unknown_score
    )
    score = max(0.0, min(1.0, base_score - 0.50 * forbidden_penalty))
    passed = score >= pass_threshold and not forbidden_hits

    notes: list[str] = []
    if missing_required:
        notes.append("missing_required")
    if forbidden_hits:
        notes.append("forbidden_claim_detected")
    if case.allow_unknown and unknown_score == 0.0:
        notes.append("missing_uncertainty_marker")

    return HallucinationCaseResult(
        case_id=case.case_id,
        score=score,
        passed=passed,
        latency_ms=0.0,
        response=response,
        missing_required=missing_required,
        forbidden_hits=forbidden_hits,
        fact_coverage=fact_coverage,
        notes=notes,
    )


class HallucinationEvalRunner:
    """Execute a full fixed hallucination benchmark and return scorecard."""

    def __init__(
        self,
        cases: Optional[list[HallucinationCase]] = None,
        pass_threshold: float = 0.70,
        timeout_seconds: float = 45.0,
    ) -> None:
        self._cases = cases or load_fixed_cases()
        self._pass_threshold = pass_threshold
        self._timeout_seconds = timeout_seconds

    async def run_case(
        self, case: HallucinationCase, agent_fn: AgentFn
    ) -> HallucinationCaseResult:
        t0 = time.monotonic()
        try:
            response = await asyncio.wait_for(
                agent_fn(case.prompt), timeout=self._timeout_seconds
            )
            elapsed = (time.monotonic() - t0) * 1000.0
            scored = score_response(
                case=case,
                response=response,
                pass_threshold=self._pass_threshold,
            )
            scored.latency_ms = elapsed
            return scored
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000.0
            return HallucinationCaseResult(
                case_id=case.case_id,
                score=0.0,
                passed=False,
                latency_ms=elapsed,
                error=str(exc),
                notes=["case_execution_error"],
            )

    async def run_suite(
        self,
        agent_fn: AgentFn,
        suite_name: str = "predacore_hallucination_fixed",
    ) -> HallucinationScorecard:
        results: list[HallucinationCaseResult] = []
        for case in self._cases:
            result = await self.run_case(case, agent_fn)
            results.append(result)
        return HallucinationScorecard(
            suite_name=suite_name,
            results=results,
            pass_threshold=self._pass_threshold,
        )
