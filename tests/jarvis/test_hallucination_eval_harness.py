from __future__ import annotations

import pytest

from tests.jarvis.hallucination_eval_harness import (
    HallucinationCase,
    HallucinationEvalRunner,
    load_fixed_cases,
    score_response,
)


def test_load_fixed_cases_has_stable_entries():
    cases = load_fixed_cases()
    assert len(cases) >= 6
    assert cases[0].case_id == "release_summary"


def test_score_response_flags_forbidden_claims():
    case = HallucinationCase(
        case_id="x",
        prompt="q",
        expected_facts=["Helios"],
        required_phrases=["Helios"],
        forbidden_phrases=["Orion"],
    )

    bad = score_response(
        case=case,
        response="The codename is Orion.",
        pass_threshold=0.70,
    )
    assert bad.passed is False
    assert "Orion" in bad.forbidden_hits

    good = score_response(
        case=case,
        response="The codename is Helios.",
        pass_threshold=0.70,
    )
    assert good.passed is True
    assert good.score >= 0.70


def test_allow_unknown_requires_uncertainty_marker():
    case = HallucinationCase(
        case_id="unknown",
        prompt="q",
        expected_facts=[],
        required_phrases=["not provided"],
        forbidden_phrases=["us-east-1"],
        allow_unknown=True,
    )

    no_marker = score_response(
        case=case,
        response="The region is us-east-1.",
        pass_threshold=0.70,
    )
    assert no_marker.passed is False
    assert "missing_uncertainty_marker" in no_marker.notes

    with_marker = score_response(
        case=case,
        response="The region is not provided in the given context.",
        pass_threshold=0.70,
    )
    assert with_marker.passed is True


@pytest.mark.asyncio
async def test_runner_builds_scorecard():
    cases = load_fixed_cases()

    async def deterministic_agent(prompt: str) -> str:
        if "codename and freeze date" in prompt:
            return "Codename: Helios. Freeze date: 2026-03-14."
        if "highest uptime" in prompt:
            return "Server A has the highest uptime at 99.95%."
        if "deployment region" in prompt:
            return "The region is not provided in the context."
        if "currently active" in prompt:
            return "Policy v3 is currently active."
        if "total number of completed tasks" in prompt:
            return "Total completed tasks: 36."
        if "default sandbox constraints" in prompt:
            return (
                "Two constraints are no root shell and outbound network disabled "
                "unless explicitly enabled."
            )
        return "not provided"

    runner = HallucinationEvalRunner(cases=cases, pass_threshold=0.70)
    scorecard = await runner.run_suite(deterministic_agent)

    assert scorecard.total == len(cases)
    assert scorecard.passed == len(cases)
    assert scorecard.avg_score >= 0.90
