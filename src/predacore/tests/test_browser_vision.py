"""Tests for the browser vision plumbing (T4c).

Six guarantees:
  1. CanvasDetector parses the JS report into a dict with ``needs_vision``
  2. Detector swallows eval errors (returns ``needs_vision=False``)
  3. OptInGate.should_use_local respects the three modes
  4. OptInGate.should_prompt only fires in ``auto`` + no-model + not-declined
  5. record_decline blocks future prompts this session
  6. record_approve switches mode to ``always``
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from predacore.operators.vision import (
    CanvasDetector, OptInGate,
)


@pytest.mark.asyncio
async def test_canvas_detector_parses_json_report():
    bridge = AsyncMock()
    bridge._eval = AsyncMock(return_value=json.dumps({
        "needs_vision": True,
        "reason": "known_canvas_host",
        "canvas_count": 1,
        "host": "figma.com",
    }))
    detector = CanvasDetector(bridge)
    report = await detector.evaluate()
    assert report["needs_vision"] is True
    assert report["reason"] == "known_canvas_host"


@pytest.mark.asyncio
async def test_canvas_detector_handles_dict_response():
    """If _eval already returns dict (some bridges), we accept that too."""
    bridge = AsyncMock()
    bridge._eval = AsyncMock(return_value={"needs_vision": False, "reason": ""})
    detector = CanvasDetector(bridge)
    report = await detector.evaluate()
    assert report["needs_vision"] is False


@pytest.mark.asyncio
async def test_canvas_detector_swallows_errors():
    bridge = AsyncMock()
    bridge._eval = AsyncMock(side_effect=RuntimeError("CDP gone"))
    detector = CanvasDetector(bridge)
    report = await detector.evaluate()
    assert report["needs_vision"] is False
    assert "eval_failed" in report.get("reason", "")


def test_opt_in_gate_off_never_uses_local():
    gate = OptInGate(mode="off", model_present=True)
    assert gate.should_use_local() is False
    assert gate.should_prompt() is False


def test_opt_in_gate_always_with_model():
    gate = OptInGate(mode="always", model_present=True)
    assert gate.should_use_local() is True
    assert gate.should_prompt() is False


def test_opt_in_gate_auto_no_model_prompts():
    gate = OptInGate(mode="auto", model_present=False)
    assert gate.should_use_local() is False  # no model yet
    assert gate.should_prompt() is True


def test_opt_in_gate_auto_decline_blocks_prompts():
    gate = OptInGate(mode="auto", model_present=False)
    gate.record_decline()
    assert gate.should_prompt() is False
    assert gate.should_use_local() is False


def test_opt_in_gate_auto_approve_switches_to_always():
    gate = OptInGate(mode="auto", model_present=False)
    gate.record_approve()
    assert gate.mode == "always"
    assert gate.should_use_local() is True
    assert gate.should_prompt() is False


def test_opt_in_gate_auto_with_model_uses_local():
    """If the model is already on disk (e.g. user did 'always' before),
    auto should just use it without prompting."""
    gate = OptInGate(mode="auto", model_present=True)
    assert gate.should_use_local() is True
    assert gate.should_prompt() is False
