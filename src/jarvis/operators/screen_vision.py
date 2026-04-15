"""
JARVIS Screen Vision Engine — AX-first realtime screen understanding.

Strategy:
  1. AX tree provides instant structured UI data (~1ms)
  2. Targeted screenshots only for ambiguous regions
  3. Vision model (Claude) called only when AX tree is insufficient
  4. Screen diffing detects changes without full re-scan

Integration with Smart Input Engine:
  - type_into() delegates to SmartInputEngine when available
  - Falls back to AX set_value → find_and_click + type_text chain

This gives ~100ms loops for known UI patterns vs ~5s for full vision.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# OCR fallback for when AX tree is unavailable
try:
    from .ocr_fallback import OCREngine as _OCREngine
except ImportError:
    _OCREngine = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


# ── Data Types ───────────────────────────────────────────────────────


@dataclass
class UIElement:
    """A detected UI element from either AX tree or vision."""

    role: str  # e.g. "AXButton", "AXTextField", "button", "text"
    label: str  # Human-readable label/title
    value: str = ""  # Current value (for text fields)
    bounds: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    enabled: bool = True
    source: str = "ax"  # "ax" or "vision"
    confidence: float = 1.0  # 1.0 for AX, 0.0-1.0 for vision
    children: list["UIElement"] = field(default_factory=list)
    ax_ref: str = ""  # AX element reference for direct interaction
    depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize screen element to dictionary."""
        d: dict[str, Any] = {
            "role": self.role,
            "label": self.label,
            "source": self.source,
        }
        if self.value:
            d["value"] = self.value
        if self.bounds != (0, 0, 0, 0):
            d["bounds"] = {"x": self.bounds[0], "y": self.bounds[1],
                           "w": self.bounds[2], "h": self.bounds[3]}
        if not self.enabled:
            d["enabled"] = False
        if self.confidence < 1.0:
            d["confidence"] = round(self.confidence, 2)
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


@dataclass
class ScreenState:
    """A snapshot of the current screen state."""

    timestamp: float
    app_name: str
    window_title: str
    elements: list[UIElement]
    ax_tree_hash: str = ""  # For diffing
    screenshot_path: str = ""
    scan_ms: float = 0.0  # How long the scan took

    @property
    def element_count(self) -> int:
        """Return the number of elements in this snapshot."""
        def _count(els: list[UIElement]) -> int:
            return sum(1 + _count(e.children) for e in els)
        return _count(self.elements)

    def find(self, role: str = "", label_contains: str = "") -> list[UIElement]:
        """Find elements matching criteria."""
        results: list[UIElement] = []
        def _search(els: list[UIElement]) -> None:
            for el in els:
                match = True
                if role and el.role.lower() != role.lower():
                    if not el.role.lower().endswith(role.lower()):
                        match = False
                if label_contains and label_contains.lower() not in el.label.lower():
                    match = False
                if match:
                    results.append(el)
                _search(el.children)
        _search(self.elements)
        return results

    def summary(self) -> str:
        """One-line summary of current screen state."""
        buttons = self.find(role="button")
        text_fields = self.find(role="textfield")
        return (
            f"[{self.app_name}] {self.window_title} | "
            f"{self.element_count} elements, "
            f"{len(buttons)} buttons, {len(text_fields)} text fields | "
            f"scanned in {self.scan_ms:.0f}ms"
        )


# ── Screen Vision Engine ────────────────────────────────────────────


class ScreenVisionEngine:
    """
    AX-first realtime screen understanding engine.

    Usage:
        engine = ScreenVisionEngine(desktop_operator)
        state = await engine.scan()           # Full scan
        state = await engine.quick_scan()     # AX only (fastest)
        changed = await engine.has_changed()  # Diff detection
        element = await engine.find_and_click("Save")
    """

    def __init__(
        self,
        desktop_operator: Any,
        llm_interface: Any = None,
        max_ax_depth: int = 4,
        max_ax_children: int = 50,
        operators_config: Any = None,
        smart_input_engine: Any = None,
    ):
        self._op = desktop_operator
        self._llm = llm_interface
        self._smart_input = smart_input_engine
        # Config overrides constructor defaults
        if operators_config:
            self._max_depth = getattr(operators_config, "ax_default_depth", max_ax_depth)
            self._max_children = getattr(operators_config, "ax_max_children", max_ax_children)
        else:
            self._max_depth = max_ax_depth
            self._max_children = max_ax_children
        self._last_state: ScreenState | None = None
        self._last_ax_hash: str = ""
        self._scan_count: int = 0

        # Auto-detect smart input from desktop operator if not passed explicitly
        if self._smart_input is None and hasattr(desktop_operator, 'smart_input'):
            self._smart_input = desktop_operator.smart_input

        # OCR fallback engine — lazy init, used when AX tree fails
        self._ocr_engine: Any = None

    # ── Core Scanning ────────────────────────────────────────────

    async def quick_scan(self) -> ScreenState:
        """AX-only scan — ~1-5ms, no screenshots, no LLM."""
        t0 = time.time()

        # Get frontmost app
        try:
            app_info = await self._run_action("frontmost_app", {})
            app_name = app_info.get("app_name", "Unknown")
        except (OSError, RuntimeError) as e:
            logger.debug("Failed to get frontmost app: %s", e)
            app_name = "Unknown"

        # Get AX tree — may fail if Accessibility not granted
        snapshot: dict[str, Any] = {}
        elements: list[UIElement] = []
        try:
            ax_result = await self._run_action("ax_query", {
                "target": "focused_window",
                "max_depth": self._max_depth,
                "max_children": self._max_children,
            })
            snapshot = ax_result.get("snapshot", {})
            elements = self._ax_to_elements(snapshot)
        except (OSError, RuntimeError) as e:
            logger.debug("AX query failed (Accessibility?): %s", e)
            # Fallback: try focused_app instead of focused_window
            try:
                ax_result = await self._run_action("ax_query", {
                    "target": "focused_app",
                    "max_depth": self._max_depth,
                    "max_children": self._max_children,
                })
                snapshot = ax_result.get("snapshot", {})
                elements = self._ax_to_elements(snapshot)
            except (OSError, RuntimeError) as e:
                # AX unavailable — return minimal state
                logger.warning("AX tree unavailable — screen vision degraded: %s", e)

        window_title = str(snapshot.get("title", "") or "")

        # Hash for diffing — md5 used as content fingerprint, not crypto
        ax_hash = hashlib.md5(
            json.dumps(snapshot, default=str, sort_keys=True).encode(),
            usedforsecurity=False,
        ).hexdigest()[:12]

        # Return cached state if AX tree hash hasn't changed
        if self._last_state and ax_hash == self._last_ax_hash:
            return self._last_state

        elapsed = (time.time() - t0) * 1000
        self._scan_count += 1

        state = ScreenState(
            timestamp=time.time(),
            app_name=app_name,
            window_title=window_title,
            elements=elements,
            ax_tree_hash=ax_hash,
            scan_ms=round(elapsed, 1),
        )
        self._last_state = state
        self._last_ax_hash = ax_hash
        return state

    async def scan(self, include_screenshot: bool = False) -> ScreenState:
        """Full scan — AX tree + optional screenshot for vision analysis."""
        state = await self.quick_scan()

        if include_screenshot:
            shot = await self._run_action("screenshot", {
                "region": "full",
            })
            state.screenshot_path = shot.get("path", "")

        return state

    async def scan_with_vision(
        self,
        task: str = "Describe what you see on screen",
    ) -> tuple[ScreenState, str]:
        """Full scan with LLM vision analysis of screenshot.

        Returns (state, vision_description).
        """
        # Skip LLM vision if AX tree has actionable elements
        ax_state = await self.scan(include_screenshot=False)
        if ax_state.element_count > 5:
            return ax_state, ""  # AX tree is rich enough, skip expensive LLM call

        state = await self.scan(include_screenshot=True)

        if not self._llm or not state.screenshot_path:
            return state, ""

        # Read screenshot as base64
        try:
            with open(state.screenshot_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
        except OSError:
            return state, "[screenshot read failed]"

        # Build AX context for the LLM
        ax_summary = state.summary()
        actionable = []
        for el in state.find(role="button"):
            actionable.append(f"- Button: {el.label}")
        for el in state.find(role="textfield"):
            actionable.append(f"- TextField: {el.label} = '{el.value}'")
        for el in state.find(role="menuitem"):
            actionable.append(f"- MenuItem: {el.label}")

        context = ax_summary
        if actionable:
            context += "\nActionable elements:\n" + "\n".join(actionable[:30])

        # Call LLM with vision — use provider-agnostic image format.
        # Anthropic uses {"type":"image","source":{"type":"base64",...}},
        # OpenAI uses {"type":"image_url","image_url":{"url":"data:..."}}.
        # The LLM router's chat() should normalize this, but we provide
        # both hints so providers can pick the right format.
        image_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img_b64,
            },
            # OpenAI-compatible alternative (router can use whichever)
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        }
        messages = [
            {"role": "system", "content": (
                "You are a screen analysis assistant. Analyze the screenshot "
                "and the AX accessibility tree data. Identify what the user is "
                "looking at and suggest available actions.\n\n"
                f"AX Context:\n{context}"
            )},
            {"role": "user", "content": [
                {"type": "text", "text": task},
                image_block,
            ]},
        ]

        try:
            result = await self._llm.chat(messages)
            description = result.get("content", "")
        except (OSError, RuntimeError, ValueError, TimeoutError) as e:
            logger.warning("Vision LLM call failed: %s", e)
            description = f"[vision failed: {e}]"

        return state, description

    # ── Change Detection ─────────────────────────────────────────

    async def has_changed(self) -> bool:
        """Check if the screen has changed since last scan (fast)."""
        old_hash = self._last_ax_hash
        state = await self.quick_scan()
        return state.ax_tree_hash != old_hash

    async def wait_for_change(
        self,
        timeout: float = 30.0,
        poll_interval: float = 0.2,
    ) -> ScreenState | None:
        """Block until screen changes or timeout. Returns new state or None."""
        deadline = time.time() + timeout
        baseline_hash = self._last_ax_hash

        while time.time() < deadline:
            state = await self.quick_scan()
            if state.ax_tree_hash != baseline_hash:
                logger.info(
                    "Screen changed after %.1fs",
                    time.time() - (deadline - timeout),
                )
                return state
            await asyncio.sleep(poll_interval)

        return None

    # ── Smart Actions ────────────────────────────────────────────

    async def find_and_click(
        self,
        label: str,
        role: str = "",
    ) -> dict[str, Any]:
        """Find a UI element by label and click it using AX (precise)."""
        match: dict[str, Any] = {"title_contains": label}
        if role:
            match["role"] = role

        try:
            result = await self._run_action("ax_click", {
                "target": "focused_window",
                "match": match,
                "max_depth": 6,
                "max_children": 50,
            })
            return {
                "ok": True,
                "method": "ax_click",
                "matched": result.get("matched", {}),
            }
        except (OSError, RuntimeError) as ax_err:
            logger.warning("AX click failed for '%s': %s", label, ax_err)

            # Fallback: scan for element position and mouse click
            state = await self.quick_scan()
            elements = state.find(label_contains=label)
            if not elements:
                return {
                    "ok": False,
                    "error": f"Element '{label}' not found",
                    "method": "none",
                }

            el = elements[0]
            if el.bounds != (0, 0, 0, 0):
                x = el.bounds[0] + el.bounds[2] // 2
                y = el.bounds[1] + el.bounds[3] // 2
                await self._run_action("mouse_click", {"x": x, "y": y})
                return {
                    "ok": True,
                    "method": "mouse_fallback",
                    "element": el.to_dict(),
                }

            return {
                "ok": False,
                "error": f"Element '{label}' found but no bounds available",
                "method": "none",
            }

    async def type_into(
        self,
        label: str,
        text: str,
    ) -> dict[str, Any]:
        """
        Find a text field by label and set its value.

        Priority chain (powered by SmartInputEngine when available):
          1. AX set_value (direct injection)
          2. SmartInputEngine verified keystroke (if smart input available)
          3. Click field + raw type_text (legacy fallback)
        """
        # ── Tier 2: Try AX set_value first (most reliable for text fields) ──
        match: dict[str, Any] = {"title_contains": label}
        try:
            result = await self._run_action("ax_set_value", {
                "target": "focused_window",
                "match": match,
                "value": text,
                "max_depth": 6,
                "max_children": 50,
            })
            return {
                "ok": True,
                "method": "ax_set_value",
                "matched": result.get("matched", {}),
            }
        except (OSError, RuntimeError) as ax_err:
            logger.warning("AX set_value failed for '%s': %s", label, ax_err)

        # ── SmartInputEngine: Click field, then use verified keystroke ──
        if self._smart_input is not None:
            try:
                # Click the field first to focus it
                click_result = await self.find_and_click(label, role="AXTextField")
                if click_result.get("ok"):
                    await asyncio.sleep(0.05)
                    # Use smart_type with verification and retry
                    smart_result = await self._smart_input.smart_type(
                        text=text,
                        verify=True,
                    )
                    return {
                        "ok": smart_result.ok,
                        "method": f"smart_input:{smart_result.method.value}",
                        "verified": smart_result.verified,
                        "attempts": smart_result.attempts,
                    }
            except (OSError, RuntimeError, TypeError) as smart_err:
                logger.warning(
                    "SmartInput type_into failed for '%s': %s — trying legacy",
                    label, smart_err,
                )

        # ── Legacy fallback: Click the field then raw type ──
        click_result = await self.find_and_click(label, role="AXTextField")
        if click_result.get("ok"):
            await asyncio.sleep(0.05)
            await self._run_action("type_text", {"text": text})
            return {
                "ok": True,
                "method": "click_and_type_fallback",
            }

        return {
            "ok": False,
            "error": f"Could not type into '{label}': field not found or not accessible",
        }

    async def read_screen_text(self, use_ocr_fallback: bool = True) -> list[dict[str, str]]:
        """Extract all visible text from the screen.

        Strategy:
          1. Try AX tree first (instant, structured)
          2. If AX returns empty and use_ocr_fallback=True, take a screenshot
             and run OCR (Vision framework or tesseract) to extract text

        Returns:
            List of {"role": ..., "text": ...} dicts.
        """
        state = await self.quick_scan()
        texts: list[dict[str, str]] = []

        def _extract(els: list[UIElement]) -> None:
            for el in els:
                if el.label:
                    texts.append({"role": el.role, "text": el.label})
                if el.value and el.value != el.label:
                    texts.append({"role": el.role, "text": el.value, "type": "value"})
                _extract(el.children)

        _extract(state.elements)

        # OCR fallback — when AX tree returns nothing useful
        if not texts and use_ocr_fallback:
            ocr_texts = await self._ocr_read_screen()
            if ocr_texts:
                texts.extend(ocr_texts)
                logger.info("OCR fallback produced %d text regions (AX was empty)", len(ocr_texts))

        return texts

    async def scan_with_ocr(self) -> dict[str, Any]:
        """Full scan with OCR — useful when AX tree is unavailable.

        Takes a screenshot and runs OCR to extract all visible text with
        positions and confidence scores. Much slower than AX (~200-500ms)
        but works regardless of Accessibility permissions.

        Returns:
            {"texts": [...], "full_text": "...", "ocr_engine": "vision|tesseract", "scan_ms": ...}
        """
        t0 = time.time()

        # Take screenshot
        try:
            shot = await self._run_action("screenshot", {"region": "full"})
            screenshot_path = shot.get("path", "")
        except (OSError, RuntimeError) as exc:
            return {"error": f"Screenshot failed: {exc}", "texts": [], "full_text": ""}

        if not screenshot_path:
            return {"error": "No screenshot path returned", "texts": [], "full_text": ""}

        # Run OCR
        ocr = self._get_ocr_engine()
        if ocr is None:
            return {
                "error": "No OCR engine available (install PyObjC or tesseract)",
                "texts": [], "full_text": "",
            }

        try:
            results = await ocr.extract_text(screenshot_path, min_confidence=0.3)
            full_text = await ocr.extract_full_text(screenshot_path, min_confidence=0.3)
        except (OSError, RuntimeError, ValueError) as exc:
            return {"error": f"OCR failed: {exc}", "texts": [], "full_text": ""}

        elapsed_ms = (time.time() - t0) * 1000
        return {
            "texts": [r.to_dict() for r in results],
            "full_text": full_text,
            "text_count": len(results),
            "ocr_engine": results[0].source if results else "none",
            "scan_ms": round(elapsed_ms, 1),
            "screenshot_path": screenshot_path,
        }

    async def _ocr_read_screen(self) -> list[dict[str, str]]:
        """Internal: Take screenshot + OCR to extract text regions."""
        ocr = self._get_ocr_engine()
        if ocr is None:
            return []

        try:
            shot = await self._run_action("screenshot", {"region": "full"})
            screenshot_path = shot.get("path", "")
            if not screenshot_path:
                return []

            results = await ocr.extract_text(screenshot_path, min_confidence=0.4)
            return [
                {"role": "ocr_text", "text": r.text, "source": "ocr", "confidence": str(round(r.confidence, 2))}
                for r in results
            ]
        except (OSError, RuntimeError, ValueError) as exc:
            logger.debug("OCR fallback failed: %s", exc)
            return []

    def _get_ocr_engine(self) -> Any:
        """Lazy-init OCR engine."""
        if self._ocr_engine is not None:
            return self._ocr_engine
        if _OCREngine is not None:
            try:
                self._ocr_engine = _OCREngine()
                if self._ocr_engine.available:
                    logger.debug("OCR engine initialized: %s", self._ocr_engine.status())
                    return self._ocr_engine
                else:
                    logger.debug("OCR engine not available (no Vision/tesseract)")
                    self._ocr_engine = None
            except (OSError, RuntimeError, ImportError) as exc:
                logger.debug("OCR engine init failed: %s", exc)
                self._ocr_engine = None
        return None

    async def focused_element(self) -> UIElement | None:
        """Return the currently focused UI element."""
        try:
            result = await self._run_action("ax_query", {
                "target": "focused_element",
                "max_depth": 0,
            })
            snapshot = result.get("snapshot", {})
            if snapshot:
                elements = self._ax_to_elements(snapshot)
                return elements[0] if elements else None
        except (OSError, RuntimeError) as e:
            logger.debug("Failed to query focused element: %s", e)
        return None

    def diff_states(
        self, old: ScreenState, new: ScreenState
    ) -> list[dict[str, Any]]:
        """Compare two screen states and return element-level changes."""
        changes: list[dict[str, Any]] = []
        if old.app_name != new.app_name:
            changes.append({"type": "app_changed", "from": old.app_name, "to": new.app_name})
        if old.window_title != new.window_title:
            changes.append({"type": "window_changed", "from": old.window_title, "to": new.window_title})

        old_els = {(e.role, e.label): e for e in self._flatten(old.elements)}
        new_els = {(e.role, e.label): e for e in self._flatten(new.elements)}

        for key in new_els:
            if key not in old_els:
                el = new_els[key]
                changes.append({"type": "added", "role": el.role, "label": el.label})
            elif new_els[key].value != old_els[key].value:
                changes.append({
                    "type": "value_changed", "role": key[0], "label": key[1],
                    "from": old_els[key].value, "to": new_els[key].value,
                })
        for key in old_els:
            if key not in new_els:
                el = old_els[key]
                changes.append({"type": "removed", "role": el.role, "label": el.label})
        return changes

    @staticmethod
    def _flatten(elements: list[UIElement]) -> list[UIElement]:
        """Flatten nested element tree into a flat list."""
        result: list[UIElement] = []
        for el in elements:
            result.append(el)
            result.extend(ScreenVisionEngine._flatten(el.children))
        return result

    async def execute_task(
        self,
        instruction: str,
        max_steps: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Autonomous task execution loop.

        Takes a natural language instruction (e.g. "Open Safari and search for Python docs")
        and executes it step-by-step using AX tree + vision + SmartInput.

        Returns a log of actions taken.
        """
        if not self._llm:
            return [{"error": "No LLM configured for autonomous execution"}]

        action_log: list[dict[str, Any]] = []

        # Conversation history so LLM can learn from previous steps
        history: list[dict[str, str]] = []

        _SYSTEM_PROMPT = (
            "You are a desktop automation agent. You control a macOS computer.\n"
            "Given the current screen state (from Accessibility API) and the task, "
            "output the NEXT SINGLE action as JSON.\n\n"
            "Available actions:\n"
            '  {"action": "click", "label": "Save"}\n'
            '  {"action": "click", "label": "Save", "role": "AXButton"}\n'
            '  {"action": "type", "label": "Search", "text": "hello"}\n'
            '  {"action": "press_key", "key": "enter"}\n'
            '  {"action": "press_key", "key": "a", "modifiers": ["command"]}\n'
            '  {"action": "open_app", "app_name": "Safari"}\n'
            '  {"action": "open_url", "url": "https://..."}\n'
            '  {"action": "run_command", "command": "ls -la", "app": "Terminal"}\n'
            '  {"action": "scroll", "direction": "down", "amount": 3}\n'
            '  {"action": "wait", "seconds": 2}\n'
            '  {"action": "done", "result": "Task completed"}\n\n'
            "Output ONLY the JSON for one action. No explanation.\n"
            "When the task is complete, use the 'done' action."
        )

        for step in range(max_steps):
            # 1. Scan current state
            state = await self.quick_scan()
            ax_context = self._state_to_context(state)

            # 2. Build messages with full conversation history
            user_msg = (
                f"Task: {instruction}\n\n"
                f"Current screen state (step {step + 1}):\n{ax_context}\n\n"
                "What is the next action?"
            )
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                *history,
                {"role": "user", "content": user_msg},
            ]

            try:
                llm_result = await self._llm.chat(messages)
                response_text = llm_result.get("content", "")
            except (OSError, RuntimeError, ValueError, TimeoutError) as e:
                action_log.append({
                    "step": step + 1,
                    "error": f"LLM call failed: {e}",
                })
                break

            # 3. Parse action
            parsed = self._parse_action(response_text)
            if parsed is None:
                action_log.append({
                    "step": step + 1,
                    "error": f"Could not parse action from LLM response: {response_text[:200]}",
                })
                break

            # Track conversation history so LLM sees previous steps
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": response_text})

            action_log.append({
                "step": step + 1,
                "action": parsed,
                "screen": state.summary(),
            })

            # 4. Check if done
            if parsed.get("action") == "done":
                action_log[-1]["result"] = parsed.get("result", "Task complete")
                break

            # 5. Execute the action
            exec_error = None
            try:
                await self._execute_agent_action(parsed)
            except (OSError, RuntimeError, ValueError) as e:
                exec_error = str(e)
                action_log[-1]["execution_error"] = exec_error

            # Add execution result to history so LLM knows what happened
            if exec_error:
                history.append({"role": "user", "content": f"Action failed: {exec_error}"})

            # Trim history to last 10 exchanges to avoid context overflow
            if len(history) > 20:
                history = history[-20:]

            # 6. Brief pause for UI to settle
            await asyncio.sleep(0.1)

        return action_log

    # ── Internal Helpers ─────────────────────────────────────────

    async def _run_action(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Run a desktop operator action (bridge sync operator to async)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._op.execute(action, params),
        )

    def _ax_to_elements(self, snapshot: dict[str, Any], depth: int = 0) -> list[UIElement]:
        """Convert AX snapshot dict to UIElement tree."""
        if not snapshot:
            return []

        role = str(snapshot.get("role", ""))
        title = str(snapshot.get("title", "") or "")
        desc = str(snapshot.get("description", "") or "")
        label = title or desc
        value = str(snapshot.get("value", "") or "")
        enabled = snapshot.get("enabled", True)

        # Extract position and size from AX data
        pos = snapshot.get("position", {})
        size = snapshot.get("size", {})
        bounds = (0, 0, 0, 0)
        if isinstance(pos, dict) and isinstance(size, dict):
            try:
                bounds = (
                    int(float(pos.get("x", 0))),
                    int(float(pos.get("y", 0))),
                    int(float(size.get("width", 0))),
                    int(float(size.get("height", 0))),
                )
            except (ValueError, TypeError):
                pass

        el = UIElement(
            role=role,
            label=label,
            value=value,
            bounds=bounds,
            enabled=bool(enabled),
            source="ax",
            confidence=1.0,
            ax_ref=str(snapshot.get("element_ref", "")),
            depth=depth,
        )

        # Recursively process children
        for child_snap in snapshot.get("children", []):
            child_elements = self._ax_to_elements(child_snap, depth=depth + 1)
            el.children.extend(child_elements)

        return [el]

    def _state_to_context(self, state: ScreenState) -> str:
        """Convert screen state to text context for LLM."""
        lines = [state.summary(), ""]

        def _render(els: list[UIElement], indent: int = 0) -> None:
            for el in els:
                prefix = "  " * indent
                parts = [f"{prefix}[{el.role}]"]
                if el.label:
                    parts.append(f'"{el.label}"')
                if el.value:
                    parts.append(f"value={el.value!r}")
                if el.bounds != (0, 0, 0, 0):
                    parts.append(f"@({el.bounds[0]},{el.bounds[1]} {el.bounds[2]}x{el.bounds[3]})")
                if not el.enabled:
                    parts.append("(disabled)")
                lines.append(" ".join(parts))
                _render(el.children, indent + 1)

        _render(state.elements)
        return "\n".join(lines)

    @staticmethod
    def _parse_action(text: str) -> dict[str, Any] | None:
        """Extract JSON action from LLM response."""
        text = text.strip()
        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block — find matching ``` pair
        import re as _re
        code_block = _re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, _re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                pass

        # Try finding first balanced { ... } (handles nested braces)
        brace_start = text.find("{")
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[brace_start:i + 1])
                        except json.JSONDecodeError:
                            break

        return None

    # Actions the autonomous agent is allowed to execute
    # NOTE: run_command removed — autonomous shell execution without user confirmation
    # is a security risk. Use desktop_control actions instead.
    _ALLOWED_AGENT_ACTIONS = frozenset({
        "click", "type", "press_key", "open_app", "open_url",
        "scroll", "wait", "screenshot", "done",
    })

    async def _execute_agent_action(self, action: dict[str, Any]) -> None:
        """Execute a parsed agent action (only allowlisted actions)."""
        act = action.get("action", "")
        if act not in self._ALLOWED_AGENT_ACTIONS:
            logger.warning("Blocked disallowed agent action: %s", act)
            return

        if act == "click":
            label = action.get("label", "")
            role = action.get("role", "")
            await self.find_and_click(label, role=role)

        elif act == "type":
            label = action.get("label", "")
            text = action.get("text", "")
            if label:
                await self.type_into(label, text)
            elif self._smart_input:
                # No label — just type the text using smart input
                await self._smart_input.smart_type(text=text, verify=False)
            else:
                await self._run_action("type_text", {"text": text})

        elif act == "press_key":
            key = action.get("key", "")
            modifiers = action.get("modifiers", [])
            await self._run_action("press_key", {
                "key": key,
                "modifiers": modifiers,
            })

        elif act == "open_app":
            app_name = action.get("app_name", "")
            await self._run_action("open_app", {"app_name": app_name})

        elif act == "open_url":
            url = action.get("url", "")
            if self._smart_input:
                browser = action.get("browser", "")
                await self._smart_input.smart_open_url(url=url, browser=browser)
            else:
                await self._run_action("open_url", {"url": url})

        # run_command removed — autonomous shell execution without user confirmation
        # is a security risk. The action is blocked by _ALLOWED_AGENT_ACTIONS.

        elif act == "scroll":
            direction = action.get("direction", "down")
            amount = action.get("amount", 3)
            clicks = amount if direction == "down" else -amount
            await self._run_action("mouse_scroll", {
                "amount": clicks,
                "direction": direction,
            })

        elif act == "wait":
            seconds = min(float(action.get("seconds", 1)), 10.0)
            await asyncio.sleep(seconds)

        elif act == "screenshot":
            await self._run_action("screenshot", {"region": "full"})

        else:
            logger.warning("Unknown agent action: %s", act)
