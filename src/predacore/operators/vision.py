"""Browser vision fallback (T4c — opt-in).

For the 10-15% of pages where CDP can't see what the user sees:

  * Canvas-rendered apps (Figma, Excalidraw, Photopea, web games)
  * Google Docs (canvas migration since 2021)
  * Closed shadow DOMs (Stripe payment iframes etc.)
  * Cross-origin iframes the OOPIF target switch can't reach

We fall back to vision-based grounding: take a screenshot, run a small
local model (OmniParser via the predacore_core Rust crate), get a
structured element tree with bbox + role + text, and let the LLM pick
an element by index. Same shape as the DOM extractor's output, just
with bounding boxes from pixels instead of DOM measurements.

Architecture
------------
- ``CanvasDetector`` (here): tells the bridge whether the current page
  needs the vision fallback. Pure JS heuristic via CDP, ~5ms.
- ``VisionProvider`` (interface here): the contract the bridge calls
  to ground a screenshot. Two concrete impls planned:

    * ``RustOmniParser``  — predacore_core.parse_screen(png_bytes) via
      ``ort`` v2 (ONNX Runtime Rust bindings) + CoreML execution
      provider on Apple Silicon. ~30-50ms per screenshot. **Pending —
      see ``predacore_core_crate/src/omniparser.rs`` skeleton.**
    * ``CloudVisionFallback`` — calls Anthropic / Gemini / OpenAI vision
      with the screenshot + page tree request. Slow (~2s) but works
      out of the box without a model download. Ships today.

- ``OptInGate``: tracks whether the user has approved the OmniParser
  model download (~500MB). Default ``"auto"`` = prompt on first canvas
  detection; ``"always"`` = pre-download on bootstrap; ``"off"`` = never
  use the local model, always fall back to cloud vision.

The user toggles this via ``predacore config set browser.local_vision auto``.
Per-call override: the bridge passes a ``vision_mode`` arg through.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal, Protocol

logger = logging.getLogger(__name__)


VisionMode = Literal["auto", "always", "off"]


# JS that returns a small report on whether this page needs vision fallback.
# Cheap heuristic, runs in <5ms via CDP Runtime.evaluate.
_CANVAS_DETECT_JS = r"""
(function _predacore_canvas_detect() {
  var canvases = document.querySelectorAll('canvas');
  if (!canvases.length) return JSON.stringify({needs_vision: false, reason: ""});

  // Sum total canvas pixel area; if it covers >40% of viewport, the
  // page is canvas-dominant and the DOM is unlikely to help.
  var vp = window.innerWidth * window.innerHeight;
  var canvasArea = 0;
  for (var i = 0; i < canvases.length; i++) {
    var r = canvases[i].getBoundingClientRect();
    if (r.width > 0 && r.height > 0) canvasArea += r.width * r.height;
  }
  var ratio = canvasArea / Math.max(vp, 1);

  // Specific URL patterns we know are canvas-heavy.
  var canvasHosts = ['figma.com', 'excalidraw.com', 'photopea.com',
                     'tldraw.com', 'docs.google.com', 'whimsical.com',
                     'miro.com'];
  var host = location.hostname.toLowerCase();
  var hostMatch = canvasHosts.some(function(h){ return host === h || host.endsWith('.' + h); });

  return JSON.stringify({
    needs_vision: hostMatch || ratio > 0.4,
    reason: hostMatch ? "known_canvas_host" : (ratio > 0.4 ? "canvas_dominant" : ""),
    canvas_count: canvases.length,
    canvas_ratio: Math.round(ratio * 100) / 100,
    host: host,
  });
})()
"""


@dataclass
class VisionElement:
    """One element grounded from a screenshot — same shape as a DOM tree row."""
    index: int
    bbox: tuple[int, int, int, int]   # x, y, w, h in viewport coords
    role: str                         # "button", "link", "input", "icon", etc.
    text: str                         # OCR'd text or aria-label
    confidence: float                 # 0..1


class VisionProvider(Protocol):
    """Contract every vision backend (local Rust or cloud) implements.

    Implementations live elsewhere; the bridge depends on this interface
    so we can swap RustOmniParser for CloudVisionFallback for tests etc.
    """

    async def parse_screen(
        self, png_bytes: bytes, hints: dict[str, Any] | None = None,
    ) -> list[VisionElement]:
        ...

    @property
    def name(self) -> str:
        ...


# ── Canvas detector ──────────────────────────────────────────────────


class CanvasDetector:
    """Cheap heuristic — should this page route through vision?"""

    def __init__(self, bridge: Any) -> None:
        # Bridge has ._eval(js) — duck-typed so tests don't need a real bridge.
        self._bridge = bridge

    async def evaluate(self) -> dict[str, Any]:
        """Run the detection JS. Returns a dict with ``needs_vision`` key."""
        try:
            raw = await self._bridge._eval(_CANVAS_DETECT_JS, timeout=2.0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Canvas detect failed: %s", exc)
            return {"needs_vision": False, "reason": "eval_failed"}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {"needs_vision": False, "reason": "bad_json"}
        return {"needs_vision": False, "reason": f"unexpected_{type(raw).__name__}"}


# ── Opt-in gate ──────────────────────────────────────────────────────


class OptInGate:
    """Tracks whether the user has approved the OmniParser model download.

    Three modes (config: ``browser.local_vision``):
      * ``"auto"``   — first canvas detection prompts the user once.
                       If they approve, switch to ``"always"`` permanently.
                       If they decline, fall back to cloud vision for this
                       session and re-prompt next session.
      * ``"always"`` — model is downloaded at bootstrap; used for every
                       canvas detection.
      * ``"off"``    — never download / use the local model; cloud vision
                       always.

    The actual download lives in :mod:`predacore.bootstrap` (planned for
    when the Rust side lands). This class just gates the choice.
    """

    def __init__(self, mode: VisionMode = "auto", *, model_present: bool = False):
        self._mode: VisionMode = mode
        self._model_present = model_present
        self._user_declined_this_session = False

    @property
    def mode(self) -> VisionMode:
        return self._mode

    def should_use_local(self) -> bool:
        """Decide whether to route to OmniParser vs cloud vision."""
        if self._mode == "off":
            return False
        if self._mode == "always":
            return True
        # auto mode
        if self._user_declined_this_session:
            return False
        return self._model_present

    def should_prompt(self) -> bool:
        """Should we ask the user about downloading OmniParser right now?"""
        return (
            self._mode == "auto"
            and not self._model_present
            and not self._user_declined_this_session
        )

    def record_decline(self) -> None:
        """User said no to the download prompt — don't ask again this session."""
        self._user_declined_this_session = True

    def record_approve(self) -> None:
        """User approved + the download succeeded. Switch to ``always``."""
        self._mode = "always"
        self._model_present = True
