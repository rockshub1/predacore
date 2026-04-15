"""
JARVIS Operators — Device automation layer.

This package provides platform-specific operator implementations that
the tool layer dispatches to. Each operator inherits from BaseOperator
and provides a uniform execute(action, params) → dict interface.

Architecture:
  base.py          — BaseOperator ABC, OperatorError, MacroAbortToken, enums
  enums.py         — DesktopAction, AndroidAction, VisionAction, ScreenshotQuality
  desktop.py       — MacDesktopOperator (macOS Accessibility + AppleScript + pyautogui)
  android.py       — AndroidOperator (ADB-based device automation)
  screen_vision.py — ScreenVisionEngine (AX-first screen understanding + OCR fallback)
  smart_input.py   — SmartInputEngine (3-tier bulletproof text input)
  native_service.py — MacDesktopNativeService (PyObjC Accessibility API)
  ocr_fallback.py  — OCREngine (Vision framework + tesseract CLI)
  mock.py          — MockDesktopOperator, MockAndroidOperator (for testing)
"""
from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

# ── Base classes and enums (always available) ──
from .base import (
    BaseOperator,
    OperatorPlatform,
    ActionCategory,
    OperatorError,
    MacroAbortToken,
)
from .enums import (
    DesktopAction,
    AndroidAction,
    VisionAction,
    ScreenshotQuality,
    SMART_ACTIONS,
    NATIVE_ONLY_ACTIONS,
    NATIVE_CAPABLE_ACTIONS,
)

# ── Concrete operators (may fail on non-macOS / missing deps) ──
# Import errors are logged so missing deps surface in debug, not silent.

try:
    from .desktop import MacDesktopOperator, AsyncBridge, DesktopControlError
except ImportError as _exc:
    _logger.debug("MacDesktopOperator unavailable: %s", _exc)
    MacDesktopOperator = None  # type: ignore[misc,assignment]
    AsyncBridge = None  # type: ignore[misc,assignment]
    DesktopControlError = None  # type: ignore[misc,assignment]

try:
    from .android import AndroidOperator, ADBError, AndroidElement
except ImportError as _exc:
    _logger.debug("AndroidOperator unavailable: %s", _exc)
    AndroidOperator = None  # type: ignore[misc,assignment]
    ADBError = None  # type: ignore[misc,assignment]
    AndroidElement = None  # type: ignore[misc,assignment]

try:
    from .screen_vision import ScreenVisionEngine
except ImportError as _exc:
    _logger.debug("ScreenVisionEngine unavailable: %s", _exc)
    ScreenVisionEngine = None  # type: ignore[misc,assignment]

try:
    from .ocr_fallback import OCREngine, OCRResult
except ImportError as _exc:
    _logger.debug("OCREngine unavailable: %s", _exc)
    OCREngine = None  # type: ignore[misc,assignment]
    OCRResult = None  # type: ignore[misc,assignment]

try:
    from .mock import MockDesktopOperator, MockAndroidOperator
except ImportError as _exc:
    _logger.debug("Mock operators unavailable: %s", _exc)
    MockDesktopOperator = None  # type: ignore[misc,assignment]
    MockAndroidOperator = None  # type: ignore[misc,assignment]


__all__ = [
    # Base
    "BaseOperator", "OperatorPlatform", "ActionCategory",
    "OperatorError", "MacroAbortToken",
    # Enums
    "DesktopAction", "AndroidAction", "VisionAction", "ScreenshotQuality",
    "SMART_ACTIONS", "NATIVE_ONLY_ACTIONS", "NATIVE_CAPABLE_ACTIONS",
    # Desktop
    "MacDesktopOperator", "AsyncBridge", "DesktopControlError",
    # Android
    "AndroidOperator", "ADBError", "AndroidElement",
    # Vision
    "ScreenVisionEngine", "OCREngine", "OCRResult",
    # Mocks
    "MockDesktopOperator", "MockAndroidOperator",
]
