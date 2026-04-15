"""
Tests for JARVIS Tool & Operator Enums.

Validates enum completeness, string compatibility, cross-module re-exports,
and that HANDLER_MAP keys, WRITE_TOOLS, and READ_ONLY_TOOLS all use valid
ToolName values.
"""
from __future__ import annotations

import pytest

# ── Tool-level enums ──────────────────────────────────────────────────
from jarvis.tools.enums import (
    ToolName,
    ToolStatus,
    WRITE_TOOLS,
    READ_ONLY_TOOLS,
    # Re-exported from operators
    DesktopAction,
    AndroidAction,
    VisionAction,
    ScreenshotQuality,
    SMART_ACTIONS,
    NATIVE_ONLY_ACTIONS,
    NATIVE_CAPABLE_ACTIONS,
)

# ── Operator-level enums (canonical source) ───────────────────────────
from jarvis.operators.enums import (
    DesktopAction as OpDesktopAction,
    AndroidAction as OpAndroidAction,
    VisionAction as OpVisionAction,
    ScreenshotQuality as OpScreenshotQuality,
)

# ── HANDLER_MAP for cross-validation ─────────────────────────────────
from jarvis.tools.handlers import HANDLER_MAP


# =====================================================================
# ToolName
# =====================================================================


class TestToolName:
    """ToolName enum tests."""

    def test_is_str_enum(self):
        """ToolName members are both str and Enum."""
        assert isinstance(ToolName.READ_FILE, str)
        assert ToolName.READ_FILE == "read_file"

    def test_value_is_string(self):
        """ToolName.value gives the plain string."""
        assert ToolName.READ_FILE.value == "read_file"
        assert ToolName.DESKTOP_CONTROL.value == "desktop_control"

    def test_equality_with_plain_string(self):
        """ToolName members compare equal to plain strings via ==."""
        assert ToolName.WEB_SEARCH == "web_search"
        assert ToolName.DESKTOP_CONTROL == "desktop_control"
        assert ToolName.TOOL_PIPELINE == "tool_pipeline"

    def test_all_handler_map_keys_are_valid_tool_names(self):
        """Every key in HANDLER_MAP should be a valid ToolName enum member."""
        valid_values = {t.value for t in ToolName}
        for key in HANDLER_MAP:
            # Keys are ToolName enum members — get value for comparison
            key_val = key.value if hasattr(key, 'value') else key
            assert key_val in valid_values, f"HANDLER_MAP key '{key}' is not a valid ToolName"

    def test_all_tool_names_have_handlers(self):
        """Every ToolName should have a corresponding handler in HANDLER_MAP."""
        # HANDLER_MAP keys are ToolName enums, so direct lookup works
        for tn in ToolName:
            assert tn in HANDLER_MAP, f"ToolName.{tn.name} ({tn.value}) has no handler"

    def test_minimum_tool_count(self):
        """Sanity check — we have at least 35 tools registered."""
        assert len(ToolName) >= 35

    def test_no_duplicate_values(self):
        """No two ToolName members should share the same value."""
        values = [t.value for t in ToolName]
        assert len(values) == len(set(values))

    def test_pipeline_and_stats_registered(self):
        """The new Phase 6 tools are in the enum."""
        assert hasattr(ToolName, "TOOL_PIPELINE")
        assert hasattr(ToolName, "TOOL_STATS")
        assert ToolName.TOOL_PIPELINE.value == "tool_pipeline"
        assert ToolName.TOOL_STATS.value == "tool_stats"

    def test_handler_map_size_matches_enum_size(self):
        """HANDLER_MAP should have exactly as many entries as ToolName."""
        assert len(HANDLER_MAP) == len(ToolName)


# =====================================================================
# ToolStatus
# =====================================================================


class TestToolStatus:
    """ToolStatus enum tests."""

    def test_is_str_enum(self):
        """ToolStatus members are both str and Enum."""
        assert isinstance(ToolStatus.OK, str)
        assert ToolStatus.OK == "ok"

    def test_all_statuses(self):
        """All expected status codes exist."""
        expected = {"ok", "error", "timeout", "cached", "circuit_open",
                    "rate_limited", "blocked", "denied", "unknown_tool"}
        actual = {s.value for s in ToolStatus}
        assert expected == actual

    def test_equality_with_plain_string(self):
        """ToolStatus members compare equal to their string values."""
        assert ToolStatus.ERROR == "error"
        assert ToolStatus.TIMEOUT == "timeout"
        assert ToolStatus.CACHED == "cached"

    def test_no_duplicate_values(self):
        """No two ToolStatus members should share the same value."""
        values = [s.value for s in ToolStatus]
        assert len(values) == len(set(values))

    def test_in_check_with_frozenset(self):
        """ToolStatus works in frozenset membership checks."""
        error_set = frozenset({ToolStatus.ERROR, ToolStatus.TIMEOUT})
        assert ToolStatus.ERROR in error_set
        assert ToolStatus.TIMEOUT in error_set
        assert ToolStatus.OK not in error_set


# =====================================================================
# WRITE_TOOLS / READ_ONLY_TOOLS
# =====================================================================


class TestToolSets:
    """Tests for WRITE_TOOLS and READ_ONLY_TOOLS sets."""

    def test_write_tools_are_valid(self):
        """Every tool in WRITE_TOOLS is a valid ToolName."""
        for tool in WRITE_TOOLS:
            val = tool.value if hasattr(tool, 'value') else tool
            assert val in {t.value for t in ToolName}, f"WRITE_TOOLS has invalid tool: {tool}"

    def test_read_only_tools_are_valid(self):
        """Every tool in READ_ONLY_TOOLS is a valid ToolName."""
        for tool in READ_ONLY_TOOLS:
            val = tool.value if hasattr(tool, 'value') else tool
            assert val in {t.value for t in ToolName}, f"READ_ONLY_TOOLS has invalid tool: {tool}"

    def test_no_overlap(self):
        """WRITE_TOOLS and READ_ONLY_TOOLS should not overlap."""
        write_vals = {t.value if hasattr(t, 'value') else t for t in WRITE_TOOLS}
        read_vals = {t.value if hasattr(t, 'value') else t for t in READ_ONLY_TOOLS}
        overlap = write_vals & read_vals
        assert not overlap, f"Overlap between WRITE and READ_ONLY: {overlap}"

    def test_write_tools_minimum(self):
        """Should have at least write_file, run_command."""
        assert ToolName.WRITE_FILE in WRITE_TOOLS
        assert ToolName.RUN_COMMAND in WRITE_TOOLS

    def test_read_only_tools_minimum(self):
        """Should include common read tools."""
        assert ToolName.READ_FILE in READ_ONLY_TOOLS
        assert ToolName.LIST_DIRECTORY in READ_ONLY_TOOLS
        assert ToolName.WEB_SEARCH in READ_ONLY_TOOLS


# =====================================================================
# DesktopAction
# =====================================================================


class TestDesktopAction:
    """DesktopAction enum tests."""

    def test_is_str_enum(self):
        assert isinstance(DesktopAction.SCREENSHOT, str)
        assert DesktopAction.SCREENSHOT == "screenshot"

    def test_core_actions_exist(self):
        """All core desktop actions are defined."""
        expected = {"open_app", "focus_app", "type_text", "press_key",
                    "mouse_click", "screenshot", "ax_query", "health_check",
                    "run_macro", "smart_type", "clipboard_read", "clipboard_write"}
        actual = {a.value for a in DesktopAction}
        assert expected.issubset(actual)

    def test_smart_actions_are_desktop_actions(self):
        """SMART_ACTIONS values should be DesktopAction members."""
        desktop_values = {a.value for a in DesktopAction}
        for sa in SMART_ACTIONS:
            val = sa.value if hasattr(sa, 'value') else sa
            assert val in desktop_values, f"{sa} not in DesktopAction"

    def test_native_only_subset_of_native_capable(self):
        """NATIVE_ONLY should be a subset of NATIVE_CAPABLE."""
        assert NATIVE_ONLY_ACTIONS.issubset(NATIVE_CAPABLE_ACTIONS)

    def test_re_export_matches_source(self):
        """tools/enums re-export matches operators/enums exactly."""
        assert DesktopAction is OpDesktopAction

    def test_no_duplicate_values(self):
        values = [a.value for a in DesktopAction]
        assert len(values) == len(set(values))

    def test_minimum_action_count(self):
        """At least 30 desktop actions defined."""
        assert len(DesktopAction) >= 30


# =====================================================================
# AndroidAction
# =====================================================================


class TestAndroidAction:
    """AndroidAction enum tests."""

    def test_is_str_enum(self):
        assert isinstance(AndroidAction.TAP, str)
        assert AndroidAction.TAP == "tap"

    def test_core_actions_exist(self):
        """All core Android actions are defined."""
        expected = {"tap", "swipe", "type_text", "screenshot", "launch_app",
                    "shell", "health_check", "ui_dump", "find_element"}
        actual = {a.value for a in AndroidAction}
        assert expected.issubset(actual)

    def test_re_export_matches_source(self):
        """tools/enums re-export matches operators/enums exactly."""
        assert AndroidAction is OpAndroidAction

    def test_no_duplicate_values(self):
        values = [a.value for a in AndroidAction]
        assert len(values) == len(set(values))

    def test_minimum_action_count(self):
        """At least 25 Android actions defined."""
        assert len(AndroidAction) >= 25


# =====================================================================
# VisionAction
# =====================================================================


class TestVisionAction:
    """VisionAction enum tests."""

    def test_is_str_enum(self):
        assert isinstance(VisionAction.SCAN, str)
        assert VisionAction.SCAN == "scan"

    def test_core_actions_exist(self):
        expected = {"quick_scan", "scan", "find_and_click", "type_into",
                    "wait_for", "read_screen_text", "diff", "scan_with_ocr"}
        actual = {a.value for a in VisionAction}
        assert expected.issubset(actual)

    def test_re_export_matches_source(self):
        assert VisionAction is OpVisionAction

    def test_no_duplicate_values(self):
        values = [a.value for a in VisionAction]
        assert len(values) == len(set(values))


# =====================================================================
# ScreenshotQuality
# =====================================================================


class TestScreenshotQuality:
    """ScreenshotQuality enum tests."""

    def test_is_str_enum(self):
        assert isinstance(ScreenshotQuality.FULL, str)

    def test_all_qualities(self):
        expected = {"full", "fast", "thumbnail"}
        actual = {q.value for q in ScreenshotQuality}
        assert expected == actual

    def test_re_export_matches_source(self):
        assert ScreenshotQuality is OpScreenshotQuality
