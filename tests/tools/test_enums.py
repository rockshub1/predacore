"""
Tests for PredaCore Tool Enums — ensures enum completeness and sync with HANDLER_MAP.

Critical invariant: every tool in HANDLER_MAP must have a corresponding
ToolName enum entry, and vice versa. A mismatch means either the enum
is stale or a handler wasn't registered.
"""
import pytest

from predacore.tools.enums import (
    ToolName,
    ToolStatus,
    WRITE_TOOLS,
    READ_ONLY_TOOLS,
    DesktopAction,
    AndroidAction,
    VisionAction,
    ScreenshotQuality,
)
from predacore.tools.handlers import HANDLER_MAP
from predacore.operators.enums import (
    SMART_ACTIONS,
    NATIVE_ONLY_ACTIONS,
    NATIVE_CAPABLE_ACTIONS,
)


# ═══════════════════════════════════════════════════════════════════
# ToolName ↔ HANDLER_MAP Sync
# ═══════════════════════════════════════════════════════════════════


class TestToolNameHandlerSync:
    """Ensure ToolName enum and HANDLER_MAP are always in sync."""

    def test_every_handler_has_enum(self):
        """Every key in HANDLER_MAP must exist as a ToolName value."""
        enum_values = {e.value for e in ToolName}
        missing = set(HANDLER_MAP.keys()) - enum_values
        assert not missing, (
            f"HANDLER_MAP has tools without ToolName enum entries: {missing}"
        )

    def test_every_enum_has_handler(self):
        """Every ToolName enum must have a handler (or be explicitly excluded)."""
        handler_keys = set(HANDLER_MAP.keys())
        enum_values = {e.value for e in ToolName}
        missing = enum_values - handler_keys
        assert not missing, (
            f"ToolName enum has entries without handlers: {missing}"
        )

    def test_handler_map_size_matches_enum(self):
        assert len(HANDLER_MAP) == len(ToolName), (
            f"HANDLER_MAP has {len(HANDLER_MAP)} entries, "
            f"ToolName has {len(ToolName)} entries"
        )


# ═══════════════════════════════════════════════════════════════════
# ToolName Tests
# ═══════════════════════════════════════════════════════════════════


class TestToolName:
    def test_string_enum(self):
        """ToolName values are usable as plain strings."""
        assert ToolName.READ_FILE == "read_file"
        assert ToolName.WEB_SEARCH == "web_search"
        assert str(ToolName.PYTHON_EXEC) == "ToolName.PYTHON_EXEC"

    def test_no_duplicates(self):
        """No two enum members should share a value."""
        values = [e.value for e in ToolName]
        assert len(values) == len(set(values)), "Duplicate ToolName values found"

    def test_minimum_tool_count(self):
        """We should have at least 30 tools (sanity check)."""
        assert len(ToolName) >= 30, f"Only {len(ToolName)} tools — expected 30+"


# ═══════════════════════════════════════════════════════════════════
# ToolStatus Tests
# ═══════════════════════════════════════════════════════════════════


class TestToolStatus:
    def test_core_statuses_exist(self):
        assert ToolStatus.OK == "ok"
        assert ToolStatus.ERROR == "error"
        assert ToolStatus.TIMEOUT == "timeout"
        assert ToolStatus.CACHED == "cached"
        assert ToolStatus.CIRCUIT_OPEN == "circuit_open"
        assert ToolStatus.RATE_LIMITED == "rate_limited"
        assert ToolStatus.BLOCKED == "blocked"

    def test_no_duplicates(self):
        values = [e.value for e in ToolStatus]
        assert len(values) == len(set(values))


# ═══════════════════════════════════════════════════════════════════
# Write/Read Tool Sets
# ═══════════════════════════════════════════════════════════════════


class TestToolSets:
    def test_write_tools_are_valid_tool_names(self):
        """All WRITE_TOOLS must be valid ToolName values."""
        enum_values = {e.value for e in ToolName}
        for tool in WRITE_TOOLS:
            assert tool in enum_values or str(tool) in enum_values, (
                f"WRITE_TOOL '{tool}' is not a valid ToolName"
            )

    def test_read_only_tools_are_valid_tool_names(self):
        enum_values = {e.value for e in ToolName}
        for tool in READ_ONLY_TOOLS:
            assert tool in enum_values or str(tool) in enum_values, (
                f"READ_ONLY_TOOL '{tool}' is not a valid ToolName"
            )

    def test_no_overlap_read_write(self):
        """A tool can't be both read-only and a write tool."""
        overlap = WRITE_TOOLS & READ_ONLY_TOOLS
        assert not overlap, f"Tools are both read and write: {overlap}"

    def test_write_tools_non_empty(self):
        assert len(WRITE_TOOLS) >= 2

    def test_read_tools_non_empty(self):
        assert len(READ_ONLY_TOOLS) >= 5


# ═══════════════════════════════════════════════════════════════════
# Desktop Action Enums
# ═══════════════════════════════════════════════════════════════════


class TestDesktopAction:
    def test_core_actions_exist(self):
        assert DesktopAction.OPEN_APP == "open_app"
        assert DesktopAction.SCREENSHOT == "screenshot"
        assert DesktopAction.TYPE_TEXT == "type_text"
        assert DesktopAction.MOUSE_CLICK == "mouse_click"
        assert DesktopAction.AX_QUERY == "ax_query"
        assert DesktopAction.RUN_MACRO == "run_macro"

    def test_smart_actions_are_desktop_actions(self):
        desktop_values = {e.value for e in DesktopAction}
        for action in SMART_ACTIONS:
            assert action in desktop_values, (
                f"SMART_ACTION '{action}' not in DesktopAction"
            )

    def test_native_only_are_desktop_actions(self):
        desktop_values = {e.value for e in DesktopAction}
        for action in NATIVE_ONLY_ACTIONS:
            assert action in desktop_values, (
                f"NATIVE_ONLY '{action}' not in DesktopAction"
            )

    def test_native_capable_superset_of_native_only(self):
        assert NATIVE_ONLY_ACTIONS.issubset(NATIVE_CAPABLE_ACTIONS)

    def test_no_duplicate_values(self):
        values = [e.value for e in DesktopAction]
        assert len(values) == len(set(values))


# ═══════════════════════════════════════════════════════════════════
# Android Action Enums
# ═══════════════════════════════════════════════════════════════════


class TestAndroidAction:
    def test_core_actions_exist(self):
        assert AndroidAction.TAP == "tap"
        assert AndroidAction.SWIPE == "swipe"
        assert AndroidAction.SCREENSHOT == "screenshot"
        assert AndroidAction.LAUNCH_APP == "launch_app"
        assert AndroidAction.SHELL == "shell"
        assert AndroidAction.RUN_MACRO == "run_macro"

    def test_no_duplicate_values(self):
        values = [e.value for e in AndroidAction]
        assert len(values) == len(set(values))

    def test_minimum_action_count(self):
        assert len(AndroidAction) >= 20


# ═══════════════════════════════════════════════════════════════════
# Vision Action Enums
# ═══════════════════════════════════════════════════════════════════


class TestVisionAction:
    def test_core_actions_exist(self):
        assert VisionAction.QUICK_SCAN == "quick_scan"
        assert VisionAction.SCAN == "scan"
        assert VisionAction.FIND_AND_CLICK == "find_and_click"
        assert VisionAction.READ_SCREEN_TEXT == "read_screen_text"

    def test_no_duplicate_values(self):
        values = [e.value for e in VisionAction]
        assert len(values) == len(set(values))


# ═══════════════════════════════════════════════════════════════════
# ScreenshotQuality Enum
# ═══════════════════════════════════════════════════════════════════


class TestScreenshotQuality:
    def test_all_qualities(self):
        assert ScreenshotQuality.FULL == "full"
        assert ScreenshotQuality.FAST == "fast"
        assert ScreenshotQuality.THUMBNAIL == "thumbnail"

    def test_exactly_three(self):
        assert len(ScreenshotQuality) == 3
