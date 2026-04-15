"""
JARVIS Operator Enums — Eliminates magic strings in operator dispatch.

These enums are the single source of truth for action names across all
operators. They live here (not in tools/) because operators own the actions
and tools depend on operators, not the other way around.

tools/enums.py re-exports these so the rest of the codebase can import
from either location.
"""
from __future__ import annotations

from enum import Enum


# ---------------------------------------------------------------------------
# Desktop Actions — used in MacDesktopOperator.execute()
# ---------------------------------------------------------------------------


class DesktopAction(str, Enum):
    """macOS desktop operator action names."""

    # App control
    OPEN_APP = "open_app"
    FOCUS_APP = "focus_app"
    FRONTMOST_APP = "frontmost_app"
    OPEN_URL = "open_url"

    # Keyboard
    TYPE_TEXT = "type_text"
    PRESS_KEY = "press_key"

    # Mouse
    MOUSE_MOVE = "mouse_move"
    MOUSE_CLICK = "mouse_click"
    MOUSE_DOUBLE_CLICK = "mouse_double_click"
    MOUSE_RIGHT_CLICK = "mouse_right_click"
    MOUSE_DRAG = "mouse_drag"
    MOUSE_SCROLL = "mouse_scroll"
    GET_MOUSE_POSITION = "get_mouse_position"

    # Screenshot
    SCREENSHOT = "screenshot"

    # Clipboard
    CLIPBOARD_READ = "clipboard_read"
    CLIPBOARD_WRITE = "clipboard_write"

    # Window management
    LIST_WINDOWS = "list_windows"
    MOVE_WINDOW = "move_window"
    RESIZE_WINDOW = "resize_window"
    MINIMIZE_WINDOW = "minimize_window"

    # Monitors
    LIST_MONITORS = "list_monitors"
    MOVE_TO_MONITOR = "move_to_monitor"

    # Menu bar
    CLICK_MENU = "click_menu"

    # System hardware
    SET_VOLUME = "set_volume"
    GET_VOLUME = "get_volume"
    SET_BRIGHTNESS = "set_brightness"
    GET_BRIGHTNESS = "get_brightness"

    # Accessibility
    AX_QUERY = "ax_query"
    AX_CLICK = "ax_click"
    AX_SET_VALUE = "ax_set_value"
    AX_REQUEST_ACCESS = "ax_request_access"

    # Smart input (via SmartInputEngine)
    SMART_TYPE = "smart_type"
    SMART_RUN_COMMAND = "smart_run_command"
    SMART_CREATE_NOTE = "smart_create_note"

    # Macro
    RUN_MACRO = "run_macro"

    # System
    HEALTH_CHECK = "health_check"
    SLEEP = "sleep"

    # System info & control (native PyObjC — no AppleScript)
    LIST_APPS = "list_apps"
    FORCE_QUIT_APP = "force_quit_app"
    GET_SYSTEM_INFO = "get_system_info"
    TOGGLE_DARK_MODE = "toggle_dark_mode"
    GET_DARK_MODE = "get_dark_mode"
    TOGGLE_DND = "toggle_dnd"
    SPOTLIGHT_SEARCH = "spotlight_search"
    OPEN_FILE = "open_file"
    GET_FINDER_SELECTION = "get_finder_selection"
    SCREEN_RECORD_START = "screen_record_start"
    SCREEN_RECORD_STOP = "screen_record_stop"
    TILE_WINDOWS = "tile_windows"
    APP_SWITCH = "app_switch"
    GET_RUNNING_PROCESSES = "get_running_processes"


# Grouped sets for dispatch logic
SMART_ACTIONS: frozenset[str] = frozenset({
    DesktopAction.SMART_TYPE,
    DesktopAction.SMART_RUN_COMMAND,
    DesktopAction.SMART_CREATE_NOTE,
})

NATIVE_ONLY_ACTIONS: frozenset[str] = frozenset({
    DesktopAction.AX_QUERY,
    DesktopAction.AX_CLICK,
    DesktopAction.AX_SET_VALUE,
    DesktopAction.AX_REQUEST_ACCESS,
    DesktopAction.CLIPBOARD_READ,
    DesktopAction.CLIPBOARD_WRITE,
    DesktopAction.LIST_WINDOWS,
    DesktopAction.MOVE_WINDOW,
    DesktopAction.RESIZE_WINDOW,
    DesktopAction.MINIMIZE_WINDOW,
    DesktopAction.LIST_MONITORS,
    DesktopAction.MOVE_TO_MONITOR,
    DesktopAction.MOUSE_DRAG,
    DesktopAction.CLICK_MENU,
    DesktopAction.SET_VOLUME,
    DesktopAction.GET_VOLUME,
    DesktopAction.SET_BRIGHTNESS,
    DesktopAction.GET_BRIGHTNESS,
    # New system actions (native PyObjC only)
    DesktopAction.LIST_APPS,
    DesktopAction.FORCE_QUIT_APP,
    DesktopAction.GET_SYSTEM_INFO,
    DesktopAction.TOGGLE_DARK_MODE,
    DesktopAction.GET_DARK_MODE,
    DesktopAction.TOGGLE_DND,
    DesktopAction.SPOTLIGHT_SEARCH,
    DesktopAction.OPEN_FILE,
    DesktopAction.GET_FINDER_SELECTION,
    DesktopAction.SCREEN_RECORD_START,
    DesktopAction.SCREEN_RECORD_STOP,
    DesktopAction.TILE_WINDOWS,
    DesktopAction.APP_SWITCH,
    DesktopAction.GET_RUNNING_PROCESSES,
})

NATIVE_CAPABLE_ACTIONS: frozenset[str] = frozenset({
    DesktopAction.OPEN_APP,
    DesktopAction.FOCUS_APP,
    DesktopAction.OPEN_URL,
    DesktopAction.TYPE_TEXT,
    DesktopAction.PRESS_KEY,
    DesktopAction.MOUSE_MOVE,
    DesktopAction.MOUSE_CLICK,
    DesktopAction.MOUSE_DOUBLE_CLICK,
    DesktopAction.MOUSE_RIGHT_CLICK,
    DesktopAction.MOUSE_SCROLL,
    DesktopAction.GET_MOUSE_POSITION,
    DesktopAction.FRONTMOST_APP,
    DesktopAction.MOUSE_DRAG,
}) | NATIVE_ONLY_ACTIONS


# ---------------------------------------------------------------------------
# Screenshot Quality — used in desktop screenshot
# ---------------------------------------------------------------------------


class ScreenshotQuality(str, Enum):
    """Screenshot quality modes."""

    FULL = "full"              # PNG, full resolution
    FAST = "fast"              # JPEG 60%, full resolution
    THUMBNAIL = "thumbnail"    # JPEG 30%, resized to 960px wide


# ---------------------------------------------------------------------------
# Android Actions — used in AndroidOperator.execute()
# ---------------------------------------------------------------------------


class AndroidAction(str, Enum):
    """Android operator action names."""

    # Touch / gestures
    TAP = "tap"
    LONG_PRESS = "long_press"
    SWIPE = "swipe"
    PINCH = "pinch"

    # Text / keys
    TYPE_TEXT = "type_text"
    PRESS_KEY = "press_key"

    # UI inspection
    UI_DUMP = "ui_dump"
    FIND_ELEMENT = "find_element"
    FIND_AND_TAP = "find_and_tap"
    FIND_AND_TYPE = "find_and_type"
    WAIT_FOR_ELEMENT = "wait_for_element"

    # Screenshot / recording
    SCREENSHOT = "screenshot"
    SCREEN_RECORD = "screen_record"

    # App management
    LAUNCH_APP = "launch_app"
    STOP_APP = "stop_app"
    CURRENT_APP = "current_app"
    LIST_PACKAGES = "list_packages"
    INSTALL_APK = "install_apk"

    # Device control
    SHELL = "shell"
    WAKE = "wake"
    SLEEP_DEVICE = "sleep_device"
    SCREEN_SIZE = "screen_size"
    SCREEN_STATE = "screen_state"

    # File transfer
    PUSH_FILE = "push_file"
    PULL_FILE = "pull_file"

    # System
    HEALTH_CHECK = "health_check"
    RUN_MACRO = "run_macro"

    # Extended capabilities (ADB shell — no extra deps)
    GET_NOTIFICATIONS = "get_notifications"
    CLEAR_NOTIFICATIONS = "clear_notifications"
    TOGGLE_WIFI = "toggle_wifi"
    TOGGLE_BLUETOOTH = "toggle_bluetooth"
    TOGGLE_AIRPLANE = "toggle_airplane"
    TOGGLE_FLASHLIGHT = "toggle_flashlight"
    GET_CLIPBOARD = "get_clipboard"
    SET_CLIPBOARD = "set_clipboard"
    GET_DEVICE_INFO = "get_device_info"
    OPEN_URL = "open_url"
    CLEAR_APP_DATA = "clear_app_data"
    UNINSTALL_APP = "uninstall_app"
    SCROLL_TO_ELEMENT = "scroll_to_element"
    GET_LOGCAT = "get_logcat"
    DOUBLE_TAP = "double_tap"
    ROTATE_SCREEN = "rotate_screen"
    SET_BRIGHTNESS = "set_brightness"
    OPEN_SETTINGS = "open_settings"
    SEND_BROADCAST = "send_broadcast"
    GET_WIFI_INFO = "get_wifi_info"
    GET_BATTERY_INFO = "get_battery_info"
    LIST_RUNNING_APPS = "list_running_apps"
    INPUT_KEYCOMBO = "input_keycombo"

    # Full-page content capture
    SCROLL_AND_COLLECT_ALL = "scroll_and_collect_all"
    WAIT_FOR_TEXT = "wait_for_text"
    SCREENSHOT_AND_OCR = "screenshot_and_ocr"
    SEARCH_IN_APP = "search_in_app"
    GET_FOCUSED_ELEMENT = "get_focused_element"
    GESTURE = "gesture"
    CONNECT_CHROME_ON_DEVICE = "connect_chrome_on_device"

    # Smart automation (handles dynamic UI)
    WAIT_FOR_STABLE_UI = "wait_for_stable_ui"
    ENSURE_IN_APP = "ensure_in_app"
    SMART_TAP = "smart_tap"
    SMART_TYPE = "smart_type"


# ---------------------------------------------------------------------------
# Screen Vision Actions — used in ScreenVisionEngine
# ---------------------------------------------------------------------------


class VisionAction(str, Enum):
    """Screen vision engine action names."""

    QUICK_SCAN = "quick_scan"
    SCAN = "scan"
    SCAN_WITH_VISION = "scan_with_vision"
    SCAN_WITH_OCR = "scan_with_ocr"
    FIND_AND_CLICK = "find_and_click"
    TYPE_INTO = "type_into"
    WAIT_FOR = "wait_for"
    READ_SCREEN_TEXT = "read_screen_text"
    DIFF = "diff"
