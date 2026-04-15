"""
Tests for JARVIS BrowserBridge — DOM access to Chrome (CDP) and Safari (AppleScript).

Covers:
  - BrowserBridge connection: auto-detect, Safari fallback, Chrome CDP
  - read_text: full page text extraction, selector-based, empty page
  - navigate: URL navigation, SPA wait time, content_loaded detection
  - click: by text, by selector, by role
  - type_text: into input fields
  - scroll: up/down with amounts
  - evaluate_js: arbitrary JS execution, error handling
  - get_page_tree: DOM tree extraction, element format
  - get_page_links: link extraction with text and URLs
  - get_page_images: image extraction with alt text
  - Media controls: get_media_info, play, pause, seek, volume
  - Captions: get_captions, enable_captions
  - Error handling: disconnected state, timeout, invalid selectors

All backends are mocked — no real browsers are launched.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from jarvis.operators.browser_bridge import (
    BrowserBridge,
    _ChromeCDP,
    _SafariJSBridge,
    _frontmost_app,
    _js_esc,
    _CHROMIUM_NAMES,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def bridge():
    """Return a fresh BrowserBridge with no backend connected."""
    return BrowserBridge(cdp_port=9222)


@pytest.fixture
def mock_cdp():
    """Return a mock _ChromeCDP backend."""
    cdp = MagicMock(spec=_ChromeCDP)
    cdp.connected = True
    cdp.browser_label = "Google Chrome"
    cdp.evaluate_js = AsyncMock()
    cdp.send = AsyncMock()
    cdp.connect = AsyncMock(return_value=True)
    cdp.close = AsyncMock()
    return cdp


@pytest.fixture
def mock_safari():
    """Return a mock _SafariJSBridge backend."""
    safari = MagicMock(spec=_SafariJSBridge)
    safari.connected = True
    safari.evaluate_js = AsyncMock()
    safari.connect = AsyncMock(return_value=True)
    safari.close = AsyncMock()
    safari._applescript = AsyncMock(return_value="OK")
    return safari


@pytest.fixture
def chrome_bridge(bridge, mock_cdp):
    """Return a BrowserBridge wired to a mock Chrome CDP backend."""
    bridge._backend = mock_cdp
    bridge._browser = "Google Chrome"
    bridge._cdp = mock_cdp
    return bridge


@pytest.fixture
def safari_bridge(bridge, mock_safari):
    """Return a BrowserBridge wired to a mock Safari backend."""
    bridge._backend = mock_safari
    bridge._browser = "Safari"
    bridge._safari = mock_safari
    return bridge


# ═══════════════════════════════════════════════════════════════════
# Helper Utilities
# ═══════════════════════════════════════════════════════════════════


class TestJsEscape:
    def test_escapes_double_quotes(self):
        assert _js_esc('say "hi"') == 'say \\"hi\\"'

    def test_escapes_backslashes(self):
        assert _js_esc("back\\slash") == "back\\\\slash"

    def test_escapes_newlines(self):
        assert _js_esc("line1\nline2") == "line1\\nline2"

    def test_empty_string(self):
        assert _js_esc("") == ""

    def test_plain_string_unchanged(self):
        assert _js_esc("hello") == "hello"


# ═══════════════════════════════════════════════════════════════════
# Connection Tests
# ═══════════════════════════════════════════════════════════════════


class TestBrowserBridgeConnection:
    def test_initially_disconnected(self, bridge):
        assert bridge.connected is False
        assert bridge.browser_name == ""

    @pytest.mark.asyncio
    async def test_connect_chrome_explicit(self, bridge):
        bridge._cdp.connect = AsyncMock(return_value=True)
        result = await bridge.connect(browser="chrome")
        assert result is True
        assert bridge._backend is bridge._cdp
        assert bridge._browser == "chrome"

    @pytest.mark.asyncio
    async def test_connect_safari_explicit(self, bridge):
        bridge._safari.connect = AsyncMock(return_value=True)
        result = await bridge.connect(browser="safari")
        assert result is True
        assert bridge._backend is bridge._safari
        assert bridge._browser == "Safari"

    @pytest.mark.asyncio
    async def test_connect_safari_failure(self, bridge):
        bridge._safari.connect = AsyncMock(return_value=False)
        result = await bridge.connect(browser="safari")
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_chrome_failure(self, bridge):
        bridge._cdp.connect = AsyncMock(return_value=False)
        result = await bridge.connect(browser="chrome")
        assert result is False

    @pytest.mark.asyncio
    async def test_auto_detect_safari_frontmost(self, bridge):
        bridge._safari.connect = AsyncMock(return_value=True)
        with patch("jarvis.operators.browser_bridge._frontmost_app", new_callable=AsyncMock, return_value="Safari"):
            result = await bridge.connect(browser="auto")
        assert result is True
        assert bridge._browser == "Safari"

    @pytest.mark.asyncio
    async def test_auto_detect_chrome_frontmost(self, bridge):
        bridge._cdp.connect = AsyncMock(return_value=True)
        with patch("jarvis.operators.browser_bridge._frontmost_app", new_callable=AsyncMock, return_value="Google Chrome"):
            result = await bridge.connect(browser="auto")
        assert result is True
        assert bridge._browser == "Google Chrome"

    @pytest.mark.asyncio
    async def test_auto_detect_chromium_variant(self, bridge):
        """Auto-detect should recognize all Chromium-based browsers."""
        bridge._cdp.connect = AsyncMock(return_value=True)
        for name in ("Brave Browser", "Microsoft Edge", "Arc", "Vivaldi"):
            assert name in _CHROMIUM_NAMES
            with patch("jarvis.operators.browser_bridge._frontmost_app", new_callable=AsyncMock, return_value=name):
                result = await bridge.connect(browser="auto")
            assert result is True
            # Reset for next iteration
            bridge._backend = None
            bridge._browser = ""

    @pytest.mark.asyncio
    async def test_auto_fallback_to_background_chrome(self, bridge):
        """When frontmost is not a browser, try background Chrome, then Safari."""
        bridge._cdp.connect = AsyncMock(return_value=True)
        with patch("jarvis.operators.browser_bridge._frontmost_app", new_callable=AsyncMock, return_value="Finder"):
            result = await bridge.connect(browser="auto")
        assert result is True
        assert bridge._browser == "Chrome"

    @pytest.mark.asyncio
    async def test_auto_fallback_to_safari_when_no_chrome(self, bridge):
        """When frontmost is not a browser and Chrome CDP fails, fall back to Safari."""
        bridge._cdp.connect = AsyncMock(return_value=False)
        bridge._safari.connect = AsyncMock(return_value=True)
        with patch("jarvis.operators.browser_bridge._frontmost_app", new_callable=AsyncMock, return_value="Finder"):
            result = await bridge.connect(browser="auto")
        assert result is True
        assert bridge._browser == "Safari"

    @pytest.mark.asyncio
    async def test_auto_no_browser_available(self, bridge):
        """When nothing is available, return False."""
        bridge._cdp.connect = AsyncMock(return_value=False)
        bridge._safari.connect = AsyncMock(return_value=False)
        with patch("jarvis.operators.browser_bridge._frontmost_app", new_callable=AsyncMock, return_value="Finder"):
            result = await bridge.connect(browser="auto")
        assert result is False

    @pytest.mark.asyncio
    async def test_auto_chrome_frontmost_but_cdp_off(self, bridge):
        """Chrome is frontmost but CDP is not enabled."""
        bridge._cdp.connect = AsyncMock(return_value=False)
        with patch("jarvis.operators.browser_bridge._frontmost_app", new_callable=AsyncMock, return_value="Google Chrome"):
            result = await bridge.connect(browser="auto")
        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect(self, chrome_bridge, mock_cdp):
        assert chrome_bridge.connected is True
        await chrome_bridge.disconnect()
        mock_cdp.close.assert_awaited_once()
        assert chrome_bridge._backend is None
        assert chrome_bridge.browser_name == ""

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, bridge):
        """Disconnect on a bridge with no backend should not raise."""
        await bridge.disconnect()
        assert bridge.connected is False

    def test_connected_property_with_backend(self, chrome_bridge, mock_cdp):
        mock_cdp.connected = True
        assert chrome_bridge.connected is True

    def test_connected_property_backend_disconnected(self, chrome_bridge, mock_cdp):
        mock_cdp.connected = False
        assert chrome_bridge.connected is False


# ═══════════════════════════════════════════════════════════════════
# read_text Tests
# ═══════════════════════════════════════════════════════════════════


class TestReadText:
    @pytest.mark.asyncio
    async def test_full_page_text(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value="Page: Example\nHeading 1\nSome content")
        text = await chrome_bridge.read_text()
        assert "Page: Example" in text
        assert "Heading 1" in text
        mock_cdp.evaluate_js.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_selector_based_text(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value="button label")
        text = await chrome_bridge.read_text(selector="#my-btn")
        assert text == "button label"
        # The JS should reference querySelector with the selector
        call_args = mock_cdp.evaluate_js.call_args
        js_code = call_args[0][0] if call_args[0] else call_args[1].get("code", "")
        assert "#my-btn" in js_code

    @pytest.mark.asyncio
    async def test_empty_page(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value="")
        text = await chrome_bridge.read_text()
        assert text == ""

    @pytest.mark.asyncio
    async def test_none_result(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value=None)
        text = await chrome_bridge.read_text()
        assert text == ""

    @pytest.mark.asyncio
    async def test_disconnected_returns_empty(self, bridge):
        text = await bridge.read_text()
        assert text == ""


# ═══════════════════════════════════════════════════════════════════
# navigate Tests
# ═══════════════════════════════════════════════════════════════════


class TestNavigate:
    @pytest.mark.asyncio
    async def test_chrome_navigation(self, chrome_bridge, mock_cdp):
        mock_cdp.send = AsyncMock(return_value={"frameId": "123"})
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await chrome_bridge.navigate("https://example.com")
        assert result["ok"] is True
        assert result["url"] == "https://example.com"
        mock_cdp.send.assert_awaited_once_with("Page.navigate", {"url": "https://example.com"})

    @pytest.mark.asyncio
    async def test_chrome_navigation_error(self, chrome_bridge, mock_cdp):
        mock_cdp.send = AsyncMock(return_value={"error": "net::ERR_NAME_NOT_RESOLVED"})
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await chrome_bridge.navigate("https://nonexistent.invalid")
        assert result["ok"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_chrome_navigation_waits_for_spa(self, chrome_bridge, mock_cdp):
        """Navigate should sleep 4s for SPA hydration."""
        mock_cdp.send = AsyncMock(return_value={"frameId": "abc"})
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await chrome_bridge.navigate("https://react-app.com")
        mock_sleep.assert_awaited_once_with(4.0)

    @pytest.mark.asyncio
    async def test_safari_navigation(self, safari_bridge, mock_safari):
        mock_safari._applescript = AsyncMock(return_value="OK")
        mock_safari.evaluate_js = AsyncMock(side_effect=["Example Page", 5000])
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await safari_bridge.navigate("https://example.com")
        assert result["ok"] is True
        assert result["url"] == "https://example.com"
        assert result["title"] == "Example Page"
        assert result["content_loaded"] is True

    @pytest.mark.asyncio
    async def test_safari_navigation_content_not_loaded(self, safari_bridge, mock_safari):
        mock_safari._applescript = AsyncMock(return_value="OK")
        mock_safari.evaluate_js = AsyncMock(side_effect=["", 50])
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await safari_bridge.navigate("https://slow-page.com")
        assert result["content_loaded"] is False

    @pytest.mark.asyncio
    async def test_safari_navigation_applescript_fails(self, safari_bridge, mock_safari):
        mock_safari._applescript = AsyncMock(return_value=None)
        mock_safari.evaluate_js = AsyncMock(side_effect=["", 0])
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await safari_bridge.navigate("https://fail.com")
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_navigate_disconnected(self, bridge):
        result = await bridge.navigate("https://example.com")
        assert result["ok"] is False
        assert "Not connected" in result["error"]


# ═══════════════════════════════════════════════════════════════════
# click Tests
# ═══════════════════════════════════════════════════════════════════


class TestClick:
    @pytest.mark.asyncio
    async def test_click_by_selector(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"ok": True, "tag": "button", "text": "Sign In"})
        result = await chrome_bridge.click(selector="#sign-in")
        assert result["ok"] is True
        # JS should include the selector
        call_args = mock_cdp.evaluate_js.call_args
        js_code = call_args[0][0]
        assert "#sign-in" in js_code

    @pytest.mark.asyncio
    async def test_click_by_text(self, chrome_bridge, mock_cdp):
        # First call resolves the text to a selector, second performs the click
        mock_cdp.evaluate_js = AsyncMock(
            side_effect=["#resolved-selector", {"ok": True, "tag": "a", "text": "Login"}]
        )
        result = await chrome_bridge.click(text="Login")
        assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_click_by_role(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(
            side_effect=["#nav-btn", {"ok": True, "tag": "button", "text": "Nav"}]
        )
        result = await chrome_bridge.click(role="navigation")
        assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_click_element_not_found_by_text(self, chrome_bridge, mock_cdp):
        # _resolve returns None when no element matches
        mock_cdp.evaluate_js = AsyncMock(return_value=None)
        result = await chrome_bridge.click(text="Nonexistent")
        assert result["ok"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_click_selector_not_in_dom(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"ok": False, "error": "not found: #missing"})
        result = await chrome_bridge.click(selector="#missing")
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_click_disconnected(self, bridge):
        result = await bridge.click(selector="#btn")
        assert result["ok"] is False
        assert "Not connected" in result["error"]


# ═══════════════════════════════════════════════════════════════════
# type_text Tests
# ═══════════════════════════════════════════════════════════════════


class TestTypeText:
    @pytest.mark.asyncio
    async def test_type_by_selector(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"ok": True, "tag": "input"})
        result = await chrome_bridge.type_text(selector="#email", value="j@example.com")
        assert result["ok"] is True
        assert result["tag"] == "input"
        js_code = mock_cdp.evaluate_js.call_args[0][0]
        assert "#email" in js_code
        assert "j@example.com" in js_code

    @pytest.mark.asyncio
    async def test_type_by_text_label(self, chrome_bridge, mock_cdp):
        # _resolve finds the selector, then type_text injects the value
        mock_cdp.evaluate_js = AsyncMock(
            side_effect=["#email-field", {"ok": True, "tag": "input"}]
        )
        result = await chrome_bridge.type_text(text="Email", value="test@test.com")
        assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_type_input_not_found(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value=None)
        result = await chrome_bridge.type_text(text="Missing Input", value="anything")
        assert result["ok"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_type_selector_not_in_dom(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"ok": False, "error": "selector not found"})
        result = await chrome_bridge.type_text(selector="#missing", value="test")
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_type_dispatches_input_and_change_events(self, chrome_bridge, mock_cdp):
        """The JS should dispatch both 'input' and 'change' events."""
        mock_cdp.evaluate_js = AsyncMock(return_value={"ok": True, "tag": "input"})
        await chrome_bridge.type_text(selector="#field", value="val")
        js_code = mock_cdp.evaluate_js.call_args[0][0]
        assert "input" in js_code
        assert "change" in js_code

    @pytest.mark.asyncio
    async def test_type_disconnected(self, bridge):
        result = await bridge.type_text(selector="#field", value="text")
        assert result["ok"] is False
        assert "Not connected" in result["error"]


# ═══════════════════════════════════════════════════════════════════
# scroll Tests
# ═══════════════════════════════════════════════════════════════════


class TestScroll:
    @pytest.mark.asyncio
    async def test_scroll_down(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value="ok")
        result = await chrome_bridge.scroll(direction="down", amount=3)
        assert result["ok"] is True
        assert result["direction"] == "down"
        assert result["pixels"] == 900  # 3 * 300

    @pytest.mark.asyncio
    async def test_scroll_up(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value="ok")
        result = await chrome_bridge.scroll(direction="up", amount=2)
        assert result["ok"] is True
        assert result["direction"] == "up"
        assert result["pixels"] == -600  # 2 * 300 * -1

    @pytest.mark.asyncio
    async def test_scroll_custom_amount(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value="ok")
        result = await chrome_bridge.scroll(direction="down", amount=5)
        assert result["pixels"] == 1500

    @pytest.mark.asyncio
    async def test_scroll_default_direction(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value="ok")
        result = await chrome_bridge.scroll()
        assert result["direction"] == "down"
        assert result["pixels"] == 900  # default amount=3

    @pytest.mark.asyncio
    async def test_scroll_js_failure(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value=None)
        result = await chrome_bridge.scroll()
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_scroll_disconnected(self, bridge):
        result = await bridge.scroll()
        assert result["ok"] is False
        assert "Not connected" in result["error"]


# ═══════════════════════════════════════════════════════════════════
# evaluate_js Tests
# ═══════════════════════════════════════════════════════════════════


class TestEvaluateJs:
    @pytest.mark.asyncio
    async def test_execute_simple_js(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value=42)
        result = await chrome_bridge.evaluate_js("2 + 40")
        assert result == 42

    @pytest.mark.asyncio
    async def test_execute_js_returns_string(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value="hello world")
        result = await chrome_bridge.evaluate_js("document.title")
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_execute_js_returns_dict(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"key": "value"})
        result = await chrome_bridge.evaluate_js("JSON.parse('{}')")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_execute_js_backend_error(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"error": "ReferenceError: x is not defined"})
        result = await chrome_bridge.evaluate_js("x.nonexistent")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_execute_js_exception(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(side_effect=RuntimeError("WebSocket closed"))
        result = await chrome_bridge.evaluate_js("anything")
        assert "error" in result
        assert "WebSocket closed" in result["error"]

    @pytest.mark.asyncio
    async def test_evaluate_js_disconnected(self, bridge):
        result = await bridge.evaluate_js("1+1")
        assert "error" in result
        assert "Not connected" in result["error"]


# ═══════════════════════════════════════════════════════════════════
# get_page_tree Tests
# ═══════════════════════════════════════════════════════════════════


class TestGetPageTree:
    @pytest.mark.asyncio
    async def test_returns_elements(self, chrome_bridge, mock_cdp):
        page_data = {
            "url": "https://example.com",
            "title": "Example",
            "elements": [
                {
                    "tag": "button", "role": "button", "label": "Submit",
                    "value": "", "bounds": {"x": 10, "y": 20, "w": 100, "h": 40},
                    "selector": "button.submit", "clickable": True, "typeable": False,
                },
                {
                    "tag": "input", "role": "input", "label": "Email",
                    "value": "test@test.com", "bounds": {"x": 10, "y": 80, "w": 200, "h": 30},
                    "selector": "input#email", "clickable": False, "typeable": True,
                },
            ],
            "timestamp": 1234567890,
        }
        mock_cdp.evaluate_js = AsyncMock(return_value=page_data)
        result = await chrome_bridge.get_page_tree()

        assert result["url"] == "https://example.com"
        assert result["title"] == "Example"
        assert result["element_count"] == 2
        assert result["browser"] == "Google Chrome"
        assert "scan_ms" in result

        # Check element format
        el = result["elements"][0]
        assert el["role"] == "button"
        assert el["label"] == "Submit"
        assert el["source"] == "browser"
        assert el["selector"] == "button.submit"
        assert el["clickable"] is True
        assert el["typeable"] is False

    @pytest.mark.asyncio
    async def test_empty_page(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={
            "url": "about:blank", "title": "", "elements": [], "timestamp": 0,
        })
        result = await chrome_bridge.get_page_tree()
        assert result["elements"] == []
        assert result["element_count"] == 0

    @pytest.mark.asyncio
    async def test_error_from_js(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"error": "Execution context was destroyed"})
        result = await chrome_bridge.get_page_tree()
        assert "error" in result
        assert result["elements"] == []

    @pytest.mark.asyncio
    async def test_bad_return_type(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value="not a dict")
        result = await chrome_bridge.get_page_tree()
        assert "error" in result
        assert "Bad DOM result" in result["error"]

    @pytest.mark.asyncio
    async def test_disconnected(self, bridge):
        result = await bridge.get_page_tree()
        assert "error" in result
        assert "Not connected" in result["error"]
        assert result["elements"] == []


# ═══════════════════════════════════════════════════════════════════
# get_page_links Tests
# ═══════════════════════════════════════════════════════════════════


class TestGetPageLinks:
    @pytest.mark.asyncio
    async def test_extracts_links(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={
            "count": 2,
            "links": [
                {"text": "Home", "href": "https://example.com/", "target": ""},
                {"text": "About", "href": "https://example.com/about", "target": "_blank"},
            ],
        })
        result = await chrome_bridge.get_page_links()
        assert result["count"] == 2
        assert len(result["links"]) == 2
        assert result["links"][0]["text"] == "Home"
        assert result["links"][1]["href"] == "https://example.com/about"
        assert result["links"][1]["target"] == "_blank"

    @pytest.mark.asyncio
    async def test_no_links(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"count": 0, "links": []})
        result = await chrome_bridge.get_page_links()
        assert result["count"] == 0
        assert result["links"] == []

    @pytest.mark.asyncio
    async def test_bad_return_type(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value="unexpected")
        result = await chrome_bridge.get_page_links()
        assert result["count"] == 0
        assert result["links"] == []

    @pytest.mark.asyncio
    async def test_disconnected(self, bridge):
        # get_page_links calls evaluate_js which returns {"error": ...} when disconnected
        result = await bridge.get_page_links()
        assert result["count"] == 0


# ═══════════════════════════════════════════════════════════════════
# get_page_images Tests
# ═══════════════════════════════════════════════════════════════════


class TestGetPageImages:
    @pytest.mark.asyncio
    async def test_extracts_images(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={
            "count": 2,
            "images": [
                {"alt": "Logo", "src": "https://example.com/logo.png", "width": 200, "height": 100, "visible": True},
                {"alt": "", "src": "https://example.com/bg.jpg", "width": 1920, "height": 1080, "visible": True},
            ],
        })
        result = await chrome_bridge.get_page_images()
        assert result["count"] == 2
        assert result["images"][0]["alt"] == "Logo"
        assert result["images"][1]["width"] == 1920

    @pytest.mark.asyncio
    async def test_no_images(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"count": 0, "images": []})
        result = await chrome_bridge.get_page_images()
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_bad_return_type(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value=None)
        result = await chrome_bridge.get_page_images()
        assert result["count"] == 0
        assert result["images"] == []


# ═══════════════════════════════════════════════════════════════════
# Media Controls Tests
# ═══════════════════════════════════════════════════════════════════


class TestMediaControls:
    @pytest.mark.asyncio
    async def test_get_media_info(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={
            "count": 1,
            "elements": [{
                "tag": "video", "src": "https://cdn.example.com/video.mp4",
                "duration": 300, "currentTime": 42.5, "paused": False,
                "ended": False, "volume": 80, "muted": False,
                "width": 1920, "height": 1080, "playbackRate": 1.0, "loop": False,
            }],
        })
        result = await chrome_bridge.get_media_info()
        assert result["count"] == 1
        assert result["elements"][0]["tag"] == "video"
        assert result["elements"][0]["duration"] == 300
        assert result["elements"][0]["paused"] is False

    @pytest.mark.asyncio
    async def test_get_media_info_no_media(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"count": 0, "elements": []})
        result = await chrome_bridge.get_media_info()
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_get_media_info_bad_return(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value="not a dict")
        result = await chrome_bridge.get_media_info()
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_media_play(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"ok": True, "action": "play"})
        result = await chrome_bridge.media_play()
        assert result["ok"] is True
        assert result["action"] == "play"

    @pytest.mark.asyncio
    async def test_media_play_no_media(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"ok": False, "error": "no media"})
        result = await chrome_bridge.media_play()
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_media_pause(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"ok": True, "action": "pause"})
        result = await chrome_bridge.media_pause()
        assert result["ok"] is True
        assert result["action"] == "pause"

    @pytest.mark.asyncio
    async def test_media_seek(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"ok": True, "action": "seek", "time": 120})
        result = await chrome_bridge.media_seek(120)
        assert result["ok"] is True
        assert result["time"] == 120

    @pytest.mark.asyncio
    async def test_media_set_volume(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"ok": True, "volume": 75})
        result = await chrome_bridge.media_set_volume(75)
        assert result["ok"] is True
        # Check that the JS uses the clamped volume (75/100)
        js_code = mock_cdp.evaluate_js.call_args[0][0]
        assert "0.75" in js_code

    @pytest.mark.asyncio
    async def test_media_set_volume_clamped(self, chrome_bridge, mock_cdp):
        """Volume should be clamped to 0-100."""
        mock_cdp.evaluate_js = AsyncMock(return_value={"ok": True, "volume": 100})
        await chrome_bridge.media_set_volume(150)
        js_code = mock_cdp.evaluate_js.call_args[0][0]
        assert "1.0" in js_code  # max(0, min(100, 150)) / 100 = 1.0

    @pytest.mark.asyncio
    async def test_media_set_speed(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"ok": True, "speed": 2.0})
        result = await chrome_bridge.media_set_speed(2.0)
        assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_media_set_speed_clamped(self, chrome_bridge, mock_cdp):
        """Speed should be clamped to 0.25-4.0."""
        mock_cdp.evaluate_js = AsyncMock(return_value={"ok": True, "speed": 4.0})
        await chrome_bridge.media_set_speed(10.0)
        js_code = mock_cdp.evaluate_js.call_args[0][0]
        assert "4.0" in js_code

    @pytest.mark.asyncio
    async def test_media_fullscreen(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"ok": True, "action": "fullscreen"})
        result = await chrome_bridge.media_fullscreen()
        assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_media_toggle_mute(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"ok": True, "muted": True})
        result = await chrome_bridge.media_toggle_mute()
        assert result["ok"] is True
        assert result["muted"] is True


# ═══════════════════════════════════════════════════════════════════
# Captions Tests
# ═══════════════════════════════════════════════════════════════════


class TestCaptions:
    @pytest.mark.asyncio
    async def test_get_youtube_captions(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={
            "source": "youtube_overlay",
            "lines": ["Hello world", "This is a caption"],
        })
        result = await chrome_bridge.get_captions()
        assert result["source"] == "youtube_overlay"
        assert len(result["lines"]) == 2

    @pytest.mark.asyncio
    async def test_get_text_track_captions(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={
            "source": "text_tracks",
            "tracks": [{
                "label": "English", "lang": "en", "kind": "subtitles", "mode": "showing",
                "cues": [
                    {"start": 0.0, "end": 2.5, "text": "Hello"},
                    {"start": 2.5, "end": 5.0, "text": "World"},
                ],
            }],
        })
        result = await chrome_bridge.get_captions()
        assert result["source"] == "text_tracks"
        assert len(result["tracks"][0]["cues"]) == 2

    @pytest.mark.asyncio
    async def test_no_captions_found(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={
            "source": "none",
            "message": "No captions found. Try enabling CC on the video.",
        })
        result = await chrome_bridge.get_captions()
        assert result["source"] == "none"

    @pytest.mark.asyncio
    async def test_get_captions_bad_return(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value="unexpected string")
        result = await chrome_bridge.get_captions()
        assert result["source"] == "none"

    @pytest.mark.asyncio
    async def test_enable_captions(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"ok": True, "action": "enabled"})
        result = await chrome_bridge.enable_captions()
        assert result["ok"] is True
        assert result["action"] == "enabled"

    @pytest.mark.asyncio
    async def test_enable_captions_already_on(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"ok": True, "action": "already_on"})
        result = await chrome_bridge.enable_captions()
        assert result["ok"] is True
        assert result["action"] == "already_on"

    @pytest.mark.asyncio
    async def test_enable_captions_no_button(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value={"ok": False, "error": "CC button not found"})
        result = await chrome_bridge.enable_captions()
        assert result["ok"] is False


# ═══════════════════════════════════════════════════════════════════
# get_url Tests
# ═══════════════════════════════════════════════════════════════════


class TestGetUrl:
    @pytest.mark.asyncio
    async def test_returns_url(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value="https://example.com/page")
        url = await chrome_bridge.get_url()
        assert url == "https://example.com/page"

    @pytest.mark.asyncio
    async def test_empty_when_disconnected(self, bridge):
        url = await bridge.get_url()
        assert url == ""

    @pytest.mark.asyncio
    async def test_empty_when_none_returned(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(return_value=None)
        url = await chrome_bridge.get_url()
        assert url == ""


# ═══════════════════════════════════════════════════════════════════
# Error Handling / Edge Cases
# ═══════════════════════════════════════════════════════════════════


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_eval_wraps_exceptions(self, chrome_bridge, mock_cdp):
        """_eval catches exceptions and returns error dicts."""
        mock_cdp.evaluate_js = AsyncMock(side_effect=TimeoutError("timed out"))
        result = await chrome_bridge._eval("anything")
        assert "error" in result
        assert "timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_eval_wraps_connection_error(self, chrome_bridge, mock_cdp):
        mock_cdp.evaluate_js = AsyncMock(side_effect=ConnectionError("pipe broken"))
        result = await chrome_bridge._eval("anything")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_all_actions_fail_when_disconnected(self, bridge):
        """Every public action should handle disconnected state gracefully."""
        assert (await bridge.read_text()) == ""
        assert (await bridge.get_url()) == ""
        assert (await bridge.evaluate_js("1"))["error"] == "Not connected"
        assert (await bridge.click(selector="#x"))["ok"] is False
        assert (await bridge.type_text(selector="#x", value="v"))["ok"] is False
        assert (await bridge.scroll())["ok"] is False
        assert (await bridge.navigate("https://x.com"))["ok"] is False
        assert (await bridge.get_page_tree())["elements"] == []

    @pytest.mark.asyncio
    async def test_type_text_unexpected_return(self, chrome_bridge, mock_cdp):
        """type_text should handle unexpected return types."""
        mock_cdp.evaluate_js = AsyncMock(return_value=42)
        result = await chrome_bridge.type_text(selector="#field", value="val")
        assert result["ok"] is False
        assert "Unexpected" in result["error"]

    @pytest.mark.asyncio
    async def test_click_sel_unexpected_return(self, chrome_bridge, mock_cdp):
        """_click_sel should handle non-dict returns."""
        mock_cdp.evaluate_js = AsyncMock(return_value=True)
        result = await chrome_bridge.click(selector="#btn")
        assert result["ok"] is False
        assert "Unexpected" in result["error"]


# ═══════════════════════════════════════════════════════════════════
# Chrome Debug Helpers
# ═══════════════════════════════════════════════════════════════════


class TestChromeDebugHelpers:
    def test_chrome_debug_command(self):
        cmd = BrowserBridge.chrome_debug_command(port=9222)
        assert "remote-debugging-port=9222" in cmd
        assert "Google Chrome" in cmd

    def test_chrome_debug_command_custom_port(self):
        cmd = BrowserBridge.chrome_debug_command(port=9333)
        assert "remote-debugging-port=9333" in cmd

    @pytest.mark.asyncio
    async def test_launch_chrome_success(self):
        with patch("subprocess.Popen") as mock_popen, \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await BrowserBridge.launch_chrome_debug(port=9222)
        assert result is True
        mock_popen.assert_called_once()

    @pytest.mark.asyncio
    async def test_launch_chrome_not_installed(self):
        with patch("subprocess.Popen", side_effect=FileNotFoundError("not found")), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await BrowserBridge.launch_chrome_debug()
        assert result is False


# ═══════════════════════════════════════════════════════════════════
# _ChromeCDP Unit Tests
# ═══════════════════════════════════════════════════════════════════


class TestChromeCDPBackend:
    def test_initially_disconnected(self):
        cdp = _ChromeCDP()
        assert cdp.connected is False
        assert cdp._ws is None

    @pytest.mark.asyncio
    async def test_send_when_disconnected(self):
        cdp = _ChromeCDP()
        result = await cdp.send("Page.navigate", {"url": "https://x.com"})
        assert "error" in result
        assert "not connected" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_evaluate_js_when_disconnected(self):
        cdp = _ChromeCDP()
        result = await cdp.evaluate_js("1+1")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        cdp = _ChromeCDP()
        await cdp.close()  # should not raise even when nothing is open


# ═══════════════════════════════════════════════════════════════════
# _SafariJSBridge Unit Tests
# ═══════════════════════════════════════════════════════════════════


class TestSafariJSBridgeBackend:
    def test_initially_disconnected(self):
        safari = _SafariJSBridge()
        assert safari.connected is False

    @pytest.mark.asyncio
    async def test_close_resets_state(self):
        safari = _SafariJSBridge()
        safari._ok = True
        assert safari.connected is True
        await safari.close()
        assert safari.connected is False

    @pytest.mark.asyncio
    async def test_connect_success(self):
        safari = _SafariJSBridge()
        with patch.object(safari, "_applescript", new_callable=AsyncMock, return_value="My Page"):
            result = await safari.connect()
        assert result is True
        assert safari.connected is True

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        safari = _SafariJSBridge()
        with patch.object(safari, "_applescript", new_callable=AsyncMock, return_value=None):
            result = await safari.connect()
        assert result is False
        assert safari.connected is False

    @pytest.mark.asyncio
    async def test_evaluate_js_parses_json(self):
        safari = _SafariJSBridge()
        safari._ok = True
        with patch.object(safari, "_applescript", new_callable=AsyncMock, return_value='{"key": "val"}'):
            result = await safari.evaluate_js("someCode()")
        assert result == {"key": "val"}

    @pytest.mark.asyncio
    async def test_evaluate_js_returns_raw_on_non_json(self):
        safari = _SafariJSBridge()
        safari._ok = True
        with patch.object(safari, "_applescript", new_callable=AsyncMock, return_value="plain text"):
            result = await safari.evaluate_js("someCode()")
        assert result == "plain text"

    @pytest.mark.asyncio
    async def test_evaluate_js_applescript_failure(self):
        safari = _SafariJSBridge()
        safari._ok = True
        with patch.object(safari, "_applescript", new_callable=AsyncMock, return_value=None):
            result = await safari.evaluate_js("failing()")
        assert result == {"error": "AppleScript execution failed"}

    def test_osascript_success(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="output\n")
            result = _SafariJSBridge._osascript("tell app to do thing")
        assert result == "output"

    def test_osascript_failure(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            result = _SafariJSBridge._osascript("bad script")
        assert result is None

    def test_osascript_timeout(self):
        import subprocess as sp
        with patch("subprocess.run", side_effect=sp.TimeoutExpired(cmd="osascript", timeout=10)):
            result = _SafariJSBridge._osascript("slow script")
        assert result is None


# ═══════════════════════════════════════════════════════════════════
# Frontmost App Detection
# ═══════════════════════════════════════════════════════════════════


class TestFrontmostApp:
    @pytest.mark.asyncio
    async def test_returns_app_name(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Safari\n")
            result = await _frontmost_app()
        assert result == "Safari"

    @pytest.mark.asyncio
    async def test_returns_empty_on_failure(self):
        with patch("subprocess.run", side_effect=OSError("no osascript")):
            result = await _frontmost_app()
        assert result == ""
