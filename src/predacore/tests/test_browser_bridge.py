"""
Real Chrome integration tests for BrowserBridge v2.

NO MOCKS. Launches a real Chrome on port 9223 (sandbox port, won't
clobber your running Chrome), serves test HTML from a local HTTP server,
and exercises every method against the real browser.

Auto-skips if Chrome is not installed.

Usage:
    cd project_prometheus/src
    python -m pytest predacore/tests/test_browser_bridge.py -v --timeout=60
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

# ── Skip if Chrome not installed or aiohttp missing ──────────────────

CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
]
CHROME_BIN = next((p for p in CHROME_PATHS if p and Path(p).exists()), None)

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytestmark = [
    pytest.mark.skipif(CHROME_BIN is None, reason="Chrome not installed"),
    pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed"),
]

CDP_PORT = 9223

# Set PREDACORE_BROWSER_HEADED=1 to watch tests in a real Chrome window
HEADED = os.environ.get("PREDACORE_BROWSER_HEADED", "").lower() in ("1", "true", "yes")

# ── Test HTML pages ──────────────────────────────────────────────────

IFRAME_HTML = """<!DOCTYPE html>
<html><head><title>Iframe Content</title></head>
<body><p id="iframe-text">Hello from inside the iframe</p></body>
</html>"""

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>PredaCore Test Page</title></head>
<body>
  <h1 id="heading">Hello PredaCore</h1>
  <p id="para">This is a test paragraph for browser bridge testing.</p>

  <a href="/page2" id="link1">Go to Page 2</a>
  <button id="btn1" onclick="document.getElementById('result').textContent='clicked'">Click Me</button>
  <div id="result"></div>

  <form id="testform">
    <input type="text" id="name-input" placeholder="Your name" />
    <input type="email" id="email-input" placeholder="Email" />
    <input type="checkbox" id="agree-checkbox" />
    <select id="color-select">
      <option value="">Choose...</option>
      <option value="red">Red</option>
      <option value="blue">Blue</option>
      <option value="green">Green</option>
    </select>
    <input type="file" id="file-input" />
  </form>

  <table id="data-table">
    <tr><th>Name</th><th>Age</th></tr>
    <tr><td>Alice</td><td>30</td></tr>
    <tr><td>Bob</td><td>25</td></tr>
  </table>

  <div id="drag-source" draggable="true" style="width:80px;height:40px;background:#369;cursor:grab;">Drag me</div>
  <div id="drop-target" style="width:120px;height:60px;background:#933;margin-top:10px;"
       ondrop="this.textContent='dropped!';event.preventDefault()"
       ondragover="event.preventDefault()">Drop here</div>

  <img id="test-img" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" alt="1px red dot" width="50" height="50" />

  <canvas id="test-canvas" width="100" height="60"></canvas>
  <script>
    var ctx = document.getElementById('test-canvas').getContext('2d');
    ctx.fillStyle = '#0066ff';
    ctx.fillRect(10, 10, 80, 40);
  </script>

  <svg id="test-svg" width="80" height="40" viewBox="0 0 80 40">
    <rect fill="#00cc66" width="80" height="40"/>
  </svg>

  <div id="bg-div" style="width:60px;height:30px;background-image:url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==')"></div>

  <iframe id="test-iframe" src="/iframe" width="200" height="100"></iframe>

  <button id="alert-btn" onclick="alert('Hello from alert!')">Trigger Alert</button>
  <button id="confirm-btn" onclick="document.getElementById('confirm-result').textContent=confirm('Are you sure?')?'yes':'no'">Trigger Confirm</button>
  <div id="confirm-result"></div>
  <button id="prompt-btn" onclick="document.getElementById('prompt-result').textContent=prompt('Enter name:','PredaCore')">Trigger Prompt</button>
  <div id="prompt-result"></div>

  <div id="dynamic-target" style="display:none;">I appeared!</div>
  <script>
    setTimeout(function(){ document.getElementById('dynamic-target').style.display='block'; }, 1500);
  </script>
</body>
</html>"""

PAGE2_HTML = """<!DOCTYPE html>
<html><head><title>Page 2</title></head>
<body><h1>Page 2</h1><a href="/">Back Home</a></body>
</html>"""


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def test_server():
    """Start a stdlib HTTP server (sync, no asyncio conflict)."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/page2":
                body = PAGE2_HTML.encode()
            elif self.path == "/iframe":
                body = IFRAME_HTML.encode()
            else:
                body = INDEX_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass  # suppress request logs

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture(scope="module")
def chrome_process():
    """Launch Chrome with remote debugging on the sandbox port."""
    assert CHROME_BIN is not None
    user_data = tempfile.mkdtemp(prefix="predacore_chrome_test_")
    cmd = [
        CHROME_BIN,
        f"--remote-debugging-port={CDP_PORT}",
        f"--user-data-dir={user_data}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-popup-blocking",
        "--disable-gpu",
    ]
    if not HEADED:
        cmd.append("--headless")
    cmd.append("about:blank")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    # Chrome cold start can take 10-20s on macOS (profile init, keychain)
    for _ in range(60):
        try:
            import urllib.request
            resp = urllib.request.urlopen(f"http://localhost:{CDP_PORT}/json", timeout=2)
            data = json.loads(resp.read())
            if any(t.get("type") == "page" for t in data):
                break
        except (OSError, Exception):
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.skip("Chrome failed to start on debug port")

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    # Clean up temp profile
    import shutil as _shutil
    _shutil.rmtree(user_data, ignore_errors=True)


@pytest_asyncio.fixture
async def bridge(chrome_process, test_server):
    """Create a connected BrowserBridge for each test."""
    from predacore.operators.browser_bridge import BrowserBridge

    b = BrowserBridge(cdp_port=CDP_PORT)
    connected = await b.connect(browser="chrome")
    assert connected, "BrowserBridge failed to connect to Chrome"
    # Navigate to test page
    await b.navigate(test_server)
    yield b
    await b.disconnect()


# ═══════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════


class TestConnection:
    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self, chrome_process):
        from predacore.operators.browser_bridge import BrowserBridge
        b = BrowserBridge(cdp_port=CDP_PORT)
        assert await b.connect(browser="chrome")
        assert b.connected
        assert b.is_cdp
        await b.disconnect()
        assert not b.connected


class TestNavigation:
    @pytest.mark.asyncio
    async def test_navigate_to_page(self, bridge, test_server):
        result = await bridge.navigate(test_server)
        assert result["ok"]
        assert "PredaCore Test Page" in result.get("title", "")

    @pytest.mark.asyncio
    async def test_get_url(self, bridge, test_server):
        url = await bridge.get_url()
        assert test_server.rstrip("/") in url

    @pytest.mark.asyncio
    async def test_back_forward(self, bridge, test_server):
        await bridge.navigate(f"{test_server}/page2")
        url = await bridge.get_url()
        assert "page2" in url
        result = await bridge.back()
        assert result["ok"]
        await asyncio.sleep(0.5)
        url = await bridge.get_url()
        assert "page2" not in url
        result = await bridge.forward()
        assert result["ok"]

    @pytest.mark.asyncio
    async def test_reload(self, bridge):
        result = await bridge.reload()
        assert result["ok"]


class TestPageTree:
    @pytest.mark.asyncio
    async def test_get_page_tree(self, bridge):
        tree = await bridge.get_page_tree()
        assert "elements" in tree
        assert tree["element_count"] > 0
        assert tree.get("url", "").startswith("http")

    @pytest.mark.asyncio
    async def test_elements_have_selectors(self, bridge):
        tree = await bridge.get_page_tree()
        for el in tree["elements"][:5]:
            assert "selector" in el
            assert "tag" in el


class TestReadText:
    @pytest.mark.asyncio
    async def test_read_full_page(self, bridge):
        text = await bridge.read_text()
        assert "Hello PredaCore" in text

    @pytest.mark.asyncio
    async def test_read_by_selector(self, bridge):
        text = await bridge.read_text(selector="#para")
        assert "test paragraph" in text


class TestClick:
    @pytest.mark.asyncio
    async def test_click_by_text(self, bridge):
        result = await bridge.click(text="Click Me")
        assert result.get("ok")
        # Verify the click handler fired
        await asyncio.sleep(0.2)
        text = await bridge.read_text(selector="#result")
        assert text == "clicked"

    @pytest.mark.asyncio
    async def test_click_by_selector(self, bridge):
        result = await bridge.click(selector="#btn1")
        assert result.get("ok")

    @pytest.mark.asyncio
    async def test_click_not_found(self, bridge):
        result = await bridge.click(text="nonexistent button xyz")
        assert not result.get("ok")


class TestType:
    @pytest.mark.asyncio
    async def test_type_inject(self, bridge):
        """Fast injection (type_text) — sets value directly."""
        result = await bridge.type_text(selector="#name-input", value="PredaCore")
        assert result.get("ok")
        assert result.get("method") == "inject"
        val = await bridge.evaluate_js('document.getElementById("name-input").value')
        assert val == "PredaCore"

    @pytest.mark.asyncio
    async def test_type_keys_real_keystrokes(self, bridge):
        """Real keystrokes (type_keys) — char by char via CDP Input events."""
        result = await bridge.type_keys(selector="#email-input", value="test@example.com", clear=True)
        assert result.get("ok")
        assert result.get("method") == "cdp_keystrokes"
        assert result.get("chars") == len("test@example.com")
        val = await bridge.evaluate_js('document.getElementById("email-input").value')
        assert val == "test@example.com"


class TestScroll:
    @pytest.mark.asyncio
    async def test_scroll_down_and_up(self, bridge):
        result = await bridge.scroll(direction="down", amount=2)
        assert result.get("ok") or isinstance(result.get("scrollY"), (int, float))
        result = await bridge.scroll(direction="up", amount=2)
        assert result.get("ok") or isinstance(result.get("scrollY"), (int, float))


class TestWaitPrimitives:
    @pytest.mark.asyncio
    async def test_wait_for_element_found(self, bridge):
        result = await bridge.wait_for_element("#heading", timeout=3)
        assert result["found"]

    @pytest.mark.asyncio
    async def test_wait_for_element_timeout(self, bridge):
        result = await bridge.wait_for_element("#nonexistent-xyz", timeout=1)
        assert not result["found"]
        assert result.get("timeout")

    @pytest.mark.asyncio
    async def test_wait_for_dynamic_element(self, bridge, test_server):
        await bridge.navigate(test_server)
        result = await bridge.wait_for_element("#dynamic-target", timeout=5)
        assert result["found"]

    @pytest.mark.asyncio
    async def test_wait_for_text(self, bridge):
        result = await bridge.wait_for_text("Hello PredaCore", timeout=3)
        assert result["found"]

    @pytest.mark.asyncio
    async def test_wait_for_text_timeout(self, bridge):
        result = await bridge.wait_for_text("text that does not exist xyz", timeout=1)
        assert not result["found"]


class TestCookies:
    @pytest.mark.asyncio
    async def test_set_and_get_cookie(self, bridge):
        await bridge.set_cookie(name="test_cookie", value="hello123")
        result = await bridge.get_cookies()
        names = [c["name"] for c in result.get("cookies", [])]
        assert "test_cookie" in names

    @pytest.mark.asyncio
    async def test_delete_cookie(self, bridge):
        await bridge.set_cookie(name="to_delete", value="bye")
        await bridge.delete_cookies(name="to_delete")
        result = await bridge.get_cookies()
        names = [c["name"] for c in result.get("cookies", [])]
        assert "to_delete" not in names


class TestStorage:
    @pytest.mark.asyncio
    async def test_set_and_get_local_storage(self, bridge):
        await bridge.set_storage("test_key", "test_value")
        result = await bridge.get_storage("test_key")
        assert result.get("value") == "test_value"

    @pytest.mark.asyncio
    async def test_clear_storage(self, bridge):
        await bridge.set_storage("key1", "val1")
        await bridge.clear_storage()
        result = await bridge.get_storage("key1")
        assert result.get("value") is None


class TestFormControls:
    @pytest.mark.asyncio
    async def test_set_checkbox(self, bridge):
        result = await bridge.set_checkbox("#agree-checkbox", checked=True)
        assert result.get("ok")
        val = await bridge.evaluate_js('document.getElementById("agree-checkbox").checked')
        assert val is True

    @pytest.mark.asyncio
    async def test_select_option_by_value(self, bridge):
        result = await bridge.select_option("#color-select", value="blue")
        assert result.get("ok")
        val = await bridge.evaluate_js('document.getElementById("color-select").value')
        assert val == "blue"

    @pytest.mark.asyncio
    async def test_select_option_by_label(self, bridge):
        result = await bridge.select_option("#color-select", label="Green")
        assert result.get("ok")


class TestKeyboard:
    @pytest.mark.asyncio
    async def test_press_key(self, bridge):
        result = await bridge.press_key("Tab")
        assert result.get("ok")

    @pytest.mark.asyncio
    async def test_key_combo(self, bridge):
        result = await bridge.key_combo(["ctrl", "a"])
        assert result.get("ok")


class TestHover:
    @pytest.mark.asyncio
    async def test_hover_by_selector(self, bridge):
        result = await bridge.hover(selector="#btn1")
        assert result.get("ok")


class TestScreenshotPDF:
    @pytest.mark.asyncio
    async def test_viewport_screenshot(self, bridge):
        result = await bridge.screenshot()
        assert result.get("ok")
        assert result.get("size_bytes", 0) > 0
        path = result.get("path", "")
        assert Path(path).exists()
        Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_full_page_screenshot(self, bridge):
        result = await bridge.screenshot(full_page=True)
        assert result.get("ok")
        path = result.get("path", "")
        assert Path(path).exists()
        Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_print_pdf(self, bridge):
        result = await bridge.print_pdf()
        assert result.get("ok")
        assert result.get("size_bytes", 0) > 0
        path = result.get("path", "")
        assert Path(path).exists()
        Path(path).unlink(missing_ok=True)


class TestTabs:
    @pytest.mark.asyncio
    async def test_list_tabs(self, bridge):
        result = await bridge.list_tabs()
        assert result.get("count", 0) >= 1

    @pytest.mark.asyncio
    async def test_new_and_close_tab(self, bridge):
        result = await bridge.new_tab("about:blank")
        assert result.get("ok")
        target_id = result.get("target_id", "")
        assert target_id
        close_result = await bridge.close_tab(target_id)
        assert close_result.get("ok")


class TestTableExtraction:
    @pytest.mark.asyncio
    async def test_extract_table(self, bridge):
        result = await bridge.extract_tables("#data-table")
        assert result.get("count", 0) == 1
        table = result["tables"][0]
        assert table["rows"] == 3  # header + 2 data rows
        assert "Alice" in table["data"][1]


class TestFindInPage:
    @pytest.mark.asyncio
    async def test_find_existing_text(self, bridge):
        result = await bridge.find_in_page("PredaCore")
        assert result.get("count", 0) > 0
        assert any("PredaCore" in m.get("text", "") for m in result.get("matches", []))

    @pytest.mark.asyncio
    async def test_find_nonexistent_text(self, bridge):
        result = await bridge.find_in_page("zzz_nonexistent_zzz")
        assert result.get("count", 0) == 0


class TestPageContent:
    @pytest.mark.asyncio
    async def test_get_page_links(self, bridge):
        result = await bridge.get_page_links()
        assert result.get("count", 0) > 0

    @pytest.mark.asyncio
    async def test_evaluate_js(self, bridge):
        result = await bridge.evaluate_js("1 + 1")
        assert result == 2

    @pytest.mark.asyncio
    async def test_evaluate_js_string(self, bridge):
        result = await bridge.evaluate_js("document.title")
        assert "PredaCore" in str(result)


# ═══════════════════════════════════════════════════════════════════════
# New v2 features — Tier A + B
# ═══════════════════════════════════════════════════════════════════════


class TestDragAndDrop:
    @pytest.mark.asyncio
    async def test_drag_and_drop(self, bridge):
        result = await bridge.drag_and_drop("#drag-source", "#drop-target")
        assert result.get("ok")
        # Check that the drop handler fired
        await asyncio.sleep(0.3)
        text = await bridge.read_text(selector="#drop-target")
        assert "dropped" in text.lower()


class TestIframes:
    @pytest.mark.asyncio
    async def test_list_frames(self, bridge):
        result = await bridge.list_frames()
        assert result.get("count", 0) >= 2  # main + iframe

    @pytest.mark.asyncio
    async def test_evaluate_in_frame(self, bridge):
        frames = await bridge.list_frames()
        iframe = next((f for f in frames.get("frames", []) if "iframe" in f.get("url", "")), None)
        if iframe is None:
            pytest.skip("iframe not found in frame tree")
        result = await bridge.evaluate_in_frame(iframe["id"], "document.title")
        assert "Iframe" in str(result)

    @pytest.mark.asyncio
    async def test_click_in_frame(self, bridge):
        frames = await bridge.list_frames()
        iframe = next((f for f in frames.get("frames", []) if "iframe" in f.get("url", "")), None)
        if iframe is None:
            pytest.skip("iframe not found in frame tree")
        result = await bridge.click_in_frame(iframe["id"], "#iframe-text")
        assert result.get("ok")


class TestClipboard:
    @pytest.mark.asyncio
    async def test_clipboard_write(self, bridge):
        result = await bridge.clipboard_write("PredaCore clipboard test 42")
        # Clipboard may be restricted in headless — just verify no crash
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_clipboard_read(self, bridge):
        result = await bridge.clipboard_read()
        # Clipboard read may fail in headless (no permission) — just verify no crash
        assert isinstance(result, dict)


class TestNetworkLog:
    @pytest.mark.asyncio
    async def test_start_and_get_network_log(self, bridge, test_server):
        start = await bridge.start_network_log()
        assert start.get("ok")
        # Trigger a network request by navigating
        await bridge.navigate(f"{test_server}/page2")
        await asyncio.sleep(0.5)
        log = await bridge.get_network_log()
        assert isinstance(log.get("requests"), list)

    @pytest.mark.asyncio
    async def test_clear_network_log(self, bridge):
        await bridge.start_network_log()
        await bridge.clear_network_log()
        log = await bridge.get_network_log()
        assert log.get("count", 0) == 0


class TestGeolocation:
    @pytest.mark.asyncio
    async def test_set_and_clear_geolocation(self, bridge):
        # Set to San Francisco
        result = await bridge.set_geolocation(latitude=37.7749, longitude=-122.4194)
        assert result.get("ok")
        assert result.get("latitude") == 37.7749
        # Clear
        result = await bridge.clear_geolocation()
        assert result.get("ok")


class TestAuthCredentials:
    @pytest.mark.asyncio
    async def test_set_and_clear_auth(self, bridge):
        result = await bridge.set_auth_credentials(username="testuser", password="testpass")
        assert result.get("ok")
        result = await bridge.clear_auth_credentials()
        assert result.get("ok")


class TestImageOperations:
    @pytest.mark.asyncio
    async def test_element_screenshot(self, bridge):
        result = await bridge.element_screenshot("#heading")
        assert result.get("ok")
        assert result.get("size_bytes", 0) > 0
        Path(result["path"]).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_download_image(self, bridge):
        # Download the inline data: image by selector
        result = await bridge.download_image(selector="#test-img")
        assert result.get("ok")
        assert result.get("size_bytes", 0) > 0
        Path(result["path"]).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_capture_canvas(self, bridge):
        result = await bridge.capture_canvas("#test-canvas")
        assert result.get("ok")
        assert result.get("size_bytes", 0) > 100  # non-trivial PNG
        Path(result["path"]).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_image_to_base64(self, bridge):
        result = await bridge.image_to_base64("#test-img")
        assert result.get("ok")
        assert result.get("width", 0) > 0
        assert "data:image" in result.get("data", "")

    @pytest.mark.asyncio
    async def test_get_background_images(self, bridge):
        result = await bridge.get_background_images()
        assert result.get("count", 0) >= 1
        assert any("data:image" in img.get("url", "") for img in result.get("images", []))

    @pytest.mark.asyncio
    async def test_get_svgs(self, bridge):
        result = await bridge.get_svgs()
        assert result.get("count", 0) >= 1
        assert any("rect" in svg.get("source", "") for svg in result.get("svgs", []))


class TestJSDialogs:
    @pytest.mark.asyncio
    async def test_alert_auto_accepted(self, bridge):
        """alert() should be auto-accepted — PredaCore should NOT hang."""
        # Click the alert button — if dialog handling works, this returns immediately
        await bridge.click(selector="#alert-btn")
        await asyncio.sleep(0.5)
        # If we got here, the dialog was auto-handled (otherwise we'd be stuck)
        last = await bridge.get_last_dialog()
        assert last.get("has_dialog")
        assert last.get("type") == "alert"
        assert "Hello from alert" in last.get("message", "")

    @pytest.mark.asyncio
    async def test_confirm_auto_accepted(self, bridge):
        """confirm() should be auto-accepted (returns true)."""
        await bridge.click(selector="#confirm-btn")
        await asyncio.sleep(0.5)
        result = await bridge.read_text(selector="#confirm-result")
        assert result == "yes"  # confirm() returned true because we auto-accept

    @pytest.mark.asyncio
    async def test_prompt_auto_accepted(self, bridge):
        """prompt() should be auto-accepted with default value."""
        await bridge.click(selector="#prompt-btn")
        await asyncio.sleep(0.5)
        result = await bridge.read_text(selector="#prompt-result")
        assert result == "PredaCore"  # prompt() returned the default value

    @pytest.mark.asyncio
    async def test_set_dialog_handler_dismiss(self, bridge, test_server):
        """Switch to dismiss mode — confirm() should return false."""
        await bridge.set_dialog_handler(mode="dismiss")
        await bridge.navigate(test_server)  # fresh page
        await bridge.click(selector="#confirm-btn")
        await asyncio.sleep(0.5)
        result = await bridge.read_text(selector="#confirm-result")
        assert result == "no"  # confirm() returned false because we dismiss
        # Reset to accept for other tests
        await bridge.set_dialog_handler(mode="accept")


class TestDownloadPath:
    @pytest.mark.asyncio
    async def test_set_download_path(self, bridge):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await bridge.set_download_path(path=tmpdir)
            assert result.get("ok")
            assert result.get("download_path") == tmpdir
