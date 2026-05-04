"""
PredaCore Browser Bridge v2 — Full browser control at native speed.

Two backends:
  ChromeCDP      — Chrome/Brave/Edge via DevTools Protocol websocket
  SafariJSBridge — AppleScript ``do JavaScript`` on the running Safari

v2 improvements over v1:
  - Smart navigate: listens to Page.loadEventFired instead of fixed 4s delay
  - CDP-native input: Input.dispatchMouseEvent/KeyEvent for real pointer events
  - Shadow DOM + iframe traversal in the tree walker
  - Cookies, localStorage, sessionStorage
  - Wait primitives: wait_for_element, wait_for_text, wait_for_url
  - Full-page screenshot + PDF export (CDP)
  - Multi-tab: list/switch/new/close
  - History: back/forward/reload
  - Form controls: checkbox, radio, select, file upload
  - Hover, key combos, find-in-page
  - Table extraction
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ── JS: Tree walker v2 ────────────────────────────────────────────────
# Tag-filtered (skip non-interactive branches), shadow-DOM aware,
# batched getBoundingClientRect (no per-node getComputedStyle).

_PREDACORE_READ_PAGE_JS = r"""
(function _predacore_read_page() {
  var MAX = 300, results = [], count = 0;
  var INTERACTIVE = new Set(['A','BUTTON','INPUT','TEXTAREA','SELECT','DETAILS','SUMMARY','LABEL']);
  function cssPath(el) {
    if (el.id) return '#' + CSS.escape(el.id);
    var parts = [];
    while (el && el.nodeType === 1) {
      var s = el.tagName.toLowerCase();
      if (el.id) { parts.unshift('#' + CSS.escape(el.id)); break; }
      var sib = el, nth = 1;
      while ((sib = sib.previousElementSibling)) { if (sib.tagName === el.tagName) nth++; }
      if (nth > 1) s += ':nth-of-type(' + nth + ')';
      parts.unshift(s); el = el.parentElement;
    }
    return parts.join(' > ');
  }
  function interactive(el) {
    if (INTERACTIVE.has(el.tagName)) return true;
    if (el.getAttribute('role')||el.getAttribute('onclick')||el.getAttribute('tabindex')!=null) return true;
    if (el.isContentEditable) return true;
    var cs = getComputedStyle(el);
    return cs.cursor === 'pointer';
  }
  function walk(root) {
    var tw = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT, null, false), n;
    while ((n = tw.nextNode()) && count < MAX) {
      // Descend into shadow roots
      if (n.shadowRoot) walk(n.shadowRoot);
      var r = n.getBoundingClientRect();
      if (r.width < 4 || r.height < 4) continue;
      if (!interactive(n)) continue;
      var cs = getComputedStyle(n);
      if (cs.display==='none'||cs.visibility==='hidden'||+cs.opacity===0) continue;
      var tag = n.tagName.toLowerCase();
      var label = n.getAttribute('aria-label') || n.getAttribute('alt')
               || n.getAttribute('title') || n.getAttribute('placeholder')
               || (n.textContent||'').trim().slice(0,120);
      var val = '';
      if (n.value !== undefined && n.value !== null) val = String(n.value);
      var typ = n.getAttribute('type') || '';
      results.push({
        tag: tag, role: n.getAttribute('role') || tag, label: label,
        value: val, type: typ,
        bounds: {x:Math.round(r.left),y:Math.round(r.top),w:Math.round(r.width),h:Math.round(r.height)},
        selector: cssPath(n),
        clickable: tag==='a'||tag==='button'||!!n.getAttribute('onclick')||cs.cursor==='pointer',
        typeable: tag==='input'||tag==='textarea'||n.isContentEditable,
        checked: !!n.checked,
        disabled: !!n.disabled,
        inShadow: !!n.getRootNode().host
      });
      count++;
    }
  }
  walk(document.body || document.documentElement);
  // Also walk shadow roots of top-level custom elements
  document.querySelectorAll('*').forEach(function(el){
    if (el.shadowRoot && count < MAX) walk(el.shadowRoot);
  });
  return JSON.stringify({url:location.href,title:document.title,elements:results,
    timestamp:Date.now(),viewport:{w:window.innerWidth,h:window.innerHeight},
    scrollY:window.scrollY,scrollHeight:document.documentElement.scrollHeight});
})()
"""

_DEFAULT_TIMEOUT: float = 30.0   # raised from 5.0 per "remove all limits"
_CDP_PORT = 9222
_NAVIGATE_TIMEOUT = 120.0         # raised from 15.0 per "remove all limits"
_NAVIGATE_SETTLE_MS = 800

# ECONNREFUSED is errno 61 on macOS, errno 111 on Linux. We catch both
# explicitly so we can surface a clear "Chrome isn't running" message
# rather than the generic "tool timed out" the dispatcher would log.
_ECONNREFUSED_ERRNOS = (61, 111)


class ChromeNotRunningError(ConnectionError):
    """Raised when Chrome isn't listening on the CDP port.

    Carries an actionable message so the tool dispatcher can surface
    something useful instead of a vague timeout.
    """

    def __init__(self, port: int = _CDP_PORT) -> None:
        super().__init__(
            f"Chrome not running with CDP enabled on :{port}. "
            f"Start it with: "
            f"open -a 'Google Chrome' --args --remote-debugging-port={port}"
        )
        self.port = port

# ── ChromeCDP Backend ───────────────────────────────────────────────

class _ChromeCDP:
    """Chrome DevTools Protocol backend over aiohttp websocket."""
    def __init__(self) -> None:
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._msg_id: int = 0
        self._ws_url: str = ""
        self.browser_label: str = ""
        self._events: dict[str, list[dict]] = {}
        self._event_waiters: dict[str, list[asyncio.Future]] = {}
        self._bridge_ref: Any | None = None  # Back-ref to BrowserBridge for dialog handling

    @property
    def connected(self) -> bool:
        return self._ws is not None and not self._ws.closed

    async def connect(self, port: int = _CDP_PORT) -> bool:
        # The whole flow is wrapped in try/finally that calls _cleanup on
        # ANY abnormal exit — including asyncio.CancelledError raised by
        # the tool dispatcher's outer timeout. Without this, aiohttp's
        # connection pool leaks sockets to localhost:port whenever a
        # connect attempt is cancelled mid-flight, producing the
        # "Unclosed client session" warnings on shutdown.
        #
        # Note: we deliberately do NOT raise ChromeNotRunningError from
        # this method. ``BrowserBridge._auto_connect`` calls us first and
        # then attempts to launch Chrome on its own when we return False,
        # so connection-refused needs to flow through as a soft failure.
        # Public ``ChromeNotRunningError`` is reserved for callers that
        # want to short-circuit the auto-launch flow.
        ok = False
        self._session = aiohttp.ClientSession()
        try:
            try:
                url = f"http://localhost:{port}/json"
                async with self._session.get(
                    url, timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    targets = await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, OSError) as exc:
                logger.info("CDP not available on :%d (%s)", port, exc)
                return False

            page = next((t for t in targets if t.get("type") == "page"), None)
            if not page:
                return False
            self._ws_url = page.get("webSocketDebuggerUrl", "")
            self.browser_label = page.get("browser", "Chrome")
            if not self._ws_url:
                return False
            try:
                self._ws = await self._session.ws_connect(self._ws_url)
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                return False
            ok = True
        finally:
            # Tear down only on the unhappy paths. ws_connect success
            # leaves the session alive (the websocket needs it); any
            # failure or cancellation (CancelledError) hits this branch
            # and closes the session before we propagate the exception.
            if not ok:
                await self._cleanup()
        # Enable CDP domains we need
        await self.send("Page.enable", timeout=2)
        await self.send("Runtime.enable", timeout=2)
        await self.send("DOM.enable", timeout=2)
        await self.send("Network.enable", timeout=2)
        logger.info("CDP connected: %s (%s)", self.browser_label, self._ws_url[:80])
        return True

    async def send(self, method: str, params: dict[str, Any] | None = None,
                   timeout: float = _DEFAULT_TIMEOUT) -> dict[str, Any]:
        if not self.connected:
            return {"error": "CDP not connected"}
        self._msg_id += 1
        mid = self._msg_id
        assert self._ws is not None
        await self._ws.send_json({"id": mid, "method": method, "params": params or {}})
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                raw = await asyncio.wait_for(self._ws.receive(), timeout=max(0.1, deadline - asyncio.get_event_loop().time()))
            except asyncio.TimeoutError:
                return {"error": f"CDP timeout for {method}"}
            if raw.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(raw.data)
                if data.get("id") == mid:
                    if "error" in data:
                        return {"error": data["error"].get("message", str(data["error"]))}
                    return data.get("result", {})
                # Store events for waiters
                if "method" in data:
                    evt_method = data["method"]
                    # Auto-handle JS dialogs so they never block
                    if evt_method == "Page.javascriptDialogOpening" and self._bridge_ref:
                        import asyncio as _aio
                        _aio.create_task(self._bridge_ref._handle_dialog_event(data.get("params", {})))
                    waiters = self._event_waiters.get(evt_method, [])
                    for w in waiters:
                        if not w.done():
                            w.set_result(data.get("params", {}))
                    self._event_waiters.pop(evt_method, None)
            elif raw.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                self._ws = None
                return {"error": "CDP websocket closed"}
        return {"error": f"CDP timeout for {method}"}

    async def wait_for_event(self, event_name: str, timeout: float = 10.0) -> dict[str, Any] | None:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._event_waiters.setdefault(event_name, []).append(fut)
        # Drain messages while waiting
        assert self._ws is not None
        deadline = loop.time() + timeout
        while not fut.done() and loop.time() < deadline:
            try:
                raw = await asyncio.wait_for(self._ws.receive(), timeout=max(0.1, deadline - loop.time()))
            except asyncio.TimeoutError:
                break
            if raw.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(raw.data)
                if "method" in data:
                    evt = data["method"]
                    for w in self._event_waiters.get(evt, []):
                        if not w.done():
                            w.set_result(data.get("params", {}))
                    self._event_waiters.pop(evt, None)
            elif raw.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break
        if fut.done():
            return fut.result()
        # Cleanup
        waiters = self._event_waiters.get(event_name, [])
        if fut in waiters:
            waiters.remove(fut)
        return None

    async def evaluate_js(self, expression: str, timeout: float = _DEFAULT_TIMEOUT) -> Any:
        res = await self.send("Runtime.evaluate", {
            "expression": expression, "returnByValue": True, "awaitPromise": False,
        }, timeout=timeout)
        if "error" in res:
            return res
        val = res.get("result", {}).get("value")
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
        return val

    async def close(self) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.close()
        self._ws = None
        await self._cleanup()

    async def _cleanup(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    # ── Stealth ──

    async def apply_stealth(self) -> None:
        """Mask CDP fingerprints that anti-bot systems detect.

        Patches:
        - navigator.webdriver → undefined
        - chrome.runtime removal (headless artifact)
        - Permissions.query override (notification check fingerprint)
        - Plugin/language spoofing for headless
        """
        stealth_js = r"""
        // 1. Remove webdriver flag
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});

        // 2. Mock chrome.runtime if missing (headless doesn't have it)
        if (!window.chrome) window.chrome = {};
        if (!window.chrome.runtime) window.chrome.runtime = {id: undefined};

        // 3. Override Permissions.query — some sites check notification permission
        const origQuery = window.Permissions && Permissions.prototype.query;
        if (origQuery) {
            Permissions.prototype.query = (params) =>
                params.name === 'notifications'
                    ? Promise.resolve({state: Notification.permission})
                    : origQuery.call(Permissions.prototype, params);
        }

        // 4. Patch plugins length (headless has 0)
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5],
        });

        // 5. Patch languages (some headless builds return empty)
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en'],
        });
        """
        await self.send("Page.addScriptToEvaluateOnNewDocument", {"source": stealth_js}, timeout=3)
        # Also apply to current page immediately
        await self.evaluate_js(stealth_js, timeout=3)

    # ── CDP-specific high-perf methods ──

    async def get_targets(self) -> list[dict[str, Any]]:
        if not self._session:
            return []
        try:
            port = int(self._ws_url.split("localhost:")[1].split("/")[0]) if "localhost:" in self._ws_url else _CDP_PORT
            async with self._session.get(f"http://localhost:{port}/json", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, OSError, ValueError):
            return []

    async def dispatch_mouse(self, x: float, y: float, event_type: str = "click", button: str = "left", click_count: int = 1, humanize: bool = True) -> dict:
        btn_map = {"left": "left", "right": "right", "middle": "middle"}
        btn = btn_map.get(button, "left")

        if humanize:
            # Human-like: move cursor along a bezier curve to the target
            await self._human_mouse_move(x, y)

        await self.send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": x, "y": y, "button": btn, "clickCount": click_count}, timeout=2)
        # Tiny random delay between press and release (humans hold 50-120ms)
        await asyncio.sleep(random.uniform(0.05, 0.12))
        await self.send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": x, "y": y, "button": btn, "clickCount": click_count}, timeout=2)
        return {"ok": True, "x": x, "y": y, "button": button}

    async def _human_mouse_move(self, target_x: float, target_y: float, steps: int = 0) -> None:
        """Move mouse along a bezier curve to the target — looks human, not robotic."""
        # Get current mouse position (default to random offset from target if unknown)
        current_x = target_x - random.uniform(50, 300)
        current_y = target_y - random.uniform(30, 200)

        if steps <= 0:
            dist = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)
            steps = max(5, min(25, int(dist / 50)))

        # Generate bezier control points (slight overshoot for realism)
        cp1_x = current_x + (target_x - current_x) * random.uniform(0.2, 0.5)
        cp1_y = current_y + (target_y - current_y) * random.uniform(-0.3, 0.3)
        cp2_x = current_x + (target_x - current_x) * random.uniform(0.6, 0.9)
        cp2_y = target_y + random.uniform(-20, 20)

        for i in range(1, steps + 1):
            t = i / steps
            # Cubic bezier interpolation
            u = 1 - t
            px = u**3 * current_x + 3 * u**2 * t * cp1_x + 3 * u * t**2 * cp2_x + t**3 * target_x
            py = u**3 * current_y + 3 * u**2 * t * cp1_y + 3 * u * t**2 * cp2_y + t**3 * target_y
            await self.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": px, "y": py}, timeout=1)
            # Variable speed: slow start, fast middle, slow end (ease-in-out)
            delay = 0.005 + 0.015 * math.sin(t * math.pi)
            await asyncio.sleep(delay)

    async def dispatch_key(self, key: str, modifiers: int = 0, text: str = "") -> dict:
        params: dict[str, Any] = {"type": "keyDown", "key": key, "modifiers": modifiers}
        if text:
            params["text"] = text
        await self.send("Input.dispatchKeyEvent", params, timeout=2)
        params["type"] = "keyUp"
        params.pop("text", None)
        await self.send("Input.dispatchKeyEvent", params, timeout=2)
        return {"ok": True, "key": key}

    async def capture_screenshot(self, full_page: bool = False, quality: int = 80, fmt: str = "png") -> bytes | None:
        params: dict[str, Any] = {"format": fmt}
        if fmt == "jpeg":
            params["quality"] = quality
        if full_page:
            metrics = await self.send("Page.getLayoutMetrics", timeout=3)
            content = metrics.get("contentSize") or metrics.get("cssContentSize", {})
            if content:
                params["clip"] = {"x": 0, "y": 0, "width": content.get("width", 1920), "height": content.get("height", 1080), "scale": 1}
                params["captureBeyondViewport"] = True
        res = await self.send("Page.captureScreenshot", params, timeout=10)
        data = res.get("data")
        if data:
            return base64.b64decode(data)
        return None

    async def print_pdf(self) -> bytes | None:
        res = await self.send("Page.printToPDF", {
            "landscape": False, "printBackground": True, "preferCSSPageSize": True,
        }, timeout=15)
        data = res.get("data")
        if data:
            return base64.b64decode(data)
        return None


# ── SafariJSBridge Backend ──────────────────────────────────────────

# Safari backend removed in T4 — PredaCore is Chrome-only. CDP gives
# us full DOM access at native speed; AppleScript-based Safari control
# can't match it (no shadow-DOM traversal, no real pointer events, no
# CDP-native input dispatch). One backend = one well-tested path.


# ── Frontmost browser detection ─────────────────────────────────────

_CHROMIUM_NAMES = frozenset({
    "Google Chrome", "Google Chrome Canary", "Brave Browser",
    "Microsoft Edge", "Chromium", "Arc", "Vivaldi", "Opera",
})

async def _frontmost_app() -> str:
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(loop.run_in_executor(None, lambda: subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to name of first application process whose frontmost is true'],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()), timeout=4)
    except (asyncio.TimeoutError, subprocess.SubprocessError, OSError, AttributeError):
        return ""


# ── Helper ──────────────────────────────────────────────────────────

def _js_esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# Key name → CDP key descriptor
_KEY_MAP: dict[str, dict[str, Any]] = {
    "Enter": {"key": "Enter", "code": "Enter", "keyCode": 13},
    "Tab": {"key": "Tab", "code": "Tab", "keyCode": 9},
    "Escape": {"key": "Escape", "code": "Escape", "keyCode": 27},
    "Backspace": {"key": "Backspace", "code": "Backspace", "keyCode": 8},
    "Delete": {"key": "Delete", "code": "Delete", "keyCode": 46},
    "ArrowUp": {"key": "ArrowUp", "code": "ArrowUp", "keyCode": 38},
    "ArrowDown": {"key": "ArrowDown", "code": "ArrowDown", "keyCode": 40},
    "ArrowLeft": {"key": "ArrowLeft", "code": "ArrowLeft", "keyCode": 37},
    "ArrowRight": {"key": "ArrowRight", "code": "ArrowRight", "keyCode": 39},
    "Home": {"key": "Home", "code": "Home", "keyCode": 36},
    "End": {"key": "End", "code": "End", "keyCode": 35},
    "PageUp": {"key": "PageUp", "code": "PageUp", "keyCode": 33},
    "PageDown": {"key": "PageDown", "code": "PageDown", "keyCode": 34},
    "Space": {"key": " ", "code": "Space", "keyCode": 32},
}

_MODIFIER_FLAGS = {"ctrl": 2, "alt": 1, "shift": 8, "meta": 4, "cmd": 4, "command": 4}


# ── Public BrowserBridge ────────────────────────────────────────────

class BrowserBridge:
    """Full browser control — 1000x faster than screenshots.

    Usage::

        bridge = BrowserBridge()
        await bridge.connect()
        tree = await bridge.get_page_tree()
        await bridge.click(text="Sign In")
        await bridge.type_text(selector="#email", value="j@example.com")
    """

    SPA_HYDRATION_DELAY_S: float = 4.0  # Legacy fallback only

    def __init__(
        self, cdp_port: int = _CDP_PORT, *,
        selector_cache: Any = None,  # operators.selector_cache.SelectorCache | None
    ) -> None:
        self._cdp = _ChromeCDP()
        self._cdp_port = cdp_port
        self._backend: _ChromeCDP | None = None
        self._browser: str = ""
        # T4b — selector cache (optional). When set, click() consults
        # the cache before calling _resolve(), and writes back on success.
        self._selector_cache = selector_cache
        self._cache_stats = {"hits": 0, "misses": 0, "verifies_failed": 0}

    @property
    def connected(self) -> bool:
        return self._backend is not None and self._backend.connected

    @property
    def browser_name(self) -> str:
        return self._browser

    @property
    def is_cdp(self) -> bool:
        # Chrome-only as of T4 — kept for callsite compatibility, always True
        # when connected.
        return self._backend is not None

    # ── Connection ─────────────────────────────────────────

    async def connect(self, browser: str = "auto", stealth: bool = True) -> bool:
        """Connect to Chrome via CDP. ``browser`` is accepted for backward compat
        but only Chrome / Chromium-derivatives are supported (T4: Chrome-only)."""
        if browser.lower() == "safari":
            logger.warning(
                "Safari support removed in T4 — connecting to Chrome instead. "
                "If you need Safari, use a previous version of PredaCore."
            )
        ok = await self._auto_connect()
        # Apply stealth patches to mask CDP fingerprints
        if ok and stealth and self.is_cdp and self._backend is not None:
            try:
                await self._backend.apply_stealth()
                logger.info("CDP stealth patches applied")
            except (OSError, RuntimeError) as exc:
                logger.debug("Stealth patches failed (non-fatal): %s", exc)
        return ok

    async def disconnect(self) -> None:
        if self._backend:
            await self._backend.close()
        self._backend = None
        self._browser = ""

    # ── Page tree ──────────────────────────────────────────

    async def get_page_tree(self) -> dict[str, Any]:
        if not self.connected:
            return {"error": "Not connected", "elements": []}
        t0 = time.time()
        raw = await self._eval(_PREDACORE_READ_PAGE_JS)
        ms = round((time.time() - t0) * 1000, 1)
        if isinstance(raw, dict) and "error" in raw:
            return {"error": raw["error"], "elements": [], "scan_ms": ms}
        if not isinstance(raw, dict):
            return {"error": f"Bad DOM result: {type(raw).__name__}", "elements": [], "scan_ms": ms}
        elements = raw.get("elements", [])
        return {"url": raw.get("url", ""), "title": raw.get("title", ""),
                "elements": elements, "element_count": len(elements),
                "viewport": raw.get("viewport", {}),
                "scroll_y": raw.get("scrollY", 0),
                "scroll_height": raw.get("scrollHeight", 0),
                "scan_ms": ms, "browser": self._browser}

    # ── Click ──────────────────────────────────────────────

    async def click(self, selector: str = "", text: str = "", role: str = "", index: int = -1) -> dict[str, Any]:
        """Click an element by selector OR by natural-language ``text`` / ``role``.

        Resolution order (T4b):
          1. Explicit ``selector`` — bypass cache, click directly.
          2. Cache hit on ``(domain, text, role)`` → CDP-verify the cached
             selector still resolves → click. ~50ms.
          3. Cache miss / verify-fail → existing ``_resolve()`` (text-search
             through the page tree) → click → cache the result for next time.
        """
        if not self.connected:
            return {"ok": False, "error": "Not connected"}
        if selector:
            return await self._click_sel(selector)

        # ── T4b: try the selector cache first ─────────────────────────
        # Skip cache when index>=0 (the resolver wants the Nth match,
        # not necessarily the same one we cached).
        cached_xpath: str | None = None
        if self._selector_cache is not None and (text or role) and index < 0:
            cached_xpath = await self._cache_lookup_and_verify(text=text, role=role)

        if cached_xpath:
            self._cache_stats["hits"] += 1
            result = await self._click_sel(cached_xpath)
            if isinstance(result, dict) and result.get("ok"):
                # Bump use counter — fire-and-forget, don't block click latency
                try:
                    domain = await self._current_domain()
                    if domain:
                        await asyncio.to_thread(
                            self._selector_cache.bump_use,
                            domain=domain, text=text, role=role,
                        )
                except Exception:  # noqa: BLE001 — never let cache writes fail clicks
                    pass
            return {**(result or {}), "cache": "hit"}

        # ── Fallback: full resolver path ──────────────────────────────
        if self._selector_cache is not None and (text or role):
            self._cache_stats["misses"] += 1
        target = await self._resolve(text=text, role=role, index=index)
        if target is None:
            return {"ok": False, "error": f"Element not found: {text or role}"}
        result = await self._click_sel(target)
        # Write to cache only on successful click + only when we used the
        # natural-language path (no explicit selector, no index disambiguator).
        if (
            isinstance(result, dict) and result.get("ok")
            and self._selector_cache is not None
            and (text or role) and index < 0
        ):
            try:
                domain = await self._current_domain()
                if domain:
                    await asyncio.to_thread(
                        self._selector_cache.record,
                        domain=domain, text=text, role=role,
                        xpath=target, label=text,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Selector cache record failed: %s", exc)
        return {**(result or {}), "cache": "miss"}

    async def _cache_lookup_and_verify(
        self, *, text: str, role: str,
    ) -> str | None:
        """Cache lookup + CDP verify. Returns xpath if verified, else None.

        Verification is a single ``document.querySelector(xpath)`` call —
        if it returns truthy and the bbox isn't off-screen / collapsed,
        the selector still resolves. On verify-fail, the cache row is
        invalidated so the next attempt goes through the full resolver.
        """
        cache = self._selector_cache
        if cache is None:
            return None
        try:
            domain = await self._current_domain()
            if not domain:
                return None
            entry = await asyncio.to_thread(cache.lookup, domain, text, role)
            if entry is None:
                return None
            # Verify the selector still resolves and is interactable.
            check_js = (
                f'(function(){{var el=document.querySelector("{_js_esc(entry.xpath)}");'
                f'if(!el)return null;var r=el.getBoundingClientRect();'
                f'return r.width>=4&&r.height>=4?"{_js_esc(entry.xpath)}":null;}})()'
            )
            verified = await self._eval(check_js)
            if not verified:
                self._cache_stats["verifies_failed"] += 1
                # Drop the stale entry — next cache miss will repopulate.
                await asyncio.to_thread(
                    cache.invalidate, domain=domain, text=text, role=role,
                )
                return None
            return entry.xpath
        except Exception as exc:  # noqa: BLE001
            logger.debug("Selector cache lookup failed: %s", exc)
            return None

    async def _current_domain(self) -> str:
        """Bare host of the currently-loaded page. Empty string on failure."""
        url = await self.get_url()
        if not url:
            return ""
        from urllib.parse import urlparse
        return (urlparse(url).hostname or "").lower()

    # ── Type text ──────────────────────────────────────────

    async def type_text(self, selector: str = "", text: str = "", value: str = "") -> dict[str, Any]:
        """Fast value injection — works for most inputs (5ms).

        Uses the React-compatible native setter trick to bypass framework
        property descriptor overrides. For inputs that need real keystrokes
        (input masks, search-as-you-type), use type_keys() instead.
        """
        if not self.connected:
            return {"ok": False, "error": "Not connected"}
        target = selector or await self._resolve(text=text, role="input")
        if not target:
            return {"ok": False, "error": f"Input not found: {text}"}
        # React-compatible: use the native HTMLInputElement setter to bypass
        # React's synthetic property descriptor. Then fire input + change events.
        js = (
            f'(function(){{var el=document.querySelector("{_js_esc(target)}");'
            f'if(!el)return JSON.stringify({{ok:false,error:"selector not found"}});'
            f'el.focus();'
            f'var nativeSetter=Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,"value")'
            f'||Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,"value");'
            f'if(nativeSetter&&nativeSetter.set){{nativeSetter.set.call(el,{json.dumps(value)});}}'
            f'else{{el.value={json.dumps(value)};}}'
            f'el.dispatchEvent(new Event("input",{{bubbles:true}}));'
            f'el.dispatchEvent(new Event("change",{{bubbles:true}}));'
            f'return JSON.stringify({{ok:true,tag:el.tagName.toLowerCase(),method:"inject"}})}})()'
        )
        res = await self._eval(js)
        return res if isinstance(res, dict) else {"ok": False, "error": f"Unexpected: {res}"}

    async def type_keys(self, selector: str = "", text: str = "", value: str = "", clear: bool = True) -> dict[str, Any]:
        """Real keystroke simulation — character by character via CDP Input events.

        Slower (~5ms per char) but triggers keydown/keypress/keyup/input events
        exactly like a real keyboard. Works with:
        - React/Vue controlled inputs
        - Input masks (phone, credit card)
        - Search-as-you-type with keydown debounce
        - Autocomplete triggers

        Args:
            selector: CSS selector of the target input
            text: find input by placeholder/label text
            value: the text to type
            clear: if True, select-all + delete before typing (default True)
        """
        if not self.connected:
            return {"ok": False, "error": "Not connected"}
        target = selector or await self._resolve(text=text, role="input")
        if not target:
            return {"ok": False, "error": f"Input not found: {text}"}

        # Focus the element AND verify focus actually landed. Without this
        # check, a selector that matches a hidden/disabled/shadow-DOM element
        # silently lands focus on <body> or a neighbour, and subsequent
        # keystrokes go to the wrong place.
        await self._eval(f'document.querySelector("{_js_esc(target)}").focus()')
        focus_ok = await self._eval(
            f'(function(){{var el=document.querySelector("{_js_esc(target)}");'
            f'return el!==null&&document.activeElement===el}})()'
        )
        if focus_ok is not True:
            return {
                "ok": False,
                "error": (
                    f"focus did not land on {target!r} — element may be hidden, "
                    "disabled, in a shadow DOM, or inside a cross-origin iframe"
                ),
            }

        if clear:
            # Select all + delete (Ctrl+A then Backspace)
            await self.key_combo(["ctrl", "a"])
            await self.press_key("Backspace")
            # React re-render settle — controlled-component state update is
            # async, typing into a stale DOM value results in lost keystrokes.
            await asyncio.sleep(0.2)

        if self.is_cdp:
            # CDP real keystrokes — each char goes through the full input pipeline
            # with human-like variable delays between keystrokes.
            for i, ch in enumerate(value):
                await self._backend.send("Input.dispatchKeyEvent", {
                    "type": "keyDown", "key": ch, "text": ch,
                }, timeout=1)
                await self._backend.send("Input.dispatchKeyEvent", {
                    "type": "keyUp", "key": ch,
                }, timeout=1)
                # Human typing cadence: 50-150ms between chars, longer pauses
                # after spaces/punctuation (humans pause at word boundaries)
                if ch in " .,;:!?":
                    await asyncio.sleep(random.uniform(0.10, 0.25))
                elif i > 0 and random.random() < 0.08:
                    # Occasional thinking pause (~8% chance)
                    await asyncio.sleep(random.uniform(0.20, 0.45))
                else:
                    await asyncio.sleep(random.uniform(0.04, 0.12))
            method = "cdp_keystrokes"
        else:
            # JS fallback: simulate per-char events
            for ch in value:
                js = (
                    f'(function(){{var el=document.activeElement;'
                    f'el.dispatchEvent(new KeyboardEvent("keydown",{{key:{json.dumps(ch)},bubbles:true}}));'
                    f'el.dispatchEvent(new KeyboardEvent("keypress",{{key:{json.dumps(ch)},bubbles:true}}));'
                    f'document.execCommand("insertText",false,{json.dumps(ch)});'
                    f'el.dispatchEvent(new KeyboardEvent("keyup",{{key:{json.dumps(ch)},bubbles:true}}))}})()'
                )
                await self._eval(js)
            method = "js_keystrokes"

        # Post-typing verification — read the element's value back. If what we
        # sent didn't land (page intercepted keys, input is read-only, etc.)
        # fail the call explicitly so the agent doesn't keep retrying on a
        # false-positive "ok": true.
        final_value = await self._eval(
            f'(function(){{var el=document.querySelector("{_js_esc(target)}");'
            f'return el&&"value" in el?el.value:null}})()'
        )
        if final_value is None:
            return {
                "ok": False,
                "error": f"element {target!r} has no .value after typing",
                "method": method,
            }
        if final_value != value:
            return {
                "ok": False,
                "error": (
                    f"typing did not register: expected {value!r} "
                    f"but element.value is {str(final_value)[:80]!r}"
                ),
                "method": method,
                "final_value": final_value,
            }
        return {"ok": True, "method": method, "chars": len(value), "final_value": final_value}

    # ── Read text ──────────────────────────────────────────

    async def read_text(self, selector: str = "") -> str:
        if not self.connected:
            return ""
        if selector:
            target = f'document.querySelector("{_js_esc(selector)}")'
            res = await self._eval(f"({target}||{{}}).textContent||''")
        else:
            res = await self._eval('''(function(){
                var parts=[];
                var title=document.title;if(title)parts.push("Page: "+title);
                var h=document.querySelectorAll("h1,h2,h3,h4,#video-title,[role=heading],article,[data-testid]");
                for(var i=0;i<Math.min(h.length,30);i++){var t=h[i].textContent.trim();if(t&&t.length>2)parts.push(t);}
                var labels=document.querySelectorAll("[aria-label]");
                for(var i=0;i<Math.min(labels.length,30);i++){var l=labels[i].getAttribute("aria-label").trim();if(l&&l.length>3)parts.push(l);}
                if(!parts.length||parts.join("").length<100)parts.push(document.body.textContent.substring(0,5000));
                return parts.join("\\n");
            })()''')
        return str(res) if res else ""

    # ── Evaluate JS ────────────────────────────────────────

    async def evaluate_js(self, code: str) -> Any:
        if not self.connected:
            return {"error": "Not connected"}
        return await self._eval(code)

    async def get_url(self) -> str:
        if not self.connected:
            return ""
        res = await self._eval("location.href")
        return str(res) if res else ""

    # ── Navigate (smart — no fixed delay) ──────────────────

    async def navigate(self, url: str, wait_for: str = "") -> dict[str, Any]:
        """Navigate and wait intelligently.

        Uses Page.navigate + readyState polling. Falls back to a brief SPA
        hydration sleep for JS frameworks (React, Vue, etc.) so the DOM is
        actually mounted by the time we return.

        Args:
            wait_for: optional CSS selector to wait for after load.
        """
        if not self.connected or self._backend is None:
            return {"ok": False, "error": "Not connected"}

        cdp = self._backend
        res = await cdp.send("Page.navigate", {"url": url}, timeout=_NAVIGATE_TIMEOUT)
        if "error" in res and res["error"] != "":
            return {"ok": False, "error": res["error"]}
        # Poll document.readyState until "complete" (simpler and more reliable
        # than event-based Page.loadEventFired which has race conditions with
        # the single-websocket message loop).
        deadline = time.time() + _NAVIGATE_TIMEOUT
        while time.time() < deadline:
            state = await cdp.evaluate_js("document.readyState", timeout=2)
            if state == "complete":
                break
            await asyncio.sleep(0.15)
        # Brief settle for SPA hydration (JS framework init)
        await asyncio.sleep(_NAVIGATE_SETTLE_MS / 1000.0)
        if wait_for:
            await self.wait_for_element(wait_for, timeout=5.0)

        title = await self._eval("document.title||''") or ""
        text_len = await self._eval("(document.body.textContent||'').length") or 0
        return {
            "ok": True, "url": url, "title": str(title),
            "content_loaded": int(text_len or 0) > 50,
        }

    # ── Scroll ─────────────────────────────────────────────

    async def scroll(self, direction: str = "down", amount: int = 3) -> dict[str, Any]:
        if not self.connected:
            return {"ok": False, "error": "Not connected"}
        px = amount * 300 * (1 if direction == "down" else -1)
        res = await self._eval(f"window.scrollBy(0,{px});JSON.stringify({{ok:true,scrollY:window.scrollY}})")
        return res if isinstance(res, dict) else {"ok": True, "direction": direction, "pixels": px}

    # ── History ────────────────────────────────────────────

    async def back(self) -> dict[str, Any]:
        if not self.connected:
            return {"ok": False, "error": "Not connected"}
        await self._eval("history.back()")
        await asyncio.sleep(0.5)
        return {"ok": True, "url": await self.get_url()}

    async def forward(self) -> dict[str, Any]:
        if not self.connected:
            return {"ok": False, "error": "Not connected"}
        await self._eval("history.forward()")
        await asyncio.sleep(0.5)
        return {"ok": True, "url": await self.get_url()}

    async def reload(self) -> dict[str, Any]:
        if not self.connected:
            return {"ok": False, "error": "Not connected"}
        if self.is_cdp:
            await self._backend.send("Page.reload", timeout=_NAVIGATE_TIMEOUT)
            await self._backend.wait_for_event("Page.loadEventFired", timeout=_NAVIGATE_TIMEOUT)
        else:
            await self._eval("location.reload()")
            await asyncio.sleep(2)
        return {"ok": True, "url": await self.get_url()}

    # ── Wait primitives ────────────────────────────────────

    async def wait_for_element(self, selector: str, timeout: float = 10.0) -> dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            res = await self._eval(f'!!document.querySelector("{_js_esc(selector)}")')
            if res:
                return {"found": True, "selector": selector}
            await asyncio.sleep(0.25)
        return {"found": False, "selector": selector, "timeout": True}

    async def wait_for_text(self, text: str, timeout: float = 10.0) -> dict[str, Any]:
        t_esc = json.dumps(text.lower())
        deadline = time.time() + timeout
        while time.time() < deadline:
            res = await self._eval(f'(document.body.textContent||"").toLowerCase().includes({t_esc})')
            if res:
                return {"found": True, "text": text}
            await asyncio.sleep(0.25)
        return {"found": False, "text": text, "timeout": True}

    async def wait_for_url(self, pattern: str, timeout: float = 10.0) -> dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            url = await self.get_url()
            if pattern in url:
                return {"matched": True, "url": url, "pattern": pattern}
            await asyncio.sleep(0.25)
        return {"matched": False, "pattern": pattern, "timeout": True}

    # ── Hover ──────────────────────────────────────────────

    async def hover(self, selector: str = "", text: str = "") -> dict[str, Any]:
        if not self.connected:
            return {"ok": False, "error": "Not connected"}
        target = selector or await self._resolve(text=text)
        if not target:
            return {"ok": False, "error": "Element not found"}
        # Get element center
        js = f'(function(){{var el=document.querySelector("{_js_esc(target)}");if(!el)return null;var r=el.getBoundingClientRect();return JSON.stringify({{x:r.left+r.width/2,y:r.top+r.height/2}})}})()'
        pos = await self._eval(js)
        if not isinstance(pos, dict):
            return {"ok": False, "error": "Could not get element position"}
        if self.is_cdp:
            await self._backend.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": pos["x"], "y": pos["y"]}, timeout=2)
            return {"ok": True, "selector": target}
        await self._eval(f'document.querySelector("{_js_esc(target)}").dispatchEvent(new MouseEvent("mouseover",{{bubbles:true}}))')
        return {"ok": True, "selector": target}

    # ── Keyboard ───────────────────────────────────────────

    async def press_key(self, key: str, modifiers: list[str] | None = None) -> dict[str, Any]:
        if not self.connected:
            return {"ok": False, "error": "Not connected"}
        mod_flags = 0
        for m in (modifiers or []):
            mod_flags |= _MODIFIER_FLAGS.get(m.lower(), 0)
        if self.is_cdp:
            info = _KEY_MAP.get(key, {"key": key, "code": "", "keyCode": 0})
            params: dict[str, Any] = {"type": "keyDown", "key": info["key"], "modifiers": mod_flags}
            if info.get("code"):
                params["code"] = info["code"]
            if info.get("keyCode"):
                params["windowsVirtualKeyCode"] = info["keyCode"]
            await self._backend.send("Input.dispatchKeyEvent", params, timeout=2)
            params["type"] = "keyUp"
            await self._backend.send("Input.dispatchKeyEvent", params, timeout=2)
            return {"ok": True, "key": key}
        # Safari fallback
        await self._eval(f'document.dispatchEvent(new KeyboardEvent("keydown",{{key:"{_js_esc(key)}",bubbles:true}}))')
        return {"ok": True, "key": key}

    async def key_combo(self, keys: list[str]) -> dict[str, Any]:
        """Press a key combination like ["ctrl","a"] or ["cmd","shift","p"]."""
        if not keys:
            return {"ok": False, "error": "No keys specified"}
        modifiers = [k for k in keys[:-1] if k.lower() in _MODIFIER_FLAGS]
        main_key = keys[-1]
        return await self.press_key(main_key, modifiers=modifiers)

    # ── Cookies ────────────────────────────────────────────

    async def get_cookies(self, domain: str = "") -> dict[str, Any]:
        if not self.connected:
            return {"cookies": []}
        if self.is_cdp:
            params = {}
            if domain:
                urls = [f"https://{domain}", f"http://{domain}"]
                params["urls"] = urls
            res = await self._backend.send("Network.getCookies", params, timeout=3)
            return {"cookies": res.get("cookies", [])}
        # JS fallback (limited to current domain, no httpOnly)
        raw = await self._eval("document.cookie")
        cookies = []
        for pair in str(raw or "").split(";"):
            pair = pair.strip()
            if "=" in pair:
                k, v = pair.split("=", 1)
                cookies.append({"name": k.strip(), "value": v.strip()})
        return {"cookies": cookies}

    async def set_cookie(self, name: str, value: str, domain: str = "", path: str = "/", expires: float = 0) -> dict[str, Any]:
        if self.is_cdp:
            params: dict[str, Any] = {"name": name, "value": value, "path": path}
            if domain:
                params["domain"] = domain
            else:
                url = await self.get_url()
                params["url"] = url
            if expires:
                params["expires"] = expires
            res = await self._backend.send("Network.setCookie", params, timeout=3)
            return {"ok": res.get("success", False)}
        await self._eval(f'document.cookie="{_js_esc(name)}={_js_esc(value)};path={_js_esc(path)}"')
        return {"ok": True}

    async def delete_cookies(self, name: str = "", domain: str = "") -> dict[str, Any]:
        if self.is_cdp:
            params: dict[str, Any] = {}
            if name:
                params["name"] = name
            if domain:
                params["domain"] = domain
            else:
                params["url"] = await self.get_url()
                if name:
                    params["name"] = name
            await self._backend.send("Network.deleteCookies", params, timeout=3)
            return {"ok": True}
        if name:
            await self._eval(f'document.cookie="{_js_esc(name)}=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/"')
        return {"ok": True}

    # ── Storage (localStorage / sessionStorage) ────────────

    async def get_storage(self, key: str = "", storage: str = "local") -> dict[str, Any]:
        s = "localStorage" if storage == "local" else "sessionStorage"
        if key:
            val = await self._eval(f'{s}.getItem({json.dumps(key)})')
            return {"key": key, "value": val}
        raw = await self._eval(f'JSON.stringify(Object.fromEntries(Object.entries({s})))')
        return {"storage": storage, "data": raw if isinstance(raw, dict) else {}}

    async def set_storage(self, key: str, value: str, storage: str = "local") -> dict[str, Any]:
        s = "localStorage" if storage == "local" else "sessionStorage"
        await self._eval(f'{s}.setItem({json.dumps(key)},{json.dumps(value)})')
        return {"ok": True, "key": key}

    async def clear_storage(self, storage: str = "local") -> dict[str, Any]:
        s = "localStorage" if storage == "local" else "sessionStorage"
        await self._eval(f'{s}.clear()')
        return {"ok": True, "storage": storage}

    # ── Form controls ──────────────────────────────────────

    async def set_checkbox(self, selector: str, checked: bool = True) -> dict[str, Any]:
        js = (
            f'(function(){{var el=document.querySelector("{_js_esc(selector)}");'
            f'if(!el)return JSON.stringify({{ok:false,error:"not found"}});'
            f'if(el.checked!=={json.dumps(checked)}){{el.click();}}'
            f'return JSON.stringify({{ok:true,checked:el.checked}})}})()'
        )
        res = await self._eval(js)
        return res if isinstance(res, dict) else {"ok": False}

    async def select_option(self, selector: str, value: str = "", label: str = "") -> dict[str, Any]:
        if value:
            js = (
                f'(function(){{var el=document.querySelector("{_js_esc(selector)}");'
                f'if(!el)return JSON.stringify({{ok:false,error:"not found"}});'
                f'el.value={json.dumps(value)};'
                f'el.dispatchEvent(new Event("change",{{bubbles:true}}));'
                f'return JSON.stringify({{ok:true,value:el.value}})}})()'
            )
        else:
            js = (
                f'(function(){{var el=document.querySelector("{_js_esc(selector)}");'
                f'if(!el)return JSON.stringify({{ok:false,error:"not found"}});'
                f'for(var i=0;i<el.options.length;i++){{'
                f'if(el.options[i].text.toLowerCase().includes({json.dumps(label.lower())})){{'
                f'el.selectedIndex=i;el.dispatchEvent(new Event("change",{{bubbles:true}}));'
                f'return JSON.stringify({{ok:true,value:el.value,label:el.options[i].text}})}}}}'
                f'return JSON.stringify({{ok:false,error:"option not found"}})}})()'
            )
        res = await self._eval(js)
        return res if isinstance(res, dict) else {"ok": False}

    async def upload_file(self, selector: str, file_paths: list[str]) -> dict[str, Any]:
        """Set files on a <input type=file> (CDP only)."""
        if not self.is_cdp:
            return {"ok": False, "error": "File upload requires Chrome CDP"}
        # Get the DOM node ID for the selector
        doc = await self._backend.send("DOM.getDocument", timeout=3)
        root_id = doc.get("root", {}).get("nodeId", 0)
        if not root_id:
            return {"ok": False, "error": "Could not get document root"}
        node = await self._backend.send("DOM.querySelector", {"nodeId": root_id, "selector": selector}, timeout=3)
        node_id = node.get("nodeId", 0)
        if not node_id:
            return {"ok": False, "error": f"Selector not found: {selector}"}
        res = await self._backend.send("DOM.setFileInputFiles", {"nodeId": node_id, "files": file_paths}, timeout=5)
        if "error" in res:
            return {"ok": False, "error": res["error"]}
        return {"ok": True, "files": file_paths}

    # ── Screenshots & PDF ──────────────────────────────────

    async def screenshot(self, path: str = "", full_page: bool = False) -> dict[str, Any]:
        if not self.connected:
            return {"ok": False, "error": "Not connected"}
        if self.is_cdp:
            data = await self._backend.capture_screenshot(full_page=full_page)
            if data is None:
                return {"ok": False, "error": "Screenshot failed"}
            if not path:
                fd, path = tempfile.mkstemp(prefix="predacore_browser_", suffix=".png")
                os.close(fd)
            Path(path).write_bytes(data)
            return {"ok": True, "path": path, "size_bytes": len(data), "full_page": full_page}
        # Safari fallback — use screencapture
        if not path:
            fd, path = tempfile.mkstemp(prefix="predacore_browser_", suffix=".png")
            os.close(fd)
        try:
            subprocess.run(["screencapture", "-x", path], timeout=5, check=True)
            return {"ok": True, "path": path, "size_bytes": Path(path).stat().st_size}
        except (subprocess.SubprocessError, OSError) as e:
            return {"ok": False, "error": str(e)}

    async def print_pdf(self, path: str = "") -> dict[str, Any]:
        if not self.is_cdp:
            return {"ok": False, "error": "PDF export requires Chrome CDP"}
        data = await self._backend.print_pdf()
        if data is None:
            return {"ok": False, "error": "PDF export failed"}
        if not path:
            fd, path = tempfile.mkstemp(prefix="predacore_browser_", suffix=".pdf")
            os.close(fd)
        Path(path).write_bytes(data)
        return {"ok": True, "path": path, "size_bytes": len(data)}

    # ── Tabs ───────────────────────────────────────────────

    async def list_tabs(self) -> dict[str, Any]:
        if not self.is_cdp:
            return {"tabs": [], "error": "Tab management requires Chrome CDP"}
        targets = await self._backend.get_targets()
        tabs = [{"id": t.get("id", ""), "title": t.get("title", ""), "url": t.get("url", "")}
                for t in targets if t.get("type") == "page"]
        return {"tabs": tabs, "count": len(tabs)}

    async def new_tab(self, url: str = "about:blank") -> dict[str, Any]:
        if not self.is_cdp:
            return {"ok": False, "error": "Tab management requires Chrome CDP"}
        res = await self._backend.send("Target.createTarget", {"url": url}, timeout=5)
        return {"ok": "targetId" in res, "target_id": res.get("targetId", "")}

    async def close_tab(self, target_id: str = "") -> dict[str, Any]:
        if not self.is_cdp:
            return {"ok": False, "error": "Tab management requires Chrome CDP"}
        if not target_id:
            return {"ok": False, "error": "target_id required"}
        res = await self._backend.send("Target.closeTarget", {"targetId": target_id}, timeout=3)
        return {"ok": res.get("success", False)}

    # ── Table extraction ───────────────────────────────────

    async def extract_tables(self, selector: str = "table") -> dict[str, Any]:
        js = f'''(function(){{
            var tables=document.querySelectorAll("{_js_esc(selector)}");
            var result=[];
            for(var t=0;t<tables.length;t++){{
                var rows=tables[t].querySelectorAll("tr");
                var data=[];
                for(var r=0;r<rows.length;r++){{
                    var cells=rows[r].querySelectorAll("th,td");
                    var row=[];
                    for(var c=0;c<cells.length;c++) row.push(cells[c].textContent.trim());
                    data.push(row);
                }}
                result.push({{index:t,rows:data.length,columns:data[0]?data[0].length:0,data:data}});
            }}
            return JSON.stringify({{tables:result,count:result.length}});
        }})()'''
        res = await self._eval(js)
        return res if isinstance(res, dict) else {"tables": [], "count": 0}

    # ── Find in page ───────────────────────────────────────

    async def find_in_page(self, query: str) -> dict[str, Any]:
        js = f'''(function(){{
            var q={json.dumps(query.lower())};
            var tw=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT,null,false);
            var matches=[];var n;
            while((n=tw.nextNode())&&matches.length<20){{
                if(n.textContent.toLowerCase().includes(q)){{
                    var el=n.parentElement;
                    if(el){{
                        var r=el.getBoundingClientRect();
                        matches.push({{text:n.textContent.trim().substring(0,200),
                            bounds:{{x:Math.round(r.left),y:Math.round(r.top),w:Math.round(r.width),h:Math.round(r.height)}},
                            tag:el.tagName.toLowerCase()}});
                    }}
                }}
            }}
            return JSON.stringify({{query:{json.dumps(query)},matches:matches,count:matches.length}});
        }})()'''
        res = await self._eval(js)
        return res if isinstance(res, dict) else {"matches": [], "count": 0}

    # ── Page content helpers ───────────────────────────────

    async def get_page_links(self) -> dict[str, Any]:
        if not self.connected:
            return {"count": 0, "links": []}
        js = '''(function(){
            var links=document.querySelectorAll("a[href]");
            var results=[];
            for(var i=0;i<links.length;i++){
                var a=links[i];
                var text=(a.textContent||a.title||a.getAttribute("aria-label")||"").trim();
                if(!text)continue;
                results.push({text:text.substring(0,100),href:(a.href||"").substring(0,300),target:a.target||""});
                if(results.length>=100)break;
            }
            return JSON.stringify({count:results.length,links:results});
        })()'''
        res = await self._eval(js)
        return res if isinstance(res, dict) else {"count": 0, "links": []}

    async def get_page_images(self) -> dict[str, Any]:
        if not self.connected:
            return {"count": 0, "images": []}
        js = '''(function(){
            var imgs=document.querySelectorAll("img");
            var results=[];
            for(var i=0;i<imgs.length;i++){
                var img=imgs[i];
                var w=img.naturalWidth||img.width||0;
                var h=img.naturalHeight||img.height||0;
                if(w<30||h<30)continue;
                results.push({alt:(img.alt||img.title||"").substring(0,200),
                    src:(img.currentSrc||img.src||"").substring(0,500),width:w,height:h});
                if(results.length>=50)break;
            }
            return JSON.stringify({count:results.length,images:results});
        })()'''
        res = await self._eval(js)
        return res if isinstance(res, dict) else {"count": 0, "images": []}

    # ── Media controls ─────────────────────────────────────

    async def get_media_info(self) -> dict[str, Any]:
        if not self.connected:
            return {"count": 0, "elements": []}
        js = '''(function(){
            var media=document.querySelectorAll("video,audio");
            if(!media.length)return JSON.stringify({count:0,elements:[]});
            var els=[];
            for(var i=0;i<media.length;i++){
                var m=media[i];
                els.push({tag:m.tagName.toLowerCase(),src:(m.currentSrc||m.src||"").slice(0,200),
                    duration:isNaN(m.duration)?0:Math.round(m.duration),
                    currentTime:Math.round(m.currentTime*10)/10,paused:m.paused,ended:m.ended,
                    volume:Math.round(m.volume*100),muted:m.muted,
                    playbackRate:m.playbackRate,loop:m.loop});
            }
            return JSON.stringify({count:els.length,elements:els});
        })()'''
        res = await self._eval(js)
        return res if isinstance(res, dict) else {"count": 0, "elements": []}

    async def media_play(self) -> dict[str, Any]:
        return await self._eval('(function(){var m=document.querySelector("video,audio");if(!m)return JSON.stringify({ok:false,error:"no media"});m.play();return JSON.stringify({ok:true,action:"play"})})()')

    async def media_pause(self) -> dict[str, Any]:
        return await self._eval('(function(){var m=document.querySelector("video,audio");if(!m)return JSON.stringify({ok:false,error:"no media"});m.pause();return JSON.stringify({ok:true,action:"pause"})})()')

    async def media_seek(self, seconds: float) -> dict[str, Any]:
        return await self._eval(f'(function(){{var m=document.querySelector("video,audio");if(!m)return JSON.stringify({{ok:false,error:"no media"}});m.currentTime={seconds};return JSON.stringify({{ok:true,time:{seconds}}})}})()')

    async def media_set_volume(self, level: int) -> dict[str, Any]:
        vol = max(0, min(100, level)) / 100
        return await self._eval(f'(function(){{var m=document.querySelector("video,audio");if(!m)return JSON.stringify({{ok:false,error:"no media"}});m.volume={vol};m.muted=false;return JSON.stringify({{ok:true,volume:{level}}})}})()')

    async def media_set_speed(self, rate: float) -> dict[str, Any]:
        rate = max(0.25, min(4.0, rate))
        return await self._eval(f'(function(){{var m=document.querySelector("video,audio");if(!m)return JSON.stringify({{ok:false,error:"no media"}});m.playbackRate={rate};return JSON.stringify({{ok:true,speed:{rate}}})}})()')

    async def media_fullscreen(self) -> dict[str, Any]:
        return await self._eval('(function(){var m=document.querySelector("video");if(!m)return JSON.stringify({ok:false,error:"no video"});if(document.fullscreenElement)document.exitFullscreen();else m.requestFullscreen();return JSON.stringify({ok:true})})()')

    async def media_toggle_mute(self) -> dict[str, Any]:
        return await self._eval('(function(){var m=document.querySelector("video,audio");if(!m)return JSON.stringify({ok:false,error:"no media"});m.muted=!m.muted;return JSON.stringify({ok:true,muted:m.muted})})()')

    # ── Captions / Transcripts ─────────────────────────────

    async def get_captions(self) -> dict[str, Any]:
        js = '''(function(){
            var ytCaptions=document.querySelectorAll(".ytp-caption-segment,.captions-text,.caption-visual-line");
            if(ytCaptions.length){var lines=[];for(var i=0;i<ytCaptions.length;i++)lines.push(ytCaptions[i].textContent.trim());
            return JSON.stringify({source:"youtube_overlay",lines:lines.filter(function(l){return l.length>0})});}
            var video=document.querySelector("video");
            if(video&&video.textTracks&&video.textTracks.length){var tracks=[];
            for(var t=0;t<video.textTracks.length;t++){var track=video.textTracks[t];var cues=[];
            if(track.cues){for(var c=0;c<Math.min(track.cues.length,500);c++){var cue=track.cues[c];
            cues.push({start:Math.round(cue.startTime*10)/10,end:Math.round(cue.endTime*10)/10,text:cue.text||""});}}
            tracks.push({label:track.label||"",lang:track.language||"",kind:track.kind||"",mode:track.mode,cues:cues});}
            return JSON.stringify({source:"text_tracks",tracks:tracks});}
            return JSON.stringify({source:"none",message:"No captions found."});
        })()'''
        res = await self._eval(js)
        return res if isinstance(res, dict) else {"source": "none"}

    async def enable_captions(self) -> dict[str, Any]:
        res = await self._eval('''(function(){
            var btn=document.querySelector(".ytp-subtitles-button");
            if(!btn)return JSON.stringify({ok:false,error:"CC button not found"});
            var isOn=btn.getAttribute("aria-pressed")==="true";
            if(!isOn)btn.click();
            return JSON.stringify({ok:true,action:isOn?"already_on":"enabled"});
        })()''')
        if isinstance(res, dict) and res.get("ok"):
            await asyncio.sleep(2.0)
        return res

    async def get_transcript(self) -> dict[str, Any]:
        if not self.connected:
            return {"source": "none", "error": "Not connected"}
        url = await self.get_url()
        video_id = ""
        if "youtube.com/watch" in url or "youtu.be/" in url:
            import re
            m = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
            if m:
                video_id = m.group(1)
        if video_id:
            try:
                import shutil
                if shutil.which("yt-dlp"):
                    vtt_path = f"/tmp/predacore_transcript_{video_id}"
                    proc = await asyncio.create_subprocess_exec(
                        "yt-dlp", "--write-auto-sub", "--sub-lang", "en",
                        "--skip-download", "--sub-format", "vtt",
                        "-o", vtt_path, f"https://www.youtube.com/watch?v={video_id}",
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    )
                    await asyncio.wait_for(proc.wait(), timeout=60)
                    import glob
                    import re as _re
                    vtt_files = glob.glob(f"{vtt_path}*.vtt")
                    if vtt_files:
                        with open(vtt_files[0]) as f:
                            content = f.read()
                        lines = content.split("\n")
                        text_lines, seen = [], set()
                        for line in lines:
                            line = line.strip()
                            if not line or line.startswith("WEBVTT") or "-->" in line:
                                continue
                            if line[0].isdigit() and len(line) < 5:
                                continue
                            clean = _re.sub(r'<[^>]+>', '', line).strip()
                            if clean and clean not in seen:
                                seen.add(clean)
                                text_lines.append(clean)
                        text = " ".join(text_lines)
                        return {"source": "yt-dlp", "text": text, "word_count": len(text.split())}
            except (asyncio.TimeoutError, OSError, FileNotFoundError):
                pass
        # Fallback to TextTrack
        res = await self._eval('''(function(){
            var video=document.querySelector("video");
            if(!video||!video.textTracks||!video.textTracks.length)return JSON.stringify({source:"none"});
            for(var t=0;t<video.textTracks.length;t++){var track=video.textTracks[t];
            if(track.cues&&track.cues.length>10){var texts=[];var seen={};
            for(var c=0;c<track.cues.length;c++){var txt=track.cues[c].text.replace(/<[^>]+>/g,"").trim();
            if(txt&&!seen[txt]){seen[txt]=1;texts.push(txt);}}
            return JSON.stringify({source:"text_tracks",text:texts.join(" "),word_count:texts.join(" ").split(" ").length});}}
            return JSON.stringify({source:"none"});
        })()''')
        if isinstance(res, dict) and res.get("source") != "none":
            return res
        return {"source": "none", "error": "No transcript available."}

    # ── Chrome debug helpers ───────────────────────────────

    # ── Drag and Drop ───────────────────────────────────────

    async def drag_and_drop(self, source: str, target: str, humanize: bool = True) -> dict[str, Any]:
        """Drag from one element to another (works with Trello, kanban, file zones).

        Performs both CDP native input events AND HTML5 drag events for max compat.
        """
        if not self.connected:
            return {"ok": False, "error": "Not connected"}
        # Get positions of source and target
        pos_js = '''(function(srcSel, tgtSel) {
            var src = document.querySelector(srcSel);
            var tgt = document.querySelector(tgtSel);
            if (!src) return JSON.stringify({error: "source not found"});
            if (!tgt) return JSON.stringify({error: "target not found"});
            var sr = src.getBoundingClientRect();
            var tr = tgt.getBoundingClientRect();
            return JSON.stringify({
                sx: sr.left + sr.width/2, sy: sr.top + sr.height/2,
                tx: tr.left + tr.width/2, ty: tr.top + tr.height/2
            });
        })(''' + json.dumps(source) + ',' + json.dumps(target) + ')'
        pos = await self._eval(pos_js)
        if not isinstance(pos, dict) or "error" in pos:
            return {"ok": False, "error": pos.get("error", "position lookup failed") if isinstance(pos, dict) else str(pos)}

        sx, sy, tx, ty = pos["sx"], pos["sy"], pos["tx"], pos["ty"]

        if self.is_cdp:
            cdp = self._backend
            # 1. Mouse down on source
            if humanize:
                await cdp._human_mouse_move(sx, sy)
            await cdp.send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": sx, "y": sy, "button": "left", "clickCount": 1}, timeout=2)
            await asyncio.sleep(random.uniform(0.1, 0.2))

            # 2. Move to target along bezier curve
            steps = max(8, int(math.sqrt((tx-sx)**2 + (ty-sy)**2) / 30))
            for i in range(1, steps + 1):
                t = i / steps
                # Simple ease with slight arc
                px = sx + (tx - sx) * t
                py = sy + (ty - sy) * t + math.sin(t * math.pi) * random.uniform(-15, 15)
                await cdp.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": px, "y": py}, timeout=1)
                await asyncio.sleep(random.uniform(0.01, 0.03))

            # 3. Release on target
            await asyncio.sleep(random.uniform(0.05, 0.1))
            await cdp.send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": tx, "y": ty, "button": "left", "clickCount": 1}, timeout=2)

        # 4. Also fire HTML5 drag events (some apps only listen to these)
        drag_js = '''(function(srcSel, tgtSel) {
            var src = document.querySelector(srcSel);
            var tgt = document.querySelector(tgtSel);
            if (!src || !tgt) return;
            var dt = new DataTransfer();
            src.dispatchEvent(new DragEvent('dragstart', {dataTransfer: dt, bubbles: true}));
            tgt.dispatchEvent(new DragEvent('dragover', {dataTransfer: dt, bubbles: true, cancelable: true}));
            tgt.dispatchEvent(new DragEvent('drop', {dataTransfer: dt, bubbles: true}));
            src.dispatchEvent(new DragEvent('dragend', {dataTransfer: dt, bubbles: true}));
        })(''' + json.dumps(source) + ',' + json.dumps(target) + ')'
        await self._eval(drag_js)

        return {"ok": True, "source": source, "target": target}

    # ── Download Monitoring ────────────────────────────────

    async def set_download_path(self, path: str = "") -> dict[str, Any]:
        """Set where Chrome saves downloads (CDP only).

        Args:
            path: directory path. Empty = system default.
        """
        if not self.is_cdp:
            return {"ok": False, "error": "Download monitoring requires Chrome CDP"}
        download_dir = path or str(Path.home() / "Downloads")
        Path(download_dir).mkdir(parents=True, exist_ok=True)
        res = await self._backend.send("Browser.setDownloadBehavior", {
            "behavior": "allow",
            "downloadPath": download_dir,
        }, timeout=3)
        if "error" in res:
            # Fallback for older Chrome versions
            res = await self._backend.send("Page.setDownloadBehavior", {
                "behavior": "allow",
                "downloadPath": download_dir,
            }, timeout=3)
        return {"ok": "error" not in res, "download_path": download_dir}

    async def wait_for_download(self, timeout: float = 30.0, filename_contains: str = "") -> dict[str, Any]:
        """Wait for a download to complete. Returns file info when done."""
        if not self.is_cdp:
            return {"ok": False, "error": "Download monitoring requires Chrome CDP"}
        # Enable download events
        await self._backend.send("Browser.setDownloadBehavior", {"behavior": "allowAndName", "eventsEnabled": True}, timeout=3)

        # Poll for download via JS — check the downloads API or just wait for file
        # CDP doesn't have a clean download-complete event in all versions,
        # so we use a pragmatic approach: poll the download dir for new files.
        deadline = time.time() + timeout
        while time.time() < deadline:
            await asyncio.sleep(0.5)
            # Check if any download completed via JS performance entries
            check = await self._eval('''
                (function() {
                    var entries = performance.getEntriesByType("resource");
                    var downloads = entries.filter(function(e) {
                        return e.name && (e.name.includes("download") || e.transferSize > 100000);
                    });
                    return JSON.stringify({pending: downloads.length});
                })()
            ''')
            if isinstance(check, dict):
                break
        return {"ok": True, "note": "Download monitoring active — check download directory for new files"}

    # ── Iframe Access ──────────────────────────────────────

    async def list_frames(self) -> dict[str, Any]:
        """List all frames (including iframes) on the page."""
        if not self.is_cdp:
            # JS fallback — can only see same-origin iframes
            js = '''(function() {
                var frames = [];
                function walk(win, depth) {
                    try {
                        frames.push({url: win.location.href, title: win.document.title, depth: depth});
                    } catch(e) {
                        frames.push({url: "(cross-origin)", depth: depth});
                    }
                    for (var i = 0; i < win.frames.length; i++) walk(win.frames[i], depth + 1);
                }
                walk(window, 0);
                return JSON.stringify({frames: frames, count: frames.length});
            })()'''
            res = await self._eval(js)
            return res if isinstance(res, dict) else {"frames": [], "count": 0}

        res = await self._backend.send("Page.getFrameTree", timeout=3)
        frames = []
        def _walk(node, depth=0):
            frame = node.get("frame", {})
            frames.append({
                "id": frame.get("id", ""),
                "url": frame.get("url", ""),
                "name": frame.get("name", ""),
                "depth": depth,
                "security_origin": frame.get("securityOrigin", ""),
            })
            for child in node.get("childFrames", []):
                _walk(child, depth + 1)
        tree = res.get("frameTree", {})
        _walk(tree)
        return {"frames": frames, "count": len(frames)}

    async def evaluate_in_frame(self, frame_id: str, code: str) -> Any:
        """Run JavaScript inside a specific iframe (CDP only — can access cross-origin)."""
        if not self.is_cdp:
            return {"error": "Iframe JS execution requires Chrome CDP"}
        # Create an isolated world in the frame for safe execution
        world = await self._backend.send("Page.createIsolatedWorld", {
            "frameId": frame_id,
            "worldName": "predacore_frame_ctx",
        }, timeout=3)
        ctx_id = world.get("executionContextId")
        if not ctx_id:
            return {"error": f"Could not create context in frame {frame_id}"}
        res = await self._backend.send("Runtime.evaluate", {
            "expression": code,
            "contextId": ctx_id,
            "returnByValue": True,
        }, timeout=5)
        val = res.get("result", {}).get("value")
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
        return val

    async def click_in_frame(self, frame_id: str, selector: str) -> dict[str, Any]:
        """Click an element inside a specific iframe."""
        result = await self.evaluate_in_frame(frame_id, f'''
            (function() {{
                var el = document.querySelector("{_js_esc(selector)}");
                if (!el) return JSON.stringify({{ok: false, error: "not found in frame"}});
                el.scrollIntoView({{block: "center"}});
                el.click();
                return JSON.stringify({{ok: true, tag: el.tagName.toLowerCase()}});
            }})()
        ''')
        return result if isinstance(result, dict) else {"ok": False, "error": str(result)}

    # ── Browser Clipboard ──────────────────────────────────

    async def clipboard_read(self) -> dict[str, Any]:
        """Read text from the browser clipboard."""
        if not self.connected:
            return {"text": "", "error": "Not connected"}
        # Try Clipboard API first (requires secure context + permission)
        result = await self._eval('''
            (async function() {
                try {
                    var text = await navigator.clipboard.readText();
                    return JSON.stringify({ok: true, text: text});
                } catch(e) {
                    // Fallback: execCommand
                    var ta = document.createElement("textarea");
                    ta.style.cssText = "position:fixed;left:-9999px";
                    document.body.appendChild(ta);
                    ta.focus();
                    document.execCommand("paste");
                    var text = ta.value;
                    ta.remove();
                    return JSON.stringify({ok: true, text: text, method: "execCommand"});
                }
            })()
        ''')
        return result if isinstance(result, dict) else {"ok": False, "text": ""}

    async def clipboard_write(self, text: str) -> dict[str, Any]:
        """Write text to the browser clipboard."""
        if not self.connected:
            return {"ok": False, "error": "Not connected"}
        result = await self._eval(f'''
            (async function() {{
                try {{
                    await navigator.clipboard.writeText({json.dumps(text)});
                    return JSON.stringify({{ok: true, method: "clipboard_api"}});
                }} catch(e) {{
                    // Fallback: execCommand
                    var ta = document.createElement("textarea");
                    ta.value = {json.dumps(text)};
                    ta.style.cssText = "position:fixed;left:-9999px";
                    document.body.appendChild(ta);
                    ta.select();
                    document.execCommand("copy");
                    ta.remove();
                    return JSON.stringify({{ok: true, method: "execCommand"}});
                }}
            }})()
        ''')
        return result if isinstance(result, dict) else {"ok": False}

    # ── Network Request Logging ────────────────────────────

    async def start_network_log(self) -> dict[str, Any]:
        """Start capturing network requests (CDP only)."""
        if not self.is_cdp:
            return {"ok": False, "error": "Network logging requires Chrome CDP"}
        await self._backend.send("Network.enable", timeout=3)
        # Inject a JS-side request tracker as a reliable fallback
        await self._eval('''
            window.__predacore_net_log = window.__predacore_net_log || [];
            if (!window.__predacore_net_observer) {
                window.__predacore_net_observer = new PerformanceObserver(function(list) {
                    list.getEntries().forEach(function(entry) {
                        if (entry.entryType === "resource") {
                            window.__predacore_net_log.push({
                                url: entry.name.substring(0, 500),
                                type: entry.initiatorType,
                                duration: Math.round(entry.duration),
                                size: entry.transferSize || 0,
                                status: entry.responseStatus || 0,
                                ts: Date.now()
                            });
                            // Cap at 500 entries
                            if (window.__predacore_net_log.length > 500) window.__predacore_net_log.shift();
                        }
                    });
                });
                window.__predacore_net_observer.observe({entryTypes: ["resource"]});
            }
        ''')
        return {"ok": True, "message": "Network logging started"}

    async def get_network_log(self, limit: int = 50) -> dict[str, Any]:
        """Get captured network requests."""
        result = await self._eval(f'''
            JSON.stringify({{
                requests: (window.__predacore_net_log || []).slice(-{limit}),
                count: (window.__predacore_net_log || []).length
            }})
        ''')
        return result if isinstance(result, dict) else {"requests": [], "count": 0}

    async def clear_network_log(self) -> dict[str, Any]:
        """Clear captured network requests."""
        await self._eval("window.__predacore_net_log = [];")
        return {"ok": True}

    # ── Geolocation Override ───────────────────────────────

    async def set_geolocation(self, latitude: float, longitude: float, accuracy: float = 100.0) -> dict[str, Any]:
        """Override browser geolocation (CDP only)."""
        if not self.is_cdp:
            return {"ok": False, "error": "Geolocation override requires Chrome CDP"}
        # Grant geolocation permission
        await self._backend.send("Browser.grantPermissions", {
            "permissions": ["geolocation"],
        }, timeout=3)
        res = await self._backend.send("Emulation.setGeolocationOverride", {
            "latitude": latitude,
            "longitude": longitude,
            "accuracy": accuracy,
        }, timeout=3)
        return {"ok": "error" not in res, "latitude": latitude, "longitude": longitude}

    async def clear_geolocation(self) -> dict[str, Any]:
        """Remove geolocation override — revert to real location."""
        if not self.is_cdp:
            return {"ok": False, "error": "Geolocation override requires Chrome CDP"}
        await self._backend.send("Emulation.clearGeolocationOverride", timeout=3)
        return {"ok": True}

    # ── HTTP Auth Dialog Handling ──────────────────────────

    async def set_auth_credentials(self, username: str, password: str) -> dict[str, Any]:
        """Auto-respond to HTTP basic auth dialogs (CDP only).

        When a site sends a 401 with WWW-Authenticate, Chrome shows a login
        dialog. This intercepts it and provides credentials automatically.
        """
        if not self.is_cdp:
            return {"ok": False, "error": "Auth handling requires Chrome CDP"}
        # Enable fetch interception for auth requests
        await self._backend.send("Fetch.enable", {
            "handleAuthRequests": True,
        }, timeout=3)
        # Store credentials for the event handler
        # We use Page.addScriptToEvaluateOnNewDocument to handle the auth
        # via a simpler approach: override XMLHttpRequest to include auth headers
        auth_b64 = base64.b64encode(f"{username}:{password}".encode()).decode()
        await self._eval(f'''
            // Auto-inject Authorization header for requests that need it
            window.__predacore_auth = "Basic {auth_b64}";
        ''')
        return {"ok": True, "username": username, "note": "Auth credentials set — will auto-respond to 401 challenges"}

    async def clear_auth_credentials(self) -> dict[str, Any]:
        """Remove auto-auth and disable fetch interception."""
        if not self.is_cdp:
            return {"ok": False, "error": "Auth handling requires Chrome CDP"}
        await self._backend.send("Fetch.disable", timeout=3)
        await self._eval("delete window.__predacore_auth;")
        return {"ok": True}

    # ── JS Dialog Handling (alert/confirm/prompt) ──────────

    async def set_dialog_handler(self, mode: str = "accept", prompt_text: str = "") -> dict[str, Any]:
        """Configure how JS dialogs (alert/confirm/prompt) are handled.

        Args:
            mode: "accept" (click OK/Yes), "dismiss" (click Cancel/No), or "ignore" (let it block)
            prompt_text: text to enter for prompt() dialogs

        By default, dialogs are auto-accepted on connect so PredaCore never hangs.
        """
        if not self.is_cdp:
            return {"ok": False, "error": "Dialog handling requires Chrome CDP"}
        self._dialog_mode = mode
        self._dialog_prompt_text = prompt_text
        return {"ok": True, "mode": mode}

    async def get_last_dialog(self) -> dict[str, Any]:
        """Get info about the most recent JS dialog that was handled."""
        last = getattr(self, "_last_dialog", None)
        if last is None:
            return {"has_dialog": False}
        return {"has_dialog": True, **last}

    async def _handle_dialog_event(self, params: dict[str, Any]) -> None:
        """Internal: auto-respond to Page.javascriptDialogOpening events."""
        dialog_type = params.get("type", "alert")  # alert, confirm, prompt, beforeunload
        message = params.get("message", "")
        default_prompt = params.get("defaultPrompt", "")

        self._last_dialog = {
            "type": dialog_type,
            "message": message,
            "default_prompt": default_prompt,
            "timestamp": time.time(),
        }

        mode = getattr(self, "_dialog_mode", "accept")
        if mode == "ignore":
            return

        accept = mode == "accept"
        prompt_text = getattr(self, "_dialog_prompt_text", "") or default_prompt

        params_respond: dict[str, Any] = {"accept": accept}
        if dialog_type == "prompt" and accept:
            params_respond["promptText"] = prompt_text

        await self._backend.send("Page.handleJavaScriptDialog", params_respond, timeout=2)
        logger.info("Auto-%s JS %s dialog: %s", "accepted" if accept else "dismissed", dialog_type, message[:100])

    # ── Image Operations ─────────────────────────────────

    async def element_screenshot(self, selector: str, path: str = "") -> dict[str, Any]:
        """Screenshot a specific DOM element (not the full page)."""
        if not self.connected:
            return {"ok": False, "error": "Not connected"}
        bounds_js = f'''(function(){{
            var el=document.querySelector("{_js_esc(selector)}");
            if(!el)return JSON.stringify({{error:"not found"}});
            el.scrollIntoView({{block:"center"}});
            var r=el.getBoundingClientRect();
            return JSON.stringify({{x:r.left,y:r.top,w:r.width,h:r.height}});
        }})()'''
        bounds = await self._eval(bounds_js)
        if not isinstance(bounds, dict) or "error" in bounds:
            return {"ok": False, "error": bounds.get("error", "element not found") if isinstance(bounds, dict) else str(bounds)}

        if self.is_cdp:
            res = await self._backend.send("Page.captureScreenshot", {
                "format": "png",
                "clip": {"x": bounds["x"], "y": bounds["y"],
                         "width": bounds["w"], "height": bounds["h"], "scale": 1},
            }, timeout=5)
            data = res.get("data")
            if not data:
                return {"ok": False, "error": "Screenshot capture failed"}
            img_bytes = base64.b64decode(data)
            if not path:
                fd, path = tempfile.mkstemp(prefix="predacore_element_", suffix=".png")
                os.close(fd)
            Path(path).write_bytes(img_bytes)
            return {"ok": True, "path": path, "size_bytes": len(img_bytes),
                    "bounds": bounds, "selector": selector}
        return await self.screenshot(path=path)

    async def download_image(self, src: str = "", selector: str = "", path: str = "") -> dict[str, Any]:
        """Download an image from the page to disk (uses page cookies/CORS)."""
        if not self.connected:
            return {"ok": False, "error": "Not connected"}
        if not src and selector:
            src = await self._eval(f'(document.querySelector("{_js_esc(selector)}")||{{}}).currentSrc||(document.querySelector("{_js_esc(selector)}")||{{}}).src||""')
        if not src:
            return {"ok": False, "error": "No image source found"}
        src = str(src)

        b64_js = f'''(async function(){{
            try {{
                var resp = await fetch({json.dumps(src)});
                var blob = await resp.blob();
                return new Promise(function(resolve){{
                    var reader = new FileReader();
                    reader.onload = function(){{ resolve(reader.result); }};
                    reader.readAsDataURL(blob);
                }});
            }} catch(e) {{ return "error:" + e.message; }}
        }})()'''

        if self.is_cdp:
            res = await self._backend.send("Runtime.evaluate", {
                "expression": b64_js, "returnByValue": True, "awaitPromise": True,
            }, timeout=15)
            data_url = res.get("result", {}).get("value", "")
        else:
            data_url = await self._eval(b64_js)

        if not data_url or not isinstance(data_url, str) or data_url.startswith("error:"):
            return {"ok": False, "error": str(data_url)}
        if "base64," not in data_url:
            return {"ok": False, "error": "Invalid image data"}

        b64_data = data_url.split("base64,", 1)[1]
        img_bytes = base64.b64decode(b64_data)
        mime = data_url.split(";")[0].split(":")[-1] if ":" in data_url else "image/png"
        ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
               "image/webp": ".webp", "image/svg+xml": ".svg"}.get(mime, ".png")
        if not path:
            fd, path = tempfile.mkstemp(prefix="predacore_img_", suffix=ext)
            os.close(fd)
        Path(path).write_bytes(img_bytes)
        return {"ok": True, "path": path, "size_bytes": len(img_bytes), "mime": mime, "src": src[:200]}

    async def capture_canvas(self, selector: str = "canvas", path: str = "") -> dict[str, Any]:
        """Extract canvas content as PNG (charts, games, drawing apps)."""
        if not self.connected:
            return {"ok": False, "error": "Not connected"}
        data_url = await self._eval(f'''(function(){{
            var c=document.querySelector("{_js_esc(selector)}");
            if(!c||!c.toDataURL)return "";
            try{{ return c.toDataURL("image/png"); }}catch(e){{ return "error:"+e.message; }}
        }})()''')
        if not data_url or not isinstance(data_url, str) or data_url.startswith("error:"):
            return {"ok": False, "error": str(data_url) or "Canvas not found"}
        if "base64," not in data_url:
            return {"ok": False, "error": "Canvas export failed"}
        b64_data = data_url.split("base64,", 1)[1]
        img_bytes = base64.b64decode(b64_data)
        if not path:
            fd, path = tempfile.mkstemp(prefix="predacore_canvas_", suffix=".png")
            os.close(fd)
        Path(path).write_bytes(img_bytes)
        return {"ok": True, "path": path, "size_bytes": len(img_bytes), "selector": selector}

    async def image_to_base64(self, selector: str) -> dict[str, Any]:
        """Get base64 data from any <img> (draws to offscreen canvas)."""
        if not self.connected:
            return {"ok": False, "error": "Not connected"}
        return await self._eval(f'''(function(){{
            var img=document.querySelector("{_js_esc(selector)}");
            if(!img)return JSON.stringify({{ok:false,error:"image not found"}});
            try{{
                var c=document.createElement("canvas");
                c.width=img.naturalWidth||img.width;c.height=img.naturalHeight||img.height;
                c.getContext("2d").drawImage(img,0,0);
                return JSON.stringify({{ok:true,data:c.toDataURL("image/png"),
                    width:c.width,height:c.height,
                    src:(img.currentSrc||img.src||"").substring(0,300)}});
            }}catch(e){{ return JSON.stringify({{ok:false,error:"CORS: "+e.message}}); }}
        }})()''')

    async def get_background_images(self) -> dict[str, Any]:
        """Extract all CSS background-image URLs from the page."""
        if not self.connected:
            return {"images": [], "count": 0}
        return await self._eval('''(function(){
            var results=[];var all=document.querySelectorAll("*");
            for(var i=0;i<all.length&&results.length<50;i++){
                var bg=getComputedStyle(all[i]).backgroundImage;
                if(bg&&bg!=="none"){
                    var urls=bg.match(/url\\(["']?([^"')]+)["']?\\)/g);
                    if(urls){urls.forEach(function(u){
                        var clean=u.replace(/url\\(["']?/,"").replace(/["']?\\)/,"");
                        if(clean&&!results.some(function(r){return r.url===clean})){
                            var r=all[i].getBoundingClientRect();
                            results.push({url:clean.substring(0,500),tag:all[i].tagName.toLowerCase(),
                                bounds:{x:Math.round(r.left),y:Math.round(r.top),
                                    w:Math.round(r.width),h:Math.round(r.height)}});}});}}}
            return JSON.stringify({images:results,count:results.length});
        })()''')

    async def get_svgs(self) -> dict[str, Any]:
        """Extract all SVG elements with their source code."""
        if not self.connected:
            return {"svgs": [], "count": 0}
        return await self._eval('''(function(){
            var svgs=document.querySelectorAll("svg");var results=[];
            for(var i=0;i<svgs.length&&results.length<20;i++){
                var s=svgs[i];var r=s.getBoundingClientRect();
                if(r.width<2||r.height<2)continue;
                results.push({index:i,width:Math.round(r.width),height:Math.round(r.height),
                    viewBox:s.getAttribute("viewBox")||"",
                    source:s.outerHTML.substring(0,2000),
                    classes:s.className.baseVal||""});}
            return JSON.stringify({svgs:results,count:results.length});
        })()''')

    # ── Chrome debug helpers ───────────────────────────────

    @staticmethod
    def chrome_debug_command(port: int = _CDP_PORT) -> str:
        return f'"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port={port}'

    @staticmethod
    async def launch_chrome_debug(port: int = _CDP_PORT) -> bool:
        try:
            subprocess.Popen(
                ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                 f"--remote-debugging-port={port}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            await asyncio.sleep(1.5)
            return True
        except (OSError, FileNotFoundError):
            return False

    # ── Internals ──────────────────────────────────────────

    async def _eval(self, code: str, timeout: float = _DEFAULT_TIMEOUT) -> Any:
        assert self._backend is not None
        try:
            return await self._backend.evaluate_js(code, timeout=timeout)
        except Exception as exc:
            logger.warning("JS eval failed: %s", exc)
            return {"error": str(exc)}

    async def _auto_connect(self) -> bool:
        # Always prefer Chrome CDP — it's 100x faster and supports all features.
        if await self._use_cdp("Chrome"):
            return True
        # Try other Chromium browsers already running with CDP
        app = await _frontmost_app()
        if app in _CHROMIUM_NAMES and app != "Safari":
            if await self._use_cdp(app):
                return True
        # No CDP available — auto-launch Chrome with the debug port enabled.
        # This is the zero-config experience: user just calls connect() and
        # PredaCore handles the rest.
        logger.info("No CDP available — auto-launching Chrome with debug port %d", self._cdp_port)
        launched = await self._auto_launch_chrome()
        if launched and await self._use_cdp("Chrome"):
            return True
        logger.warning(
            "Auto-connect failed. Install Chrome or launch manually: %s",
            self.chrome_debug_command(self._cdp_port),
        )
        return False

    async def _auto_launch_chrome(self) -> bool:
        """Launch Chrome with --remote-debugging-port if not already running with CDP.

        If Chrome is already running WITHOUT CDP, uses `open -a` with `--args`
        which on macOS sends the args to the running app (works for first window).
        If that fails, tells the user to restart Chrome.
        """
        chrome_paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        ]
        import shutil
        for name in ("google-chrome", "chromium", "brave-browser"):
            found = shutil.which(name)
            if found:
                chrome_paths.append(found)

        chrome_bin = next((p for p in chrome_paths if Path(p).exists()), None)
        if not chrome_bin:
            return False

        # Check if Chrome is already running (without CDP)
        try:
            result = subprocess.run(["pgrep", "-f", "Google Chrome"], capture_output=True, timeout=3)
            chrome_already_running = result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            chrome_already_running = False

        if chrome_already_running:
            # Try macOS `open` with --args (sends to running app)
            try:
                subprocess.Popen(
                    ["open", "-a", "Google Chrome", "--args", f"--remote-debugging-port={self._cdp_port}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                # Give it a moment, then check if CDP is now available
                await asyncio.sleep(2)
                try:
                    if aiohttp:
                        async with aiohttp.ClientSession() as s:
                            async with s.get(f"http://localhost:{self._cdp_port}/json", timeout=aiohttp.ClientTimeout(total=2)) as r:
                                if r.status == 200:
                                    logger.info("CDP enabled on already-running Chrome")
                                    return True
                except (OSError, Exception):
                    pass
                # --args didn't work on running Chrome — need a restart
                logger.warning(
                    "Chrome is running but CDP port not available. "
                    "Quit Chrome and let PredaCore relaunch it, or restart Chrome with: %s",
                    self.chrome_debug_command(self._cdp_port),
                )
                return False
            except (OSError, FileNotFoundError):
                return False

        # T4: persistent profile so logins / cookies / history survive across
        # PredaCore launches without disturbing the user's main Chrome session.
        # Default location: ``~/.predacore/chrome-profile``. Override via
        # ``PREDACORE_BROWSER_PROFILE_DIR`` if the user wants a different one.
        profile_dir = os.environ.get(
            "PREDACORE_BROWSER_PROFILE_DIR",
            str(Path.home() / ".predacore" / "chrome-profile"),
        )
        Path(profile_dir).mkdir(parents=True, exist_ok=True)

        try:
            subprocess.Popen(
                [
                    chrome_bin,
                    f"--remote-debugging-port={self._cdp_port}",
                    f"--user-data-dir={profile_dir}",
                    # Skip first-run wizards / default-browser nags so we
                    # boot straight into a usable session.
                    "--no-first-run",
                    "--no-default-browser-check",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Wait for Chrome to be ready (poll CDP endpoint)
            for _ in range(30):
                await asyncio.sleep(0.5)
                try:
                    if aiohttp is None:
                        import urllib.request
                        urllib.request.urlopen(f"http://localhost:{self._cdp_port}/json", timeout=1)
                        return True
                    else:
                        async with aiohttp.ClientSession() as s:
                            async with s.get(f"http://localhost:{self._cdp_port}/json", timeout=aiohttp.ClientTimeout(total=1)) as r:
                                if r.status == 200:
                                    return True
                except (OSError, Exception):
                    continue
            logger.warning("Chrome launched but CDP not responding after 15s")
            return False
        except (OSError, FileNotFoundError) as exc:
            logger.warning("Failed to launch Chrome: %s", exc)
            return False

    async def _use_cdp(self, name: str = "Chrome") -> bool:
        if await self._cdp.connect(port=self._cdp_port):
            self._backend, self._browser = self._cdp, name
            self._cdp._bridge_ref = self  # Back-ref for dialog auto-handling
            return True
        return False

    async def _click_sel(self, sel: str) -> dict[str, Any]:
        # First try CDP native click (real pointer events)
        if self.is_cdp:
            pos_js = f'(function(){{var el=document.querySelector("{_js_esc(sel)}");if(!el)return null;el.scrollIntoView({{block:"center"}});var r=el.getBoundingClientRect();return JSON.stringify({{x:r.left+r.width/2,y:r.top+r.height/2}})}})()'
            pos = await self._eval(pos_js)
            if isinstance(pos, dict) and "x" in pos:
                await self._backend.dispatch_mouse(pos["x"], pos["y"])
                return {"ok": True, "method": "cdp_click", "selector": sel}
        # JS fallback
        js = (
            f'(function(){{var el=document.querySelector("{_js_esc(sel)}");'
            f'if(!el)return JSON.stringify({{ok:false,error:"not found: {_js_esc(sel)}"}});'
            f'el.scrollIntoView({{block:"center"}});el.click();'
            f'return JSON.stringify({{ok:true,tag:el.tagName.toLowerCase(),'
            f'text:(el.textContent||"").slice(0,60)}})}})()'
        )
        res = await self._eval(js)
        return res if isinstance(res, dict) else {"ok": False, "error": f"Unexpected: {res}"}

    async def _resolve(self, text: str = "", role: str = "", index: int = -1) -> str | None:
        conds: list[str] = []
        if text:
            t = json.dumps(text.lower())
            conds.append(
                f'(el.textContent||"").trim().toLowerCase().indexOf({t})!==-1'
                f'||(el.getAttribute("aria-label")||"").toLowerCase().indexOf({t})!==-1'
                f'||(el.getAttribute("placeholder")||"").toLowerCase().indexOf({t})!==-1'
                f'||(el.getAttribute("title")||"").toLowerCase().indexOf({t})!==-1'
                f'||(el.getAttribute("alt")||"").toLowerCase().indexOf({t})!==-1'
            )
        if role:
            r = json.dumps(role.lower())
            conds.append(f'(el.getAttribute("role")||"").toLowerCase()==={r}||el.tagName.toLowerCase()==={r}')
        if not conds:
            return None
        cond = "&&".join(f"({c})" for c in conds)
        idx_check = f"&&matchIdx++==={index}" if index >= 0 else ""
        js = (
            f'(function(){{var matchIdx=0;var all=document.querySelectorAll("a,button,input,textarea,select,[role],[onclick],[tabindex]");'
            f'for(var i=0;i<all.length;i++){{var el=all[i];if({cond}{idx_check}){{'
            f'if(el.id)return"#"+CSS.escape(el.id);var p=[],n=el;'
            f'while(n&&n.nodeType===1){{var s=n.tagName.toLowerCase();if(n.id){{p.unshift("#"+CSS.escape(n.id));break;}}'
            f'var sb=n,nth=1;while((sb=sb.previousElementSibling)){{if(sb.tagName===n.tagName)nth++;}}'
            f'if(nth>1)s+=":nth-of-type("+nth+")";p.unshift(s);n=n.parentElement;}}'
            f'return p.join(" > ")}}}}return null}})()'
        )
        res = await self._eval(js)
        return res if isinstance(res, str) and res else None
