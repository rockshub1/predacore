"""Desktop & mobile handlers: desktop_control, screen_vision, android_control."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ._context import (
    ToolContext,
    ToolError,
    ToolErrorKind,
    missing_param,
    subsystem_unavailable,
)

logger = logging.getLogger(__name__)

# Trust-tiered timeout. yolo gets the original "no limits" 300s for long
# automation flows; ask_everytime caps at 60s so a misbehaving action can't
# hold the macOS Accessibility framework for 5 minutes and stall everything
# else. Operators that need a longer hold can still pass `timeout` in args
# (the handler honors it, capped by the trust ceiling).
_DESKTOP_TIMEOUT_BY_TRUST: dict[str, float] = {
    "yolo": 300.0,
    "ask_everytime": 60.0,
}
_DESKTOP_TIMEOUT_DEFAULT = 60.0


async def handle_desktop_control(args: dict[str, Any], ctx: ToolContext) -> str:
    """Control local desktop actions through the macOS operator."""
    action = str(args.get("action") or "").strip()
    if not action:
        raise missing_param("action", tool="desktop_control")
    if ctx.desktop_operator is None:
        raise subsystem_unavailable("Desktop control", tool="desktop_control")

    trust = getattr(ctx.config.security, "trust_level", "ask_everytime")
    timeout = _DESKTOP_TIMEOUT_BY_TRUST.get(trust, _DESKTOP_TIMEOUT_DEFAULT)

    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, ctx.desktop_operator.execute, action, args),
            timeout=timeout,
        )
        out = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        return out[:50000] if len(out) > 50000 else out
    except asyncio.TimeoutError as exc:
        raise ToolError(
            f"Desktop control timed out after {timeout:.0f}s (trust={trust}) — "
            f"action '{action}' took too long",
            kind=ToolErrorKind.TIMEOUT,
            tool_name="desktop_control",
            suggestion="Try a simpler action, switch to yolo trust, or increase timeout",
        ) from exc
    except ToolError:
        raise  # Re-raise our own errors
    except (RuntimeError, OSError, ValueError) as exc:
        raise ToolError(
            f"Desktop control error: {exc}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="desktop_control",
        ) from exc


async def handle_screen_vision(args: dict[str, Any], ctx: ToolContext) -> str:
    """Realtime screen understanding — AX-first with OCR and vision fallback."""
    action = str(args.get("action") or "").strip()
    if not action:
        raise missing_param("action", tool="screen_vision")
    if ctx.desktop_operator is None:
        raise subsystem_unavailable(
            "Desktop control (required by screen vision)", tool="screen_vision"
        )

    from predacore.operators.screen_vision import ScreenVisionEngine

    engine = getattr(ctx, "_screen_vision_engine", None)
    if engine is None:
        llm = getattr(ctx, "llm_for_collab", None)
        ops_cfg = getattr(ctx.config, "operators", None) if ctx.config else None
        engine = ScreenVisionEngine(ctx.desktop_operator, llm_interface=llm, operators_config=ops_cfg)
        ctx._screen_vision_engine = engine  # type: ignore[attr-defined]

    try:
        if action == "quick_scan":
            state = await engine.quick_scan()
            return json.dumps({
                "summary": state.summary(),
                "elements": [e.to_dict() for e in state.elements],
                "scan_ms": state.scan_ms,
            }, default=str, indent=2)

        elif action == "scan":
            include_ss = bool(args.get("include_screenshot", False))
            state = await engine.scan(include_screenshot=include_ss)
            result: dict[str, Any] = {
                "summary": state.summary(),
                "elements": [e.to_dict() for e in state.elements],
                "scan_ms": state.scan_ms,
            }
            if state.screenshot_path:
                result["screenshot_path"] = state.screenshot_path
            return json.dumps(result, default=str, indent=2)

        elif action == "scan_with_vision":
            task = str(args.get("task", "Describe what you see on screen"))
            state, description = await engine.scan_with_vision(task=task)
            return json.dumps({
                "summary": state.summary(),
                "vision_analysis": description,
                "scan_ms": state.scan_ms,
            }, default=str, indent=2)

        elif action == "scan_with_ocr":
            result = await engine.scan_with_ocr()
            return json.dumps(result, default=str, indent=2)

        elif action == "find_and_click":
            label = str(args.get("label", ""))
            role = str(args.get("role", ""))
            if not label:
                raise missing_param("label", tool="screen_vision.find_and_click")
            result = await engine.find_and_click(label, role=role)
            return json.dumps(result, default=str, indent=2)

        elif action == "type_into":
            label = str(args.get("label", ""))
            text = str(args.get("text", ""))
            if not label:
                raise missing_param("label", tool="screen_vision.type_into")
            if not text:
                raise missing_param("text", tool="screen_vision.type_into")
            result = await engine.type_into(label, text)
            return json.dumps(result, default=str, indent=2)

        elif action == "read_screen_text":
            use_ocr = bool(args.get("use_ocr_fallback", True))
            texts = await engine.read_screen_text(use_ocr_fallback=use_ocr)
            return json.dumps({"texts": texts, "count": len(texts)}, indent=2)

        elif action == "has_changed":
            changed = await engine.has_changed()
            return json.dumps({"changed": changed})

        elif action == "wait_for_change":
            timeout = float(args.get("timeout", 30))
            state = await engine.wait_for_change(timeout=timeout)
            if state:
                return json.dumps({
                    "changed": True, "summary": state.summary(),
                }, default=str, indent=2)
            return json.dumps({"changed": False, "timeout": True})

        elif action == "execute_task":
            task = str(args.get("task", ""))
            if not task:
                raise missing_param("task", tool="screen_vision.execute_task")
            max_steps = max(1, min(int(args.get("max_steps", 20)), 50))
            log = await engine.execute_task(task, max_steps=max_steps)
            return json.dumps({"action_log": log, "steps": len(log)}, default=str, indent=2)

        elif action == "focused_element":
            el = await engine.focused_element()
            if el is None:
                return json.dumps({"focused": False, "element": None})
            return json.dumps({"focused": True, "element": el.to_dict()}, default=str, indent=2)

        elif action == "ocr_status":
            ocr = engine._get_ocr_engine()
            if ocr is not None:
                return json.dumps({"available": True, **ocr.status()}, indent=2)
            return json.dumps({"available": False, "reason": "No OCR backend (install PyObjC or tesseract)"})

        else:
            raise ToolError(
                f"Unknown screen_vision action: {action}",
                kind=ToolErrorKind.INVALID_PARAM,
                tool_name="screen_vision",
                suggestion="Valid actions: quick_scan, scan, scan_with_vision, scan_with_ocr, "
                           "find_and_click, type_into, read_screen_text, has_changed, "
                           "wait_for_change, execute_task, focused_element, ocr_status",
            )

    except ToolError:
        raise  # Re-raise structured errors as-is
    except (RuntimeError, OSError, ValueError, AttributeError) as exc:
        logger.warning("Screen vision failed (attempt 1): %s — retrying", exc)
        try:
            await asyncio.sleep(0.1)
            llm = getattr(ctx, "llm_for_collab", None)
            ops_cfg = getattr(ctx.config, "operators", None) if ctx.config else None
            engine = ScreenVisionEngine(ctx.desktop_operator, llm_interface=llm, operators_config=ops_cfg)
            ctx._screen_vision_engine = engine  # type: ignore[attr-defined]

            if action == "quick_scan":
                state = await engine.quick_scan()
                return json.dumps({
                    "summary": state.summary(),
                    "elements": [e.to_dict() for e in state.elements],
                    "scan_ms": state.scan_ms, "retried": True,
                }, default=str, indent=2)
            elif action in ("scan", "scan_with_vision"):
                state = await engine.scan()
                return json.dumps({
                    "summary": state.summary(),
                    "elements": [e.to_dict() for e in state.elements],
                    "scan_ms": state.scan_ms, "retried": True,
                }, default=str, indent=2)
            elif action == "scan_with_ocr":
                result = await engine.scan_with_ocr()
                result["retried"] = True
                return json.dumps(result, default=str, indent=2)
            else:
                raise ToolError(
                    f"Screen vision error (after retry): {exc}",
                    kind=ToolErrorKind.EXECUTION,
                    tool_name="screen_vision",
                ) from exc
        except ToolError:
            raise
        except (RuntimeError, OSError, ValueError, AttributeError) as retry_exc:
            raise ToolError(
                f"Screen vision error (after retry): {retry_exc}",
                kind=ToolErrorKind.EXECUTION,
                tool_name="screen_vision",
            ) from retry_exc


async def handle_android_control(args: dict[str, Any], ctx: ToolContext) -> str:
    """Android device control via ADB."""
    action = str(args.get("action") or "").strip()
    if not action:
        raise missing_param("action", tool="android_control")

    from predacore.operators.android import ADBError, AndroidOperator

    serial = str(args.get("device_serial", "")).strip() or None
    cache_key = f"_android_operator_{serial or 'default'}"
    operator = getattr(ctx, cache_key, None)
    if operator is None:
        operator = AndroidOperator(device_serial=serial)
        setattr(ctx, cache_key, operator)

    try:
        params = dict(args)
        params.pop("action", None)
        params.pop("device_serial", None)
        result = operator.execute(action=action, params=params)
        out = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        return out[:50000] if len(out) > 50000 else out
    except ADBError as exc:
        raise ToolError(
            f"Android control error: {exc}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="android_control",
        ) from exc
    except ToolError:
        raise
    except (RuntimeError, OSError, ValueError) as exc:
        raise ToolError(
            f"Android control error: {exc}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="android_control",
        ) from exc


# ── Browser Bridge ──────────────────────────────────────────────────

# Singleton bridge instance (persists connection across tool calls)
_browser_bridge = None


async def handle_browser_control(args: dict[str, Any], ctx: ToolContext) -> str:
    """Control YOUR running Chrome browser via instant DOM access."""
    global _browser_bridge

    action = str(args.get("action") or "").strip()
    if not action:
        raise missing_param("action", tool="browser_control")

    try:
        from predacore.operators.browser_bridge import BrowserBridge
    except ImportError as e:
        raise subsystem_unavailable(f"Browser Bridge: {e}") from e

    # Auto-connect on first use (Chrome only — Safari support is disabled)
    if _browser_bridge is None or not _browser_bridge.connected:
        _browser_bridge = BrowserBridge()
        connected = await _browser_bridge.connect("chrome")
        if not connected:
            return json.dumps({"error": "Could not connect to Chrome. Is it running with remote debugging enabled?"})

    bridge = _browser_bridge

    try:
        if action == "connect":
            browser = str(args.get("browser", "auto"))
            _browser_bridge = BrowserBridge()
            ok = await _browser_bridge.connect(browser)
            result = {"connected": ok, "browser": _browser_bridge.browser_name}

        elif action == "get_page_tree":
            result = await bridge.get_page_tree()

        elif action == "click":
            result = await bridge.click(
                selector=str(args.get("selector", "")),
                text=str(args.get("text", "")),
                role=str(args.get("role", "")),
            )

        elif action == "type":
            result = await bridge.type_text(
                selector=str(args.get("selector", "")),
                text=str(args.get("text", "")),
                value=str(args.get("value", "")),
            )

        elif action == "type_keys":
            result = await bridge.type_keys(
                selector=str(args.get("selector", "")),
                text=str(args.get("text", "")),
                value=str(args.get("value", "")),
                clear=bool(args.get("clear", True)),
            )

        elif action == "read_text":
            text = await bridge.read_text(selector=str(args.get("selector", "")))
            result = {"text": text[:30000] if isinstance(text, str) else str(text)}

        elif action == "navigate":
            url = str(args.get("url", ""))
            if not url:
                raise missing_param("url", tool="browser_control")
            result = await bridge.navigate(url)

        elif action == "scroll":
            result = await bridge.scroll(
                direction=str(args.get("direction", "down")),
                amount=int(args.get("amount", 3)),
            )

        elif action == "evaluate_js":
            code = str(args.get("code", ""))
            if not code:
                raise missing_param("code", tool="browser_control")
            raw = await bridge.evaluate_js(code)
            result = {"result": raw}

        elif action == "get_url":
            url = await bridge.get_url()
            result = {"url": url}

        # Media controls
        elif action == "get_media_info":
            result = await bridge.get_media_info()
        elif action == "media_play":
            result = await bridge.media_play()
        elif action == "media_pause":
            result = await bridge.media_pause()
        elif action == "media_seek":
            result = await bridge.media_seek(float(args.get("seconds", 0)))
        elif action == "media_set_speed":
            result = await bridge.media_set_speed(float(args.get("speed", 1.0)))
        elif action == "media_set_volume":
            result = await bridge.media_set_volume(int(args.get("level", 50)))
        elif action == "media_toggle_mute":
            result = await bridge.media_toggle_mute()
        elif action == "media_fullscreen":
            result = await bridge.media_fullscreen()

        # Captions
        elif action == "get_captions":
            result = await bridge.get_captions()
        elif action == "enable_captions":
            result = await bridge.enable_captions()
        elif action == "get_transcript":
            result = await bridge.get_transcript()

        # Page content
        elif action == "get_page_images":
            result = await bridge.get_page_images()
        elif action == "get_page_links":
            result = await bridge.get_page_links()

        # Wait primitives
        elif action == "wait_for_element":
            sel = str(args.get("selector", ""))
            if not sel:
                raise missing_param("selector", tool="browser_control")
            result = await bridge.wait_for_element(sel, timeout=float(args.get("timeout", 10)))
        elif action == "wait_for_text":
            txt = str(args.get("text", ""))
            if not txt:
                raise missing_param("text", tool="browser_control")
            result = await bridge.wait_for_text(txt, timeout=float(args.get("timeout", 10)))
        elif action == "wait_for_url":
            pattern = str(args.get("pattern", ""))
            if not pattern:
                raise missing_param("pattern", tool="browser_control")
            result = await bridge.wait_for_url(pattern, timeout=float(args.get("timeout", 10)))

        # History
        elif action == "back":
            result = await bridge.back()
        elif action == "forward":
            result = await bridge.forward()
        elif action == "reload":
            result = await bridge.reload()

        # Hover & keyboard
        elif action == "hover":
            result = await bridge.hover(
                selector=str(args.get("selector", "")),
                text=str(args.get("text", "")),
            )
        elif action == "press_key":
            k = str(args.get("key", ""))
            if not k:
                raise missing_param("key", tool="browser_control")
            result = await bridge.press_key(k, modifiers=args.get("modifiers"))
        elif action == "key_combo":
            keys = args.get("keys", [])
            if not keys:
                raise missing_param("keys", tool="browser_control")
            result = await bridge.key_combo(keys)

        # Cookies
        elif action == "get_cookies":
            result = await bridge.get_cookies(domain=str(args.get("domain", "")))
        elif action == "set_cookie":
            result = await bridge.set_cookie(
                name=str(args.get("name", "")),
                value=str(args.get("value", "")),
                domain=str(args.get("domain", "")),
            )
        elif action == "delete_cookies":
            result = await bridge.delete_cookies(
                name=str(args.get("name", "")),
                domain=str(args.get("domain", "")),
            )

        # Storage
        elif action == "get_storage":
            result = await bridge.get_storage(
                key=str(args.get("key", "")),
                storage=str(args.get("storage_type", "local")),
            )
        elif action == "set_storage":
            result = await bridge.set_storage(
                key=str(args.get("key", "")),
                value=str(args.get("value", "")),
                storage=str(args.get("storage_type", "local")),
            )
        elif action == "clear_storage":
            result = await bridge.clear_storage(storage=str(args.get("storage_type", "local")))

        # Forms
        elif action == "set_checkbox":
            sel = str(args.get("selector", ""))
            if not sel:
                raise missing_param("selector", tool="browser_control")
            result = await bridge.set_checkbox(sel, checked=bool(args.get("checked", True)))
        elif action == "select_option":
            sel = str(args.get("selector", ""))
            if not sel:
                raise missing_param("selector", tool="browser_control")
            result = await bridge.select_option(
                sel, value=str(args.get("value", "")), label=str(args.get("label", "")),
            )
        elif action == "upload_file":
            sel = str(args.get("selector", ""))
            paths = args.get("file_paths", [])
            if not sel:
                raise missing_param("selector", tool="browser_control")
            result = await bridge.upload_file(sel, paths)

        # Screenshot & PDF
        elif action == "screenshot":
            result = await bridge.screenshot(
                path=str(args.get("path", "")),
                full_page=bool(args.get("full_page", False)),
            )
        elif action == "print_pdf":
            result = await bridge.print_pdf(path=str(args.get("path", "")))

        # Tabs
        elif action == "list_tabs":
            result = await bridge.list_tabs()
        elif action == "new_tab":
            result = await bridge.new_tab(url=str(args.get("url", "about:blank")))
        elif action == "close_tab":
            result = await bridge.close_tab(target_id=str(args.get("target_id", "")))

        # Table extraction & find
        elif action == "extract_tables":
            result = await bridge.extract_tables(selector=str(args.get("selector", "table")))
        elif action == "find_in_page":
            q = str(args.get("query", ""))
            if not q:
                raise missing_param("query", tool="browser_control")
            result = await bridge.find_in_page(q)

        # Drag and drop
        elif action == "drag_and_drop":
            src = str(args.get("source", ""))
            tgt = str(args.get("target_selector", args.get("target", "")))
            if not src or not tgt:
                raise missing_param("source and target_selector", tool="browser_control")
            result = await bridge.drag_and_drop(src, tgt)

        # Downloads
        elif action == "set_download_path":
            result = await bridge.set_download_path(path=str(args.get("path", "")))

        # Iframes
        elif action == "list_frames":
            result = await bridge.list_frames()
        elif action == "evaluate_in_frame":
            fid = str(args.get("frame_id", ""))
            code = str(args.get("code", ""))
            if not fid or not code:
                raise missing_param("frame_id and code", tool="browser_control")
            raw = await bridge.evaluate_in_frame(fid, code)
            result = {"result": raw}
        elif action == "click_in_frame":
            fid = str(args.get("frame_id", ""))
            sel = str(args.get("selector", ""))
            if not fid or not sel:
                raise missing_param("frame_id and selector", tool="browser_control")
            result = await bridge.click_in_frame(fid, sel)

        # Clipboard
        elif action == "clipboard_read":
            result = await bridge.clipboard_read()
        elif action == "clipboard_write":
            txt = str(args.get("text", args.get("value", "")))
            if not txt:
                raise missing_param("text", tool="browser_control")
            result = await bridge.clipboard_write(txt)

        # Network logging
        elif action == "start_network_log":
            result = await bridge.start_network_log()
        elif action == "get_network_log":
            result = await bridge.get_network_log(limit=int(args.get("limit", 50)))
        elif action == "clear_network_log":
            result = await bridge.clear_network_log()

        # Geolocation
        elif action == "set_geolocation":
            result = await bridge.set_geolocation(
                latitude=float(args.get("latitude", 0)),
                longitude=float(args.get("longitude", 0)),
                accuracy=float(args.get("accuracy", 100)),
            )
        elif action == "clear_geolocation":
            result = await bridge.clear_geolocation()

        # Images
        elif action == "element_screenshot":
            sel = str(args.get("selector", ""))
            if not sel:
                raise missing_param("selector", tool="browser_control")
            result = await bridge.element_screenshot(sel, path=str(args.get("path", "")))
        elif action == "download_image":
            result = await bridge.download_image(
                src=str(args.get("src", "")),
                selector=str(args.get("selector", "")),
                path=str(args.get("path", "")),
            )
        elif action == "capture_canvas":
            result = await bridge.capture_canvas(
                selector=str(args.get("selector", "canvas")),
                path=str(args.get("path", "")),
            )
        elif action == "image_to_base64":
            sel = str(args.get("selector", ""))
            if not sel:
                raise missing_param("selector", tool="browser_control")
            result = await bridge.image_to_base64(sel)
        elif action == "get_background_images":
            result = await bridge.get_background_images()
        elif action == "get_svgs":
            result = await bridge.get_svgs()

        # JS Dialogs
        elif action == "set_dialog_handler":
            result = await bridge.set_dialog_handler(
                mode=str(args.get("mode", "accept")),
                prompt_text=str(args.get("prompt_text", "")),
            )
        elif action == "get_last_dialog":
            result = await bridge.get_last_dialog()

        # Auth
        elif action == "set_auth_credentials":
            result = await bridge.set_auth_credentials(
                username=str(args.get("username", "")),
                password=str(args.get("password", "")),
            )
        elif action == "clear_auth_credentials":
            result = await bridge.clear_auth_credentials()

        else:
            result = {"error": f"Unknown action: {action}"}

        out = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        return out[:50000] if len(out) > 50000 else out

    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(
            f"Browser control error: {exc}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="browser_control",
        ) from exc
