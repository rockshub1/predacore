"""Localhost callback listener for the OAuth redirect.

The OAuth flow tells the provider to redirect the user's browser back
to ``http://localhost:PORT/callback?code=...&state=...``. We run a
minimal aiohttp server on that port that:

  1. Captures the ``code`` query param (and validates ``state``)
  2. Returns a short HTML "you can close this tab now" page
  3. Resolves an ``asyncio.Future`` so the caller in ``flow.py`` can
     proceed to the token exchange

The server lives only for the lifetime of one OAuth attempt — started,
waits for the callback, then shut down. We pick a fresh ephemeral port
each time so two concurrent ``predacore login`` runs (rare but possible
in tests) don't collide.
"""
from __future__ import annotations

import asyncio
import logging
import socket
from dataclasses import dataclass
from typing import Any

from aiohttp import web

logger = logging.getLogger(__name__)


_SUCCESS_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>PredaCore — Authorized</title>
<style>
body { font-family:-apple-system,system-ui,sans-serif;
       background:#0a0a0a;color:#e6e6e6;
       display:flex;flex-direction:column;align-items:center;
       justify-content:center;min-height:100vh;margin:0; }
.card { background:#161616;border:1px solid #2a2a2a;border-radius:12px;
        padding:32px 40px;max-width:480px;text-align:center; }
.tick { font-size:48px;color:#4ade80;line-height:1; }
h1 { font-size:18px;margin:16px 0 6px 0;font-weight:600; }
p  { font-size:14px;margin:0;color:#a0a0a0;line-height:1.5; }
</style></head><body>
<div class="card">
  <div class="tick">✓</div>
  <h1>PredaCore is authorized.</h1>
  <p>You can close this tab. The CLI will continue automatically.</p>
</div></body></html>"""


_FAILURE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>PredaCore — Auth failed</title></head>
<body><pre style="font-family:monospace;padding:32px">
PredaCore login failed.

{detail}

You can close this tab and re-run `predacore login`.
</pre></body></html>"""


@dataclass
class CallbackResult:
    """What the listener captured from the redirect."""
    code: str = ""
    state: str = ""
    error: str = ""
    error_description: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.code) and not self.error


def _pick_free_port() -> int:
    """Find a free localhost TCP port — bind to 0 and read what the OS gave us."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def wait_for_authorization_code(
    *,
    expected_state: str,
    port: int = 0,
    path: str = "/callback",
    timeout_seconds: float = 300.0,
    bind_host: str = "127.0.0.1",
) -> CallbackResult:
    """Run the listener and return as soon as the redirect arrives.

    Args:
      expected_state: The opaque CSRF token we sent in the auth URL.
        If the redirect comes back with a different ``state``, treat
        it as an attacker-driven request and reject.
      port: 0 (default) picks an ephemeral free port. Pass an explicit
        port only when the OAuth client is registered with a fixed
        ``redirect_uri`` requirement — Codex's PKCE-public client lets
        us use any localhost port, so 0 is the right default.
      path: Request path the provider will redirect to. Default
        ``/callback`` matches what we register in the auth URL.
      timeout_seconds: Hard cap so a failed redirect doesn't leave the
        listener running forever. 5 minutes is generous for human
        click latency.

    Returns:
      ``CallbackResult`` — check ``.ok`` and ``.error`` to dispatch.
    """
    if port == 0:
        port = _pick_free_port()
    loop = asyncio.get_running_loop()
    future: asyncio.Future[CallbackResult] = loop.create_future()

    async def _handler(request: web.Request) -> web.Response:
        # Ignore favicon + anything else; only the registered path matters.
        if request.path != path:
            return web.Response(status=404, text="not found")

        params = request.rel_url.query
        result = CallbackResult(
            code=params.get("code", ""),
            state=params.get("state", ""),
            error=params.get("error", ""),
            error_description=params.get("error_description", ""),
        )

        if result.error:
            detail = f"{result.error}: {result.error_description}".strip(": ")
            if not future.done():
                future.set_result(result)
            return web.Response(
                body=_FAILURE_HTML.format(detail=_html_escape(detail)),
                content_type="text/html",
            )

        if result.state != expected_state:
            # State mismatch — refuse the code, do NOT resolve the future
            # successfully. The caller will time out and retry / fail
            # cleanly. This blocks CSRF-style cross-site callbacks.
            logger.warning(
                "oauth callback: state mismatch (got %r, expected %r) — rejecting",
                result.state[:16], expected_state[:16],
            )
            if not future.done():
                future.set_result(
                    CallbackResult(
                        error="state_mismatch",
                        error_description="callback state did not match request",
                    )
                )
            return web.Response(
                status=400,
                body=_FAILURE_HTML.format(detail="state mismatch"),
                content_type="text/html",
            )

        if not future.done():
            future.set_result(result)
        return web.Response(body=_SUCCESS_HTML, content_type="text/html")

    app = web.Application()
    app.router.add_get(path, _handler)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, bind_host, port)
    await site.start()
    logger.info("oauth callback listener: http://%s:%d%s", bind_host, port, path)

    try:
        return await asyncio.wait_for(future, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return CallbackResult(
            error="timeout",
            error_description=f"no callback received within {timeout_seconds:.0f}s",
        )
    finally:
        await runner.cleanup()


# Tiny re-implementation; we don't want to pull html.escape's dependency
# graph for this one small hot path. ``html.escape`` from stdlib is a
# couple hundred bytes — actually fine, switching:
def _html_escape(s: str) -> str:
    import html as _html
    return _html.escape(s, quote=True)


# Public so callers (flow.py + tests) can build the redirect_uri string
# that matches whatever port we end up listening on.
def build_redirect_uri(port: int, path: str = "/callback") -> str:
    return f"http://127.0.0.1:{port}{path}"
