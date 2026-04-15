"""
WebChat Channel Adapter — browser-based chat UI via WebSocket.

Serves a dark-themed single-page chat application and communicates
with JARVIS over WebSocket for real-time messaging.

Setup:
  1. Enable webchat in channels.enabled list
  2. Optionally configure the port:
       channels:
         webchat:
           port: 3000
  3. Open http://localhost:3000 in your browser

Features:
  - WebSocket real-time messaging
  - Dark theme with glassmorphism design
  - Markdown rendering in chat bubbles
  - Code syntax highlighting
  - No build step — pure HTML/CSS/JS
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import WSMsgType, web

from ..config import JARVISConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

if TYPE_CHECKING:
    from ..gateway import Gateway

logger = logging.getLogger(__name__)

# Directory containing static files (index.html, style.css, app.js)
STATIC_DIR = Path(__file__).parent / "webchat_static"


class WebChatAdapter(ChannelAdapter):
    """
    WebSocket-based browser chat interface.

    Serves static files and handles WebSocket connections for
    real-time chat with JARVIS.
    """

    channel_name = "webchat"
    channel_capabilities = {
        "supports_media": True,
        "supports_buttons": True,
        "supports_markdown": True,
        "max_message_length": 100000,  # Practically unlimited
    }

    # Channel catalog: built-in channels + available integrations
    CHANNEL_CATALOG = [
        {"id": "webchat", "name": "Webchat", "builtin": True},
        {"id": "telegram", "name": "Telegram", "builtin": True},
        {"id": "discord", "name": "Discord", "builtin": True},
        {"id": "whatsapp", "name": "WhatsApp", "builtin": True},
        {"id": "slack", "name": "Slack", "builtin": False},
        {"id": "email", "name": "Email (IMAP/SMTP)", "builtin": False},
        {"id": "sms", "name": "SMS (Twilio)", "builtin": False},
        {"id": "signal", "name": "Signal", "builtin": False},
        {"id": "matrix", "name": "Matrix", "builtin": False},
        {"id": "teams", "name": "Microsoft Teams", "builtin": False},
        {"id": "line", "name": "LINE", "builtin": False},
        {"id": "messenger", "name": "Messenger", "builtin": False},
        {"id": "webhook", "name": "Custom Webhook", "builtin": False},
    ]

    def __init__(self, config: JARVISConfig):
        self.config = config
        self._message_handler = None
        wc_config = config.channels.__dict__.get("webchat", {})
        self._port = wc_config.get("port", 3000)
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._connections: dict[str, web.WebSocketResponse] = {}
        self._max_connections = 50  # Max simultaneous WebSocket connections
        self._gateway: Gateway | None = None
        self._stats_task: asyncio.Task | None = None
        self._stats_cache: dict | None = None
        self._stats_cache_time: float = 0.0

    def set_gateway(self, gateway: Gateway) -> None:
        """Receive gateway reference for dashboard stats access."""
        self._gateway = gateway

    async def start(self) -> None:
        """Start the WebChat HTTP + WebSocket server."""
        # Ensure static files exist
        if not STATIC_DIR.exists():
            STATIC_DIR.mkdir(parents=True, exist_ok=True)
            self._create_default_static_files()

        self._app = web.Application()
        self._app.router.add_get("/ws", self._websocket_handler)
        self._app.router.add_get("/api/stats", self._api_stats)
        self._app.router.add_get("/api/channels", self._api_channels)
        self._app.router.add_get("/api/identity", self._api_identity)
        self._app.router.add_get("/", self._index_handler)
        self._app.router.add_static("/static", STATIC_DIR)

        # CORS headers for localhost only
        @web.middleware
        async def cors_middleware(request, handler):
            resp = await handler(request)
            resp.headers["Access-Control-Allow-Origin"] = f"http://localhost:{self._port}"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
            return resp

        self._app.middlewares.append(cors_middleware)

        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", self._port)
        await site.start()

        # Start periodic stats broadcast to all connected clients
        self._stats_task = asyncio.create_task(self._stats_broadcast_loop())

        logger.info("✅ WebChat adapter started on http://localhost:%d", self._port)

    async def stop(self) -> None:
        """Stop the WebChat server and close all connections."""
        # Cancel stats broadcast
        if self._stats_task and not self._stats_task.done():
            self._stats_task.cancel()

        # Close all WebSocket connections (snapshot keys to avoid mutation during iteration)
        for ws in list(self._connections.values()):
            try:
                await ws.close()
            except (ConnectionError, OSError):
                pass
        self._connections.clear()

        if self._runner:
            await self._runner.cleanup()
            logger.info("WebChat adapter stopped")

    async def send(self, message: OutgoingMessage) -> None:
        """Send a message to a connected WebSocket client."""
        ws = self._connections.get(message.user_id)
        if ws and not ws.closed:
            await ws.send_json(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": message.text,
                    "timestamp": time.time(),
                }
            )
        else:
            logger.warning("WebChat: user %s not connected", message.user_id)

    async def _index_handler(self, request: web.Request) -> web.Response:
        """Serve the main chat page."""
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return web.FileResponse(index_path)
        return web.Response(
            text="<h1>JARVIS WebChat</h1><p>Static files not found.</p>",
            content_type="text/html",
        )

    async def _websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections for real-time chat."""
        # Reject new connections if at capacity
        if len(self._connections) >= self._max_connections:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.close(code=1013, message=b"Server at capacity")
            return ws

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Reuse identity from query param if the client has one stored,
        # otherwise generate a fresh connection ID.  The gateway's
        # IdentityService will resolve this to a canonical user later —
        # using a stable connection_id lets the same browser tab keep
        # the same canonical user across reconnects.
        connection_id = request.query.get("cid", "").strip()
        # Validate: must be a valid UUID format to prevent injection
        try:
            if connection_id:
                uuid.UUID(connection_id)
        except ValueError:
            connection_id = ""
        if not connection_id:
            connection_id = str(uuid.uuid4())

        # If another tab is already using this connection_id, close the old one
        old_ws = self._connections.get(connection_id)
        if old_ws and not old_ws.closed:
            try:
                await old_ws.close(code=1000, message=b"Replaced by new connection")
            except (ConnectionError, OSError):
                pass

        self._connections[connection_id] = ws
        session_id = connection_id

        logger.info("WebChat client connected: %s", session_id)

        # Send welcome message with connection_id so frontend can persist it
        await ws.send_json(
            {
                "type": "system",
                "content": "Connected to JARVIS. How can I help?",
                "session_id": session_id,
                "connection_id": connection_id,
                "timestamp": time.time(),
            }
        )

        _MAX_WS_MESSAGE_BYTES = 65536  # 64KB max message size

        # Build event/stream closures once per connection (not per message)
        async def _ws_event_fn(event_type, data):
            if ws.closed:
                return
            try:
                if event_type == "thinking":
                    await ws.send_json({"type": "thinking"})
                elif event_type == "tool_start":
                    await ws.send_json({"type": "tool_start", "name": data.get("name", "")})
                elif event_type == "tool_end":
                    await ws.send_json({
                        "type": "tool_end",
                        "name": data.get("name", ""),
                        "duration_ms": data.get("duration_ms", 0),
                        "result_preview": data.get("result", "")[:100],
                    })
                elif event_type == "response":
                    await ws.send_json({
                        "type": "response_stats",
                        "tokens": data.get("tokens", 0),
                        "tools_used": data.get("tools_used", 0),
                        "elapsed_s": data.get("elapsed_s", 0),
                        "provider": data.get("provider", ""),
                    })
            except (ConnectionError, OSError):
                pass

        async def _ws_stream_fn(chunk: str):
            if ws.closed:
                return
            try:
                await ws.send_json({"type": "stream", "content": chunk})
            except (ConnectionError, OSError):
                pass

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    if len(msg.data) > _MAX_WS_MESSAGE_BYTES:
                        await ws.send_json({
                            "type": "error",
                            "content": f"Message too large ({len(msg.data)} bytes). Max: {_MAX_WS_MESSAGE_BYTES}.",
                        })
                        continue

                    try:
                        data = json.loads(msg.data)

                        if not isinstance(data, dict):
                            await ws.send_json(
                                {"type": "error", "content": "Invalid message format: expected JSON object"}
                            )
                            continue

                        content = str(data.get("content") or "").strip()
                        content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)

                        if content and self._message_handler:
                            await ws.send_json({"type": "typing", "active": True})

                            outgoing = await self._message_handler(
                                IncomingMessage(
                                    channel=self.channel_name,
                                    user_id=session_id,
                                    text=content,
                                    metadata={"session_id": session_id},
                                ),
                                event_fn=_ws_event_fn,
                                stream_fn=_ws_stream_fn,
                            )

                            await ws.send_json({"type": "typing", "active": False})

                            if outgoing:
                                await ws.send_json({
                                    "type": "message",
                                    "role": "assistant",
                                    "content": outgoing.text,
                                    "timestamp": time.time(),
                                })

                    except json.JSONDecodeError:
                        await ws.send_json({"type": "error", "content": "Invalid message format"})

                elif msg.type == WSMsgType.ERROR:
                    logger.error("WebSocket error: %s", ws.exception())

        finally:
            self._connections.pop(session_id, None)
            logger.info("WebChat client disconnected: %s", session_id)

        return ws

    # ── Dashboard API + Stats Push ─────────────────────────────────────

    def _build_dashboard_stats(self) -> dict:
        """Gather lightweight stats from all subsystems for the dashboard."""
        now = time.time()
        # Cache for 2 seconds to avoid hammering get_stats()
        if self._stats_cache and (now - self._stats_cache_time) < 2.0:
            return self._stats_cache

        stats: dict = {
            "uptime_s": 0,
            "messages": {"in": 0, "out": 0, "errors": 0},
            "brain": None,
            "memory": None,
            "identity": None,
            "channels": {"connected": 0, "list": []},
            "provider": "--",
            "model": "--",
        }

        if not self._gateway:
            return stats

        # Gateway stats
        try:
            gw = self._gateway.get_stats()
            stats["uptime_s"] = int(gw.get("uptime_seconds", 0))
            stats["messages"]["in"] = gw.get("messages_received", 0)
            stats["messages"]["out"] = gw.get("messages_sent", 0)
            stats["messages"]["errors"] = gw.get("errors", 0)
            active = gw.get("active_channels", [])
            stats["channels"]["connected"] = len(active)
            stats["channels"]["list"] = active
            stats["channels"]["health"] = gw.get("channel_health", {})
        except Exception:
            pass

        # Core / LLM stats
        core = getattr(self._gateway, "_core", None) or getattr(self._gateway, "core", None)
        if core:
            llm = getattr(core, "llm", None)
            if llm:
                stats["provider"] = getattr(llm, "active_provider", "--")
                stats["model"] = getattr(llm, "active_model", "--")

        # Memory stats
        if core:
            mem = getattr(core, "memory", None)
            if mem:
                try:
                    count = getattr(mem, "count", None)
                    if callable(count):
                        stats["memory"] = {"total": count()}
                    elif hasattr(mem, "_db"):
                        db = getattr(mem, "_db", None)
                        if db:
                            cur = db.execute("SELECT COUNT(*) FROM memories")
                            stats["memory"] = {"total": cur.fetchone()[0]}
                except Exception:
                    pass

        # Identity stats
        try:
            from ..identity.engine import get_identity_engine  # noqa: deferred to avoid circular import
            engine = get_identity_engine(self.config)
            if engine.seed_exists:
                gs = engine.get_growth_stats()
                journal_entries = 0
                jf = gs.get("files", {}).get("JOURNAL.md", {})
                journal_entries = jf.get("entry_count", 0)
                # Calculate identity age from workspace creation
                identity_file = gs.get("files", {}).get("IDENTITY.md", {})
                age_days = 0
                if identity_file.get("last_modified"):
                    try:
                        mod = datetime.fromisoformat(identity_file["last_modified"])
                        age_days = max(0, (datetime.now() - mod).days)
                    except (ValueError, TypeError):
                        pass
                stats["identity"] = {
                    "bootstrapped": gs.get("bootstrapped", False),
                    "age_days": age_days,
                    "journal_entries": journal_entries,
                }
        except Exception:
            pass

        self._stats_cache = stats
        self._stats_cache_time = now
        return stats

    async def _stats_broadcast_loop(self) -> None:
        """Push live stats to all connected WebSocket clients every 5 seconds."""
        while True:
            try:
                await asyncio.sleep(5)
                if not self._connections:
                    continue
                stats = self._build_dashboard_stats()
                msg = {"type": "stats_update", "data": stats}
                for ws in list(self._connections.values()):
                    if not ws.closed:
                        try:
                            await ws.send_json(msg)
                        except (ConnectionError, OSError):
                            pass
            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("Stats broadcast error", exc_info=True)

    async def _api_stats(self, request: web.Request) -> web.Response:
        """GET /api/stats — full dashboard stats."""
        return web.json_response(self._build_dashboard_stats())

    async def _api_channels(self, request: web.Request) -> web.Response:
        """GET /api/channels — channel status + catalog."""
        active = []
        health = {}
        if self._gateway:
            gw = self._gateway.get_stats()
            active = gw.get("active_channels", [])
            health = gw.get("channel_health", {})

        return web.json_response({
            "active": active,
            "catalog": self.CHANNEL_CATALOG,
            "health": health,
        })

    async def _api_identity(self, request: web.Request) -> web.Response:
        """GET /api/identity — identity engine stats."""
        try:
            from ..identity.engine import get_identity_engine  # noqa: deferred
            engine = get_identity_engine(self.config)
            return web.json_response(engine.get_growth_stats())
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    def _create_default_static_files(self) -> None:
        """Create default static files if they don't exist.

        The actual UI files are maintained in the webchat_static/ directory.
        This method only creates minimal fallback files if the directory was wiped.
        """
        index = STATIC_DIR / "index.html"
        if not index.exists():
            index.write_text(DEFAULT_INDEX_HTML)

        style = STATIC_DIR / "style.css"
        if not style.exists():
            style.write_text(DEFAULT_STYLE_CSS)

        app_js = STATIC_DIR / "app.js"
        if not app_js.exists():
            app_js.write_text(DEFAULT_APP_JS)

        # Create empty stubs for module files so <script> tags don't 404
        for fname in ("sounds.js", "widgets.js", "commands.js"):
            p = STATIC_DIR / fname
            if not p.exists():
                p.write_text(f"// {fname} stub — see webchat_static/ for full version\n")


# ── Default Static Files ─────────────────────────────────────────────

DEFAULT_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JARVIS — Project Prometheus</title>
    <meta name="description" content="JARVIS AI Assistant — Personal AI powered by Project Prometheus">
    <link rel="stylesheet" href="/static/style.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/dompurify@3/dist/purify.min.js"></script>
</head>
<body>
    <div class="app">
        <header class="header">
            <div class="header-left">
                <div class="logo">
                    <span class="logo-icon">⚡</span>
                    <span class="logo-text">JARVIS</span>
                </div>
                <span class="header-subtitle">Project Prometheus</span>
            </div>
            <div class="header-right">
                <div class="status-dot" id="statusDot"></div>
                <span class="status-text" id="statusText">Connecting...</span>
            </div>
        </header>

        <main class="chat-container" id="chatContainer">
            <div class="chat-messages" id="chatMessages"></div>
        </main>

        <footer class="input-area">
            <div class="input-wrapper">
                <textarea
                    id="messageInput"
                    placeholder="Message JARVIS..."
                    rows="1"
                    autofocus
                ></textarea>
                <button id="sendButton" class="send-btn" disabled>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                    </svg>
                </button>
            </div>
        </footer>
    </div>

    <script src="/static/app.js"></script>
</body>
</html>"""

DEFAULT_STYLE_CSS = """/* JARVIS WebChat — Dark Glassmorphism Theme */
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-glass: rgba(18, 18, 26, 0.8);
    --bg-glass-hover: rgba(25, 25, 38, 0.9);
    --border-glass: rgba(255, 255, 255, 0.06);
    --border-accent: rgba(99, 102, 241, 0.3);
    --text-primary: #e4e4e7;
    --text-secondary: #a1a1aa;
    --text-dim: #71717a;
    --accent: #818cf8;
    --accent-glow: rgba(129, 140, 248, 0.15);
    --user-bubble: rgba(99, 102, 241, 0.15);
    --user-border: rgba(99, 102, 241, 0.25);
    --assistant-bubble: rgba(255, 255, 255, 0.03);
    --assistant-border: rgba(255, 255, 255, 0.06);
    --success: #34d399;
    --error: #f87171;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    --radius: 16px;
    --radius-sm: 10px;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

html, body {
    height: 100%;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: var(--font-sans);
    font-size: 14px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

.app {
    display: flex;
    flex-direction: column;
    height: 100vh;
    max-width: 900px;
    margin: 0 auto;
}

/* ── Header ── */
.header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    background: var(--bg-glass);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border-glass);
    position: sticky;
    top: 0;
    z-index: 10;
}

.header-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.logo {
    display: flex;
    align-items: center;
    gap: 8px;
}

.logo-icon {
    font-size: 20px;
    animation: pulse 2s infinite;
}

.logo-text {
    font-weight: 700;
    font-size: 18px;
    letter-spacing: 0.5px;
    background: linear-gradient(135deg, var(--accent), #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header-subtitle {
    font-size: 12px;
    color: var(--text-dim);
    font-weight: 400;
}

.header-right {
    display: flex;
    align-items: center;
    gap: 8px;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-dim);
    transition: background 0.3s;
}

.status-dot.connected { background: var(--success); box-shadow: 0 0 8px var(--success); }
.status-dot.error { background: var(--error); }

.status-text {
    font-size: 12px;
    color: var(--text-dim);
}

/* ── Chat Container ── */
.chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    scroll-behavior: smooth;
}

.chat-messages {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

/* ── Message Bubbles ── */
.message {
    display: flex;
    gap: 12px;
    max-width: 85%;
    animation: fadeIn 0.3s ease;
}

.message.user {
    align-self: flex-end;
    flex-direction: row-reverse;
}

.message.assistant {
    align-self: flex-start;
}

.message-avatar {
    width: 32px;
    height: 32px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
}

.message.user .message-avatar {
    background: var(--user-bubble);
    border: 1px solid var(--user-border);
}

.message.assistant .message-avatar {
    background: var(--accent-glow);
    border: 1px solid var(--border-accent);
}

.message-content {
    padding: 12px 16px;
    border-radius: var(--radius);
    font-size: 14px;
    line-height: 1.7;
    word-wrap: break-word;
}

.message.user .message-content {
    background: var(--user-bubble);
    border: 1px solid var(--user-border);
    border-bottom-right-radius: 4px;
}

.message.assistant .message-content {
    background: var(--assistant-bubble);
    border: 1px solid var(--assistant-border);
    border-bottom-left-radius: 4px;
}

.message-content pre {
    background: rgba(0, 0, 0, 0.4);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-sm);
    padding: 12px 16px;
    overflow-x: auto;
    margin: 8px 0;
    font-family: var(--font-mono);
    font-size: 13px;
}

.message-content code {
    font-family: var(--font-mono);
    font-size: 13px;
    background: rgba(0, 0, 0, 0.3);
    padding: 2px 6px;
    border-radius: 4px;
}

.message-content pre code {
    background: none;
    padding: 0;
}

/* ── Typing Indicator ── */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 12px 16px;
}

.typing-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-dim);
    animation: typingBounce 1.4s infinite;
}

.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

/* ── Input Area ── */
.input-area {
    padding: 16px 24px 24px;
    background: var(--bg-glass);
    backdrop-filter: blur(20px);
    border-top: 1px solid var(--border-glass);
}

.input-wrapper {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius);
    padding: 8px 12px;
    transition: border-color 0.2s;
}

.input-wrapper:focus-within {
    border-color: var(--border-accent);
    box-shadow: 0 0 0 3px var(--accent-glow);
}

textarea {
    flex: 1;
    background: none;
    border: none;
    color: var(--text-primary);
    font-family: var(--font-sans);
    font-size: 14px;
    line-height: 1.5;
    resize: none;
    outline: none;
    max-height: 150px;
    padding: 4px 0;
}

textarea::placeholder {
    color: var(--text-dim);
}

.send-btn {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    border: none;
    background: var(--accent);
    color: white;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
    flex-shrink: 0;
}

.send-btn:hover:not(:disabled) {
    transform: scale(1.05);
    box-shadow: 0 0 12px var(--accent-glow);
}

.send-btn:disabled {
    opacity: 0.3;
    cursor: not-allowed;
}

/* ── Animations ── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

@keyframes typingBounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-6px); }
}

/* ── System Messages ── */
.system-message {
    text-align: center;
    color: var(--text-dim);
    font-size: 12px;
    padding: 8px 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-glass); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.12); }

/* ── Responsive ── */
@media (max-width: 640px) {
    .header { padding: 12px 16px; }
    .chat-container { padding: 16px; }
    .input-area { padding: 12px 16px 16px; }
    .message { max-width: 92%; }
}"""

DEFAULT_APP_JS = """// JARVIS WebChat — WebSocket Client
(function() {
    'use strict';

    const chatMessages = document.getElementById('chatMessages');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');

    let ws = null;
    let sessionId = null;
    let reconnectAttempts = 0;
    const MAX_RECONNECT = 5;
    let currentStreamDiv = null;

    // ── WebSocket Connection ──
    function connect() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${protocol}//${location.host}/ws`;

        ws = new WebSocket(url);

        ws.onopen = () => {
            setStatus('connected', 'Connected');
            sendButton.disabled = false;
            reconnectAttempts = 0;
        };

        ws.onclose = () => {
            setStatus('error', 'Disconnected');
            sendButton.disabled = true;
            attemptReconnect();
        };

        ws.onerror = () => {
            setStatus('error', 'Error');
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleMessage(data);
            } catch (e) {
                console.error('Failed to parse message:', e);
            }
        };
    }

    function attemptReconnect() {
        if (reconnectAttempts >= MAX_RECONNECT) {
            setStatus('error', 'Failed to connect');
            return;
        }
        reconnectAttempts++;
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000);
        setStatus('error', `Reconnecting in ${delay/1000}s...`);
        setTimeout(connect, delay);
    }

    function setStatus(state, text) {
        statusDot.className = 'status-dot ' + state;
        statusText.textContent = text;
    }

    // ── Message Handling ──
    function handleMessage(data) {
        switch (data.type) {
            case 'system':
                sessionId = data.session_id;
                addSystemMessage(data.content);
                break;
            case 'stream':
                removeTyping();
                addStreamMessage(data.content);
                break;
            case 'message':
                removeTyping();
                currentStreamDiv = null; // Reset stream tracker
                addMessage(data.role, data.content);
                break;
            case 'typing':
                if (data.active) showTyping();
                else removeTyping();
                break;
            case 'error':
                addSystemMessage('Error: ' + data.content);
                break;
        }
    }

    function addStreamMessage(content) {
        if (!currentStreamDiv) {
            currentStreamDiv = document.createElement('div');
            currentStreamDiv.className = `message assistant`;

            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = '⚡';

            const bubble = document.createElement('div');
            bubble.className = 'message-content stream-content';
            
            currentStreamDiv.appendChild(avatar);
            currentStreamDiv.appendChild(bubble);
            chatMessages.appendChild(currentStreamDiv);
        }
        
        const bubble = currentStreamDiv.querySelector('.stream-content');
        // We accumulate raw text here until the final 'message' comes
        // Note: real-time markdown parsing is tricky with partials,
        // so we'll just append text and let the final 'message' render properly.
        bubble.textContent += content;
        scrollToBottom();
    }

    function addMessage(role, content) {
        // If an assistant message follows a stream, we replace the stream div
        if (role === 'assistant' && currentStreamDiv) {
            currentStreamDiv.remove();
            currentStreamDiv = null;
        }
        const div = document.createElement('div');
        div.className = `message ${role}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? '👤' : '⚡';

        const bubble = document.createElement('div');
        bubble.className = 'message-content';

        // Render markdown for assistant messages
        if (role === 'assistant') {
            bubble.innerHTML = DOMPurify.sanitize(marked.parse(content));
            // Highlight code blocks
            bubble.querySelectorAll('pre code').forEach(block => {
                hljs.highlightElement(block);
            });
        } else {
            bubble.textContent = content;
        }

        div.appendChild(avatar);
        div.appendChild(bubble);
        chatMessages.appendChild(div);
        scrollToBottom();
    }

    function addSystemMessage(content) {
        const div = document.createElement('div');
        div.className = 'system-message';
        div.textContent = content;
        chatMessages.appendChild(div);
        scrollToBottom();
    }

    function showTyping() {
        if (document.getElementById('typingIndicator')) return;
        const div = document.createElement('div');
        div.id = 'typingIndicator';
        div.className = 'message assistant';
        div.innerHTML = `
            <div class="message-avatar">⚡</div>
            <div class="message-content typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        chatMessages.appendChild(div);
        scrollToBottom();
    }

    function removeTyping() {
        const el = document.getElementById('typingIndicator');
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // ── Send Message ──
    function sendMessage() {
        const text = messageInput.value.trim();
        if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

        addMessage('user', text);
        ws.send(JSON.stringify({ content: text }));

        messageInput.value = '';
        messageInput.style.height = 'auto';
    }

    // ── Event Listeners ──
    sendButton.addEventListener('click', sendMessage);

    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 150) + 'px';
        sendButton.disabled = !messageInput.value.trim();
    });

    // ── Configure marked ──
    marked.setOptions({
        breaks: true,
        gfm: true,
        highlight: function(code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                return hljs.highlight(code, { language: lang }).value;
            }
            return hljs.highlightAuto(code).value;
        }
    });

    // ── Start ──
    connect();
})();"""
