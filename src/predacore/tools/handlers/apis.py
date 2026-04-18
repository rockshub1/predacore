"""
API registry — a lightweight equivalent of ``channel_configure`` for
arbitrary REST services the user wants the agent to consume.

``api_add`` registers a named service (base URL + auth header template +
default headers). Later ``api_call service=<name> method=GET path=/users``
fires a real HTTP request. Secrets go to ``~/.predacore/.env`` via the
usual ``${VAR}`` pattern — no plaintext tokens in config.

This is intentionally simpler than MCP: no subprocess, no tool discovery,
no protocol — just "PredaCore now knows how to hit $service." Think
``httpie`` aliases the agent can manage conversationally.

Storage:
    ~/.predacore/apis.yaml         (structure below)
    ~/.predacore/.env              (referenced secrets)

Example ``apis.yaml``::

    apis:
      notion:
        base_url: https://api.notion.com/v1
        auth: "Bearer ${NOTION_TOKEN}"
        default_headers:
          "Notion-Version": "2022-06-28"
      linear:
        base_url: https://api.linear.app/graphql
        auth: "${LINEAR_API_KEY}"
        default_headers:
          "Content-Type": "application/json"
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import httpx

from ._context import (
    ToolContext,
    ToolError,
    ToolErrorKind,
    invalid_param,
    missing_param,
)

logger = logging.getLogger(__name__)

_APIS_FILENAME = "apis.yaml"

# URL / header sanity — guard against prompt-injection-induced garbage.
_SCHEME_RE = re.compile(r"^https?://", re.IGNORECASE)
_SERVICE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
_HEADER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

_DEFAULT_TIMEOUT_S = 30.0
_MAX_RESPONSE_BYTES = 250_000  # don't flood the LLM with huge payloads
_MAX_RESPONSE_PREVIEW = 8_000  # preview returned to the LLM


# ── api_add ──────────────────────────────────────────────────────────


async def handle_api_add(args: dict[str, Any], ctx: ToolContext) -> str:
    """Register a REST API so the agent can hit it via ``api_call``.

    Args:
        service: short name (``"notion"``, ``"linear"``). Used as the
            lookup key in ``api_call service=...``.
        base_url: scheme + host + optional path prefix (e.g.
            ``https://api.notion.com/v1``). Required.
        auth: optional value inserted verbatim into the
            ``Authorization`` header. Use ``${VAR}`` to reference a
            secret stored via ``secret_set``. If you need a non-standard
            auth header name, use ``default_headers`` instead.
        default_headers: optional dict of extra headers sent on every
            call. ``${VAR}`` expansion applies to values.
        description: optional human-readable note shown by ``api_list``.
    """
    service = (args.get("service") or "").strip().lower()
    if not service:
        raise missing_param("service", tool="api_add")
    if not _SERVICE_NAME_RE.match(service):
        raise invalid_param(
            "service",
            f"must be lowercase alphanumeric + _/-, got {service!r}",
            tool="api_add",
        )

    base_url = (args.get("base_url") or "").strip()
    if not base_url:
        raise missing_param("base_url", tool="api_add")
    if not _SCHEME_RE.match(base_url):
        raise invalid_param(
            "base_url",
            f"must start with http:// or https://, got {base_url!r}",
            tool="api_add",
        )

    entry: dict[str, Any] = {
        "base_url": base_url.rstrip("/"),
    }
    if args.get("auth"):
        entry["auth"] = str(args["auth"])

    default_headers = args.get("default_headers") or {}
    if default_headers:
        if not isinstance(default_headers, dict):
            raise invalid_param(
                "default_headers", "must be a dict", tool="api_add",
            )
        for h in default_headers:
            if not _HEADER_NAME_RE.match(str(h)):
                raise invalid_param(
                    "default_headers",
                    f"invalid header name: {h!r}",
                    tool="api_add",
                )
        entry["default_headers"] = {str(k): str(v) for k, v in default_headers.items()}

    if args.get("description"):
        entry["description"] = str(args["description"])

    home = _home(ctx)
    data = _load_apis(home)
    apis = data.setdefault("apis", {})
    apis[service] = entry
    _save_apis(home, data)

    return json.dumps({
        "status": "ok",
        "service": service,
        "base_url": entry["base_url"],
        "note": (
            f"Call with: api_call service={service} method=GET path=/some/path"
        ),
    }, indent=2)


# ── api_remove ───────────────────────────────────────────────────────


async def handle_api_remove(args: dict[str, Any], ctx: ToolContext) -> str:
    service = (args.get("service") or "").strip().lower()
    if not service:
        raise missing_param("service", tool="api_remove")

    home = _home(ctx)
    data = _load_apis(home)
    apis = data.get("apis") or {}
    if service not in apis:
        raise ToolError(
            f"No API registered as {service!r}.",
            kind=ToolErrorKind.NOT_FOUND,
            tool_name="api_remove",
        )
    apis.pop(service, None)
    data["apis"] = apis
    _save_apis(home, data)
    return json.dumps({"status": "ok", "service": service}, indent=2)


# ── api_list ─────────────────────────────────────────────────────────


async def handle_api_list(args: dict[str, Any], ctx: ToolContext) -> str:
    home = _home(ctx)
    data = _load_apis(home)
    apis = data.get("apis") or {}
    out = []
    for name, spec in sorted(apis.items()):
        out.append({
            "service": name,
            "base_url": spec.get("base_url", ""),
            "has_auth": bool(spec.get("auth")),
            "description": spec.get("description", ""),
            "default_headers": list((spec.get("default_headers") or {}).keys()),
        })
    return json.dumps({"status": "ok", "count": len(out), "apis": out}, indent=2)


# ── api_call ─────────────────────────────────────────────────────────


async def handle_api_call(args: dict[str, Any], ctx: ToolContext) -> str:
    """Invoke a registered API. Returns status + headers + body preview."""
    service = (args.get("service") or "").strip().lower()
    if not service:
        raise missing_param("service", tool="api_call")
    method = (args.get("method") or "GET").strip().upper()
    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
        raise invalid_param("method", f"unsupported method {method!r}", tool="api_call")
    path = str(args.get("path") or "")
    if not path.startswith("/"):
        path = "/" + path

    home = _home(ctx)
    data = _load_apis(home)
    spec = (data.get("apis") or {}).get(service)
    if spec is None:
        raise ToolError(
            f"No API registered as {service!r}. Use api_add first.",
            kind=ToolErrorKind.NOT_FOUND,
            tool_name="api_call",
            suggestion="api_add service=... base_url=...",
        )

    base_url = str(spec.get("base_url", "")).rstrip("/")
    url = base_url + path

    headers: dict[str, str] = {}
    # default_headers from config
    for k, v in (spec.get("default_headers") or {}).items():
        headers[str(k)] = os.path.expandvars(str(v))
    # auth → Authorization header (unless the user overrode via default_headers)
    if spec.get("auth") and "Authorization" not in headers:
        headers["Authorization"] = os.path.expandvars(str(spec["auth"]))
    # Per-call overrides
    for k, v in (args.get("headers") or {}).items():
        headers[str(k)] = str(v)

    params = args.get("params") or None
    body = args.get("body")
    json_body = args.get("json")
    if json_body is not None and body is not None:
        raise invalid_param(
            "body",
            "pass either `body` (raw string) OR `json` (structured), not both",
            tool="api_call",
        )

    timeout = float(args.get("timeout") or _DEFAULT_TIMEOUT_S)
    trust = getattr(ctx.config.security, "trust_level", "normal")
    if trust == "paranoid" and method != "GET":
        raise ToolError(
            f"api_call with method={method} is blocked in paranoid trust mode. "
            "Only GET is allowed without confirmation.",
            kind=ToolErrorKind.BLOCKED,
            tool_name="api_call",
        )

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.request(
                method, url,
                params=params,
                headers=headers,
                content=body if body is not None else None,
                json=json_body if json_body is not None else None,
            )
    except httpx.HTTPError as exc:
        raise ToolError(
            f"HTTP error calling {service} {method} {path}: {exc}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="api_call",
        ) from exc

    raw = resp.content[:_MAX_RESPONSE_BYTES]
    try:
        body_text = raw.decode("utf-8")
    except UnicodeDecodeError:
        body_text = f"<{len(raw)} bytes of binary>"
    truncated = len(resp.content) > len(raw)

    preview = body_text[:_MAX_RESPONSE_PREVIEW]
    if len(body_text) > _MAX_RESPONSE_PREVIEW:
        preview += f"\n...[truncated — {len(body_text) - _MAX_RESPONSE_PREVIEW} more chars]"

    return json.dumps({
        "status": resp.status_code,
        "ok": 200 <= resp.status_code < 300,
        "url": str(resp.url),
        "content_type": resp.headers.get("content-type", ""),
        "body": preview,
        "truncated": truncated,
    }, indent=2)


# ── Helpers ──────────────────────────────────────────────────────────


def _home(ctx: ToolContext) -> Path:
    return Path(
        getattr(ctx.config, "home_dir", None) or (Path.home() / ".predacore")
    ).expanduser()


def _load_apis(home: Path) -> dict[str, Any]:
    import yaml  # core dep
    path = home / _APIS_FILENAME
    if not path.exists():
        return {"apis": {}}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            return {"apis": {}}
        loaded.setdefault("apis", {})
        return loaded
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse %s: %s", path, exc)
        return {"apis": {}}


def _save_apis(home: Path, data: dict[str, Any]) -> None:
    import yaml
    home.mkdir(parents=True, exist_ok=True)
    path = home / _APIS_FILENAME
    fd, tmp = tempfile.mkstemp(dir=str(home), suffix=".yaml.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
