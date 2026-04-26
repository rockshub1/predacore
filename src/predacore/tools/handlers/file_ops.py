"""File operation handlers: read_file, write_file, list_directory."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ._context import (
    SENSITIVE_READ_PATTERNS,
    SENSITIVE_WRITE_FILES,
    SENSITIVE_WRITE_PATHS,
    ToolContext,
    ToolError,
    ToolErrorKind,
    blocked,
    invalid_param,
    missing_param,
    resource_not_found,
)

logger = logging.getLogger(__name__)


async def handle_read_file(args: dict[str, Any], ctx: ToolContext) -> str:
    """Read a file's contents from disk.

    Validates path existence, blocks sensitive files (.ssh, .env, etc.),
    and enforces a 2 MB size limit. Returns raw file text with
    ``errors='replace'`` for binary-safe decoding.

    Args:
        args: ``{"path": str}``
        ctx: Tool context (unused beyond validation).

    Raises:
        ToolError(MISSING_PARAM): if path is empty.
        ToolError(NOT_FOUND): if the file does not exist.
        ToolError(BLOCKED): if path matches a sensitive pattern.
        ToolError(LIMIT_EXCEEDED): if file exceeds 2 MB.
    """
    raw_path = str(args.get("path") or "").strip()
    if not raw_path:
        raise missing_param("path", tool="read_file")
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise resource_not_found("File", raw_path, tool="read_file")
    if path.is_dir():
        raise invalid_param("path", "is a directory, not a file", tool="read_file")
    path_str = str(path).lower()
    for pattern in SENSITIVE_READ_PATTERNS:
        if pattern in path_str:
            raise blocked(
                f"refusing to read sensitive file matching '{pattern}'",
                tool="read_file",
            )
    try:
        size = path.stat().st_size
    except OSError as e:
        raise ToolError(
            f"Error accessing file: {e}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="read_file",
        ) from e
    if size > 2_000_000:
        raise ToolError(
            f"File too large: {size:,} bytes — use head/tail or pdf_reader instead",
            kind=ToolErrorKind.LIMIT_EXCEEDED,
            tool_name="read_file",
            suggestion="Use `run_command` with head/tail, or `pdf_reader` for PDFs",
        )
    try:
        return path.read_text(errors="replace")
    except (OSError, UnicodeDecodeError) as e:
        raise ToolError(
            f"Error reading file: {e}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="read_file",
        ) from e


async def handle_write_file(args: dict[str, Any], ctx: ToolContext) -> str:
    """Write content to a file, creating parent directories as needed.

    Blocks writes to sensitive system paths (/etc, /var, .ssh, etc.).
    Creates intermediate directories automatically via ``mkdir(parents=True)``.

    Args:
        args: ``{"path": str, "content": str}``
        ctx: Tool context (unused beyond validation).

    Raises:
        ToolError(MISSING_PARAM): if path or content is empty.
        ToolError(BLOCKED): if path is in a sensitive system directory.
    """
    raw_path = str(args.get("path") or "").strip()
    if not raw_path:
        raise missing_param("path", tool="write_file")
    content = args.get("content")
    if content is None:
        raise missing_param("content", tool="write_file")
    path = Path(raw_path).expanduser().resolve()
    path_str = str(path).lower()
    for sensitive_dir in SENSITIVE_WRITE_PATHS:
        if path_str.startswith(sensitive_dir):
            raise blocked(
                f"refusing to write to sensitive system path: {path_str}",
                tool="write_file",
            )
    for sensitive_file in SENSITIVE_WRITE_FILES:
        if path_str.endswith(sensitive_file) or f"/{sensitive_file}" in path_str:
            raise blocked(
                f"refusing to write to sensitive file: {path_str}",
                tool="write_file",
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    # Auto-index the freshly written file into memory so future recall
    # reflects what we just wrote. Mirrors the lab Claude Code hook
    # (memory_touch.py), but ported in-process because predacore-public
    # dispatches its tools in-process. Failures are non-fatal — a memory
    # subsystem hiccup must NEVER fail the file write itself. Binary /
    # oversized / unreadable files are skipped by reindex_file's own
    # safe_read_text gate.
    if ctx.unified_memory:
        try:
            # L5 — tag the chunks with the current project_id (env →
            # git repo basename → cwd basename → "default"), resolved
            # from the file's parent dir so multi-repo agents don't
            # cross-pollute memory across projects.
            try:
                from predacore.memory.project_id import default_project
                proj = default_project(cwd=str(path.parent))
            except ImportError:
                proj = "default"
            await ctx.unified_memory.reindex_file(str(path), project_id=proj)
        except Exception as exc:  # broad: memory must not break writes
            logger.debug("Memory reindex_file(%s) skipped: %s", path, exc)
    return f"Successfully wrote {len(content)} bytes to {path}"


async def handle_list_directory(args: dict[str, Any], ctx: ToolContext) -> str:
    """List files and folders in a directory.

    Supports recursive listing via ``rglob``. Truncates at 500 entries
    to prevent overwhelming output. Prefixes entries with folder/file
    emoji for visual clarity.

    Args:
        args: ``{"path": str, "recursive": bool}``
        ctx: Tool context (unused beyond validation).

    Raises:
        ToolError(MISSING_PARAM): if path is empty.
        ToolError(NOT_FOUND): if directory does not exist.
        ToolError(INVALID_PARAM): if path is not a directory.
    """
    raw_path = str(args.get("path") or "").strip()
    if not raw_path:
        raise missing_param("path", tool="list_directory")
    path = Path(raw_path).expanduser().resolve()
    # Validate resolved path is under home or cwd to prevent path traversal
    _home = Path.home().resolve()
    _cwd = Path.cwd().resolve()
    import tempfile
    _tmpdir = str(Path(tempfile.gettempdir()).resolve())
    _allowed = (str(_home), str(_cwd), "/tmp", "/private/tmp", _tmpdir)
    if not any(str(path).startswith(p) for p in _allowed):
        raise blocked(
            "path must be under home directory, current working directory, or /tmp",
            tool="list_directory",
        )
    recursive = args.get("recursive", False)
    if not path.exists():
        raise resource_not_found("Directory", str(path), tool="list_directory")
    if not path.is_dir():
        raise invalid_param("path", "not a directory", tool="list_directory")
    try:
        entries: list[str] = []
        if recursive:
            for p in sorted(path.rglob("*")):
                rel = p.relative_to(path)
                prefix = "\U0001f4c1 " if p.is_dir() else "\U0001f4c4 "
                entries.append(f"{prefix}{rel}")
                if len(entries) > 500:
                    entries.append("...[truncated]")
                    break
        else:
            for p in sorted(path.iterdir()):
                prefix = "\U0001f4c1 " if p.is_dir() else "\U0001f4c4 "
                entries.append(f"{prefix}{p.name}")
        return "\n".join(entries) if entries else "[Empty directory]"
    except OSError as e:
        raise ToolError(
            f"Error listing directory: {e}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="list_directory",
        ) from e
