"""Creative handlers: image_gen, pdf_reader, diagram."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from ._context import (
    SENSITIVE_READ_PATTERNS,
    SENSITIVE_WRITE_FILES,
    SENSITIVE_WRITE_PATHS,
    ToolContext,
    ToolError,
    ToolErrorKind,
    missing_param,
    invalid_param,
    blocked,
    resource_not_found,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image Generation (DALL-E 3)
# ---------------------------------------------------------------------------


async def handle_image_gen(args: dict[str, Any], ctx: ToolContext) -> str:
    """Generate images via DALL-E 3 API."""
    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        raise missing_param("prompt", tool="image_gen")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ToolError(
            "Image generation requires OPENAI_API_KEY environment variable",
            kind=ToolErrorKind.UNAVAILABLE,
            tool_name="image_gen",
            suggestion="Set OPENAI_API_KEY in your .env file",
        )

    size = str(args.get("size") or "1024x1024")
    quality = str(args.get("quality") or "standard")
    style = str(args.get("style") or "vivid")
    save_path = str(args.get("save_path") or "").strip()

    valid_sizes = {"1024x1024", "1792x1024", "1024x1792"}
    if size not in valid_sizes:
        size = "1024x1024"
    if quality not in {"standard", "hd"}:
        quality = "standard"
    if style not in {"vivid", "natural"}:
        style = "vivid"

    if save_path:
        resolved = str(Path(save_path).expanduser().resolve())
        for sp in SENSITIVE_WRITE_PATHS:
            if resolved.startswith(sp):
                raise blocked(
                    f"cannot save to sensitive path: {resolved}",
                    tool="image_gen",
                )

    try:
        import httpx

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "dall-e-3",
                    "prompt": prompt[:4000],
                    "n": 1,
                    "size": size,
                    "quality": quality,
                    "style": style,
                    "response_format": "url",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        image_url = data["data"][0]["url"]
        revised_prompt = data["data"][0].get("revised_prompt", "")

        result: dict[str, Any] = {
            "image_url": image_url,
            "revised_prompt": revised_prompt[:500],
            "size": size,
            "quality": quality,
            "style": style,
        }

        if save_path:
            async with httpx.AsyncClient(timeout=60) as client:
                img_resp = await client.get(image_url)
                img_resp.raise_for_status()
                save_resolved = Path(save_path).expanduser().resolve()
                save_resolved.parent.mkdir(parents=True, exist_ok=True)
                save_resolved.write_bytes(img_resp.content)
                result["saved_to"] = str(save_resolved)
                result["size_bytes"] = len(img_resp.content)

        return json.dumps(result, indent=2)
    except ToolError:
        raise
    except (OSError, ValueError, RuntimeError, ConnectionError, TimeoutError) as e:
        raise ToolError(
            f"Image generation failed: {e}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="image_gen",
        ) from e
    except Exception as e:
        # Catch httpx and other network errors
        if "httpx" in type(e).__module__ or "HTTP" in type(e).__name__:
            raise ToolError(
                f"Image generation network error: {e}",
                kind=ToolErrorKind.EXECUTION,
                tool_name="image_gen",
            ) from e
        raise


# ---------------------------------------------------------------------------
# PDF Reader
# ---------------------------------------------------------------------------


async def handle_pdf_reader(args: dict[str, Any], ctx: ToolContext) -> str:
    """Extract text from PDF files."""
    raw_path = str(args.get("path") or "").strip()
    if not raw_path:
        raise missing_param("path", tool="pdf_reader")

    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise resource_not_found("PDF file", raw_path, tool="pdf_reader")
    if not path.is_file():
        raise invalid_param("path", "is not a file", tool="pdf_reader")
    # Check sensitive path patterns (same as read_file)
    path_lower = str(path).lower()
    for pattern in SENSITIVE_READ_PATTERNS:
        if pattern in path_lower:
            raise blocked(
                f"refusing to read sensitive file: {path}", tool="pdf_reader"
            )

    max_pages = max(1, min(int(args.get("max_pages") or 50), 200))

    try:
        import pymupdf  # fitz
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore
        except ImportError:
            raise ToolError(
                "PDF reader requires PyMuPDF: pip install pymupdf",
                kind=ToolErrorKind.UNAVAILABLE,
                tool_name="pdf_reader",
                suggestion="pip install pymupdf",
            )

    try:
        doc = pymupdf.open(str(path))
        pages_text: list[str] = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                pages_text.append(f"...[truncated at {max_pages} pages]")
                break
            text = page.get_text()
            if text.strip():
                pages_text.append(f"--- Page {i + 1} ---\n{text}")
        doc.close()

        if not pages_text:
            return "[PDF has no extractable text (may be image-based — try OCR)]"

        output = "\n\n".join(pages_text)
        if len(output) > 100000:
            output = output[:100000] + "\n...[truncated]"
        return output
    except ToolError:
        raise
    except (OSError, ValueError, RuntimeError) as e:
        raise ToolError(
            f"PDF read error: {e}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="pdf_reader",
        ) from e


# ---------------------------------------------------------------------------
# Diagram (Mermaid)
# ---------------------------------------------------------------------------


async def handle_diagram(args: dict[str, Any], ctx: ToolContext) -> str:
    """Render a Mermaid diagram to PNG/SVG."""
    code = str(args.get("code") or "").strip()
    if not code:
        raise missing_param("code", tool="diagram")

    output_format = str(args.get("format") or "png").strip().lower()
    if output_format not in ("png", "svg", "pdf"):
        output_format = "png"
    save_path = str(args.get("save_path") or "").strip()

    is_temp = not bool(save_path)
    if save_path:
        resolved = str(Path(save_path).expanduser().resolve())
        for sp in SENSITIVE_WRITE_PATHS:
            if resolved.startswith(sp):
                raise blocked(
                    f"cannot save to sensitive path: {resolved}",
                    tool="diagram",
                )
        out_path = Path(save_path).expanduser().resolve()
    else:
        fd, tmp = tempfile.mkstemp(suffix=f".{output_format}", prefix="jarvis_diagram_")
        os.close(fd)
        out_path = Path(tmp)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write mermaid code to temp file
    fd2, mmd_path = tempfile.mkstemp(suffix=".mmd", prefix="jarvis_mermaid_")
    os.close(fd2)
    Path(mmd_path).write_text(code)

    render_ok = False
    try:
        proc = await asyncio.create_subprocess_exec(
            "mmdc", "-i", mmd_path, "-o", str(out_path),
            "-b", "transparent",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

        if proc.returncode == 0:
            render_ok = True
            size = out_path.stat().st_size if out_path.exists() else 0
            return json.dumps({
                "path": str(out_path),
                "format": output_format,
                "size_bytes": size,
            }, indent=2)
        else:
            err = (stderr or stdout or b"").decode(errors="replace").strip()
            if "not found" in err.lower() or proc.returncode == 127:
                return (
                    f"[Diagram renderer (mmdc) not found — install: npm i -g @mermaid-js/mermaid-cli]\n\n"
                    f"Mermaid code (render manually):\n```mermaid\n{code}\n```"
                )
            return f"[Diagram error: {err}]\n\nMermaid code:\n```mermaid\n{code}\n```"
    except FileNotFoundError:
        return (
            f"[Diagram renderer (mmdc) not found — install: npm i -g @mermaid-js/mermaid-cli]\n\n"
            f"Mermaid code (render manually):\n```mermaid\n{code}\n```"
        )
    except asyncio.TimeoutError:
        raise ToolError(
            "Diagram rendering timed out after 30s",
            kind=ToolErrorKind.TIMEOUT,
            tool_name="diagram",
        )
    except ToolError:
        raise
    except (OSError, RuntimeError) as e:
        return f"[Diagram error: {e}]\n\nMermaid code:\n```mermaid\n{code}\n```"
    finally:
        Path(mmd_path).unlink(missing_ok=True)
        if not render_ok and is_temp:
            Path(out_path).unlink(missing_ok=True)
