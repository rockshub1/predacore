"""
Semantic chunker for the Predacore Memory MCP.

Goal: never split a function/class/heading in half. BGE embeddings of
partial functions are meaningless, and citations that point into the
middle of a split unit lie to the reader.

Strategy by file type:

  .py                 — AST walk. One chunk per top-level def/class. Any
                        module-level code between top-level defs becomes
                        its own chunk ("module").
  .md, .markdown      — split on heading lines (^#, ^##, ...). Each heading
                        + its body = one chunk.
  .rs, .js, .ts, .tsx — regex-based brace-aware splitter (not perfect but
                        close enough for a v1; Python-level parsers would
                        bring heavier deps).
  other               — line-window fallback: 400-line windows with 40-line
                        overlap. Keeps chunks small enough for BGE's
                        256-token window AND preserves some overlap context.

Every chunk carries enough context that a reader could read only the
chunk and understand what it is. Trivially tiny chunks (< MIN_CHUNK_CHARS)
are merged into their neighbor.
"""
from __future__ import annotations

import ast
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────

# A chunk below this size gets merged into its neighbor. Chunks this
# small almost never carry enough context to be searchable on their own.
MIN_CHUNK_CHARS = 120

# Upper bound per chunk. Larger chunks get split into windows. This also
# protects BGE-small (256-token window ≈ ~1000 chars) from wasting context.
MAX_CHUNK_CHARS = 4000

# Fallback window size for unknown file types. 400 lines is comfortable
# for most code; 40 line overlap means a concept straddling a window
# boundary lands in both halves.
FALLBACK_WINDOW_LINES = 400
FALLBACK_OVERLAP_LINES = 40

# File extensions we know how to semantically chunk.
_PY_EXT = frozenset({".py", ".pyi"})
_MD_EXT = frozenset({".md", ".markdown", ".mdx"})
_BRACE_EXT = frozenset({".rs", ".js", ".jsx", ".ts", ".tsx", ".go", ".c",
                        ".h", ".cpp", ".hpp", ".java", ".kt", ".swift"})


# ─────────────────────────────────────────────────────────────
# Chunk dataclass
# ─────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """One piece of a file, ready to be embedded + stored."""
    content: str
    line_start: int           # 1-indexed inclusive
    line_end: int             # 1-indexed inclusive
    ordinal: int              # position within file (0-based)
    kind: str                 # "function" | "class" | "module" | "heading" | "block" | "window"
    anchor: str = ""          # short human-readable anchor (func name, heading text)

    def __post_init__(self) -> None:
        # anchor fallback — first non-blank line, truncated
        if not self.anchor:
            for line in self.content.splitlines():
                s = line.strip()
                if s:
                    self.anchor = s[:80]
                    break

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────


def chunk_text(path: str | Path, content: str) -> list[Chunk]:
    """Route to the right chunker for this file type.

    `path` determines strategy (by extension); `content` is what gets
    chunked. A caller can override routing by passing a fake path — e.g.
    pass 'x.py' to force Python AST chunking on any text.
    """
    p = Path(str(path))
    suffix = p.suffix.lower()

    try:
        if suffix in _PY_EXT:
            chunks = _chunk_python(content)
        elif suffix in _MD_EXT:
            chunks = _chunk_markdown(content)
        elif suffix in _BRACE_EXT:
            chunks = _chunk_brace(content)
        else:
            chunks = _chunk_windows(content)
    except Exception as exc:
        # Any chunker failure → fall back to window chunking. We never
        # want a chunker bug to block indexing entirely.
        logger.warning(
            "Chunker %s failed on %s (%s) — falling back to windows",
            suffix or "?", path, exc,
        )
        chunks = _chunk_windows(content)

    chunks = _merge_tiny(chunks)
    chunks = _split_oversized(chunks)
    # Re-number ordinals after merge/split rewrites the list
    for i, c in enumerate(chunks):
        c.ordinal = i
    return chunks


# ─────────────────────────────────────────────────────────────
# Python — AST walk
# ─────────────────────────────────────────────────────────────


def _chunk_python(content: str) -> list[Chunk]:
    """One chunk per top-level def/class plus interstitial 'module' chunks.

    We walk only top-level statements so nested classes/functions stay
    inside their enclosing chunk (keeps semantically-coupled code together).
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        # Broken Python file — fall back to windows rather than skip.
        return _chunk_windows(content)

    lines = content.splitlines(keepends=True)
    n_lines = len(lines)
    if n_lines == 0:
        return []

    spans: list[tuple[int, int, str, str]] = []  # (line_start, line_end, kind, anchor)

    for node in tree.body:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is None or end is None:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append((start, end, "function", f"def {node.name}"))
        elif isinstance(node, ast.ClassDef):
            spans.append((start, end, "class", f"class {node.name}"))
        # Everything else (imports, assignments, expressions, try, etc.)
        # gets swept into the surrounding "module" chunk below.

    # Build chunks, filling gaps between known spans with "module" chunks.
    chunks: list[Chunk] = []
    cursor = 1  # 1-indexed line cursor
    spans.sort(key=lambda s: s[0])
    for start, end, kind, anchor in spans:
        if start > cursor:
            gap = "".join(lines[cursor - 1:start - 1])
            if gap.strip():
                chunks.append(Chunk(
                    content=gap.rstrip("\n") + "\n",
                    line_start=cursor,
                    line_end=start - 1,
                    ordinal=-1,
                    kind="module",
                ))
        body = "".join(lines[start - 1:end])
        chunks.append(Chunk(
            content=body,
            line_start=start,
            line_end=end,
            ordinal=-1,
            kind=kind,
            anchor=anchor,
        ))
        cursor = end + 1

    # Trailing module-level content after the last span
    if cursor <= n_lines:
        tail = "".join(lines[cursor - 1:n_lines])
        if tail.strip():
            chunks.append(Chunk(
                content=tail,
                line_start=cursor,
                line_end=n_lines,
                ordinal=-1,
                kind="module",
            ))

    return chunks


# ─────────────────────────────────────────────────────────────
# Markdown — heading split
# ─────────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _chunk_markdown(content: str) -> list[Chunk]:
    """Each heading (any level) + its body until the next heading."""
    if not content.strip():
        return []

    lines = content.splitlines(keepends=True)
    heading_positions: list[tuple[int, str]] = []  # (line_index_1based, heading_text)

    for i, line in enumerate(lines, start=1):
        m = _HEADING_RE.match(line)
        if m:
            heading_positions.append((i, line.strip()))

    if not heading_positions:
        # No headings — treat the whole doc as one chunk.
        return [Chunk(
            content=content,
            line_start=1,
            line_end=len(lines),
            ordinal=0,
            kind="block",
        )]

    chunks: list[Chunk] = []
    # Optional preface before the first heading
    first_line = heading_positions[0][0]
    if first_line > 1:
        preface = "".join(lines[0:first_line - 1])
        if preface.strip():
            chunks.append(Chunk(
                content=preface,
                line_start=1,
                line_end=first_line - 1,
                ordinal=-1,
                kind="block",
            ))

    for idx, (start, heading) in enumerate(heading_positions):
        end = heading_positions[idx + 1][0] - 1 if idx + 1 < len(heading_positions) else len(lines)
        body = "".join(lines[start - 1:end])
        chunks.append(Chunk(
            content=body,
            line_start=start,
            line_end=end,
            ordinal=-1,
            kind="heading",
            anchor=heading,
        ))
    return chunks


# ─────────────────────────────────────────────────────────────
# Brace-based — Rust/JS/TS/Go/C/etc.
# ─────────────────────────────────────────────────────────────


# Matches top-level function/class/struct/impl/trait declarations across
# our supported brace languages. Intentionally loose — we just need a
# reliable "chunk boundary" signal, not precise parsing.
# Only matches top-level declarations — no leading whitespace before the
# (optional) visibility modifier. This avoids nesting pub fn inside impl,
# which would produce overlapping/contained chunks.
_BRACE_DECL_RE = re.compile(
    r"""
    ^
    (?:
        (?:pub(?:\s*\([^)]*\))?\s+)?            # Rust visibility
        (?:async\s+)?
        (?:unsafe\s+)?
        (?:export\s+)?                          # TS export
        (?:default\s+)?                         # TS default export
        (?:static\s+|public\s+|private\s+|protected\s+)?
        (?:
            fn\s+\w+                           |  # Rust fn
            impl(?:\s+[\w:<>,\s]+)?             |  # Rust impl
            trait\s+\w+                         |  # Rust trait
            struct\s+\w+                        |  # Rust/Go struct
            enum\s+\w+                          |  # Rust enum
            class\s+\w+                         |  # JS/TS/Java/Kotlin class
            interface\s+\w+                     |  # TS/Go interface
            function\s*\*?\s*\w+                |  # JS function
            func\s+\w+                          |  # Go func
            type\s+\w+\s+struct                 |  # Go type ... struct
            (?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?(?:function|\()  # JS arrow/function expr
        )
    )
    """,
    re.VERBOSE | re.MULTILINE,
)


def _chunk_brace(content: str) -> list[Chunk]:
    """
    Brace-balanced splitting. We find declaration lines via regex, then
    walk forward counting braces to find the matching close. Respects
    string/char/comment contexts so braces inside strings don't confuse us.
    """
    if not content.strip():
        return []
    lines = content.splitlines(keepends=True)
    n = len(lines)

    decls: list[int] = []  # line numbers (1-indexed) where a declaration starts
    for m in _BRACE_DECL_RE.finditer(content):
        # Count newlines before m.start() to find the line number
        line_no = content.count("\n", 0, m.start()) + 1
        decls.append(line_no)

    if not decls:
        return _chunk_windows(content)

    chunks: list[Chunk] = []

    # Preface before first declaration
    first_decl = decls[0]
    if first_decl > 1:
        preface = "".join(lines[0:first_decl - 1])
        if preface.strip():
            chunks.append(Chunk(
                content=preface,
                line_start=1,
                line_end=first_decl - 1,
                ordinal=-1,
                kind="module",
            ))

    for idx, decl_line in enumerate(decls):
        end_line = _find_block_end(lines, decl_line - 1)
        if end_line is None:
            # No matching brace — take up to the next declaration or EOF
            end_line = decls[idx + 1] - 1 if idx + 1 < len(decls) else n
        body = "".join(lines[decl_line - 1:end_line])
        anchor_line = lines[decl_line - 1].strip()
        chunks.append(Chunk(
            content=body,
            line_start=decl_line,
            line_end=end_line,
            ordinal=-1,
            kind="block",
            anchor=anchor_line[:80],
        ))

    # Any trailing content after the last chunk's end
    last_end = max(c.line_end for c in chunks) if chunks else 0
    if last_end < n:
        tail = "".join(lines[last_end:n])
        if tail.strip():
            chunks.append(Chunk(
                content=tail,
                line_start=last_end + 1,
                line_end=n,
                ordinal=-1,
                kind="module",
            ))

    return chunks


def _find_block_end(lines: list[str], start_index: int) -> int | None:
    """
    Given 0-indexed start line of a declaration, return 1-indexed end line
    where the outermost '{' matching '}' closes. Handles strings, chars,
    line comments, and block comments. None if no opening brace found or
    braces never balance.
    """
    depth = 0
    seen_open = False
    in_line_comment = False
    in_block_comment = False
    in_string: str | None = None  # the delimiter, if inside one
    escape = False
    for line_idx in range(start_index, len(lines)):
        line = lines[line_idx]
        in_line_comment = False  # resets each line
        i = 0
        while i < len(line):
            ch = line[i]
            nxt = line[i + 1] if i + 1 < len(line) else ""

            if in_block_comment:
                if ch == "*" and nxt == "/":
                    in_block_comment = False
                    i += 2
                    continue
                i += 1
                continue

            if in_line_comment:
                i += 1
                continue

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == in_string:
                    in_string = None
                i += 1
                continue

            # Not in string/comment
            if ch == "/" and nxt == "/":
                in_line_comment = True
                i += 2
                continue
            if ch == "/" and nxt == "*":
                in_block_comment = True
                i += 2
                continue
            if ch in ('"', "'", "`"):
                in_string = ch
                i += 1
                continue
            if ch == "{":
                depth += 1
                seen_open = True
            elif ch == "}":
                depth -= 1
                if seen_open and depth == 0:
                    return line_idx + 1  # 1-indexed end line
            i += 1
    return None


# ─────────────────────────────────────────────────────────────
# Fallback — fixed-size line windows with overlap
# ─────────────────────────────────────────────────────────────


def _chunk_windows(content: str) -> list[Chunk]:
    if not content.strip():
        return []
    lines = content.splitlines(keepends=True)
    n = len(lines)
    if n == 0:
        return []

    chunks: list[Chunk] = []
    step = max(1, FALLBACK_WINDOW_LINES - FALLBACK_OVERLAP_LINES)
    start = 0
    while start < n:
        end = min(n, start + FALLBACK_WINDOW_LINES)
        body = "".join(lines[start:end])
        if body.strip():
            chunks.append(Chunk(
                content=body,
                line_start=start + 1,
                line_end=end,
                ordinal=-1,
                kind="window",
            ))
        if end >= n:
            break
        start += step
    return chunks


# ─────────────────────────────────────────────────────────────
# Post-processing: merge tiny / split oversized
# ─────────────────────────────────────────────────────────────


def _merge_tiny(chunks: list[Chunk]) -> list[Chunk]:
    """Merge tiny chunks into their neighbor, but preserve semantic
    boundaries. Rules:
      - `function`, `class`, `heading` chunks are NEVER merged into
        another kind (even if small). A short function is still a
        function — hiding it inside a 'module' chunk loses signal.
      - Low-semantic kinds (`module`, `block`, `window`) can be merged
        into any preceding neighbor.
    """
    if not chunks:
        return chunks
    # `block` covers brace-language declarations (Rust fn/struct/impl,
    # JS function, Go func, etc). These ARE the semantic unit for those
    # languages, so treat them like Python function/class for merge protection.
    _semantic = {"function", "class", "heading", "block"}
    merged: list[Chunk] = []
    for c in chunks:
        if c.kind in _semantic:
            merged.append(c)
            continue
        if c.char_count < MIN_CHUNK_CHARS and merged and merged[-1].kind not in _semantic:
            prev = merged[-1]
            combined = prev.content
            if not combined.endswith("\n"):
                combined += "\n"
            combined += c.content
            merged[-1] = Chunk(
                content=combined,
                line_start=prev.line_start,
                line_end=c.line_end,
                ordinal=prev.ordinal,
                kind=prev.kind,
                anchor=prev.anchor or c.anchor,
            )
        else:
            merged.append(c)
    return merged


def _split_oversized(chunks: list[Chunk]) -> list[Chunk]:
    """
    If a chunk exceeds MAX_CHUNK_CHARS, re-split by character budget
    while preserving line boundaries. A 10,000-line generated file or
    a 200-line megafunction gets broken into ≤MAX_CHUNK_CHARS pieces
    without slicing mid-line.
    """
    out: list[Chunk] = []
    for c in chunks:
        if c.char_count <= MAX_CHUNK_CHARS:
            out.append(c)
            continue
        # Walk lines, keep a running char budget
        lines = c.content.splitlines(keepends=True)
        target = MAX_CHUNK_CHARS
        # Overlap lines between windows so concepts straddling boundaries
        # land in both. Kept small (10% of window) to avoid duplication.
        overlap_chars = max(200, target // 10)

        i = 0
        n = len(lines)
        while i < n:
            buf: list[str] = []
            buf_chars = 0
            chunk_start_line = c.line_start + i
            overlap_start = i  # for next window
            while i < n and buf_chars + len(lines[i]) <= target:
                buf.append(lines[i])
                buf_chars += len(lines[i])
                i += 1
            if not buf:
                # Single line exceeds target — take it anyway to make progress
                buf.append(lines[i])
                buf_chars += len(lines[i])
                i += 1
            body = "".join(buf)
            chunk_end_line = c.line_start + i - 1
            out.append(Chunk(
                content=body,
                line_start=chunk_start_line,
                line_end=chunk_end_line,
                ordinal=-1,
                kind=c.kind + "_window",
                anchor=c.anchor,
            ))
            if i >= n:
                break
            # Rewind for overlap
            back_chars = 0
            back_i = i
            while back_i > overlap_start and back_chars < overlap_chars:
                back_i -= 1
                back_chars += len(lines[back_i])
            i = back_i if back_i > overlap_start else i
    return out


# ─────────────────────────────────────────────────────────────
# Binary/large-file gate — helper for callers
# ─────────────────────────────────────────────────────────────


def looks_binary(sample: bytes) -> bool:
    """True if the first KB contains null bytes or a high ratio of
    non-printables — a cheap filter before we try to decode UTF-8."""
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    printable = sum(
        1 for b in sample
        if b in (9, 10, 13) or 32 <= b < 127 or b >= 128  # UTF-8 multibyte starts >= 128
    )
    return (printable / len(sample)) < 0.85


def safe_read_text(path: str | Path, max_bytes: int = 4_000_000) -> str | None:
    """Read a file as text with guardrails. Returns None if:
      - file missing
      - binary (null bytes / non-printable heavy)
      - larger than max_bytes (default 4 MB)
    Falls back to replacement decoding so weird encodings still yield text."""
    p = Path(path)
    try:
        stat = p.stat()
    except OSError:
        return None
    if stat.st_size > max_bytes:
        logger.info("skipping %s — %d bytes > max %d", p, stat.st_size, max_bytes)
        return None
    try:
        with open(p, "rb") as f:
            head = f.read(1024)
            if looks_binary(head):
                return None
            rest = f.read()
    except OSError:
        return None
    data = head + rest
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")
