"""
PredaCore Git Integration — Semantic git tools for the agent loop.

Provides high-level git operations that go beyond raw shell commands:
  - git_context: repository status + branch info + recent commits
  - git_diff_summary: LLM-friendly diff with file-level change descriptions
  - git_commit_suggest: generate commit messages from staged changes
  - git_find_files: intelligent file search using git's index

Usage:
    from predacore.services.git_integration import (
        git_context, git_diff_summary, git_commit_suggest, git_find_files,
    )
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class GitStatus:
    """Parsed `git status` information."""

    branch: str = ""
    tracking: str = ""
    ahead: int = 0
    behind: int = 0
    staged: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    untracked: list[str] = field(default_factory=list)
    is_clean: bool = True
    is_repo: bool = False
    repo_root: str = ""


@dataclass
class FileDiff:
    """Per-file diff summary."""

    path: str
    status: str  # A=added, M=modified, D=deleted, R=renamed
    insertions: int = 0
    deletions: int = 0
    old_path: str = ""  # for renames
    binary: bool = False

    @property
    def summary(self) -> str:
        status_map = {"A": "added", "M": "modified", "D": "deleted", "R": "renamed"}
        label = status_map.get(self.status, self.status)
        if self.binary:
            return f"{self.path} ({label}, binary)"
        parts = []
        if self.insertions:
            parts.append(f"+{self.insertions}")
        if self.deletions:
            parts.append(f"-{self.deletions}")
        change = ", ".join(parts) if parts else "no changes"
        if self.status == "R" and self.old_path:
            return f"{self.old_path} → {self.path} ({label}, {change})"
        return f"{self.path} ({label}, {change})"


@dataclass
class DiffSummary:
    """Complete diff summary."""

    files: list[FileDiff] = field(default_factory=list)
    total_insertions: int = 0
    total_deletions: int = 0
    total_files: int = 0
    diff_text: str = ""  # raw diff, optionally truncated

    def to_text(self, include_diff: bool = False, max_diff_lines: int = 200) -> str:
        lines = [
            f"Changes: {self.total_files} files, +{self.total_insertions}/-{self.total_deletions}"
        ]
        lines.append("")
        for f in self.files:
            lines.append(f"  {f.summary}")
        if include_diff and self.diff_text:
            diff_lines = self.diff_text.splitlines()
            if len(diff_lines) > max_diff_lines:
                diff_lines = diff_lines[:max_diff_lines]
                diff_lines.append(
                    f"... ({len(self.diff_text.splitlines()) - max_diff_lines} more lines)"
                )
            lines.append("")
            lines.append("--- Diff ---")
            lines.extend(diff_lines)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_cwd(cwd: str | None) -> str | None:
    """Resolve *cwd* to a real path and reject traversal outside the repo."""
    if cwd is None:
        return None
    resolved = Path(cwd).resolve()
    if not resolved.is_dir():
        raise ValueError(f"cwd is not a directory: {cwd}")
    return str(resolved)


async def _run_git(
    *args: str,
    cwd: str | None = None,
    timeout: float = 15.0,
) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    cwd = _validate_cwd(cwd)
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return -1, "", "git command timed out"
    return (
        proc.returncode or 0,
        stdout.decode("utf-8", errors="replace").strip(),
        stderr.decode("utf-8", errors="replace").strip(),
    )


async def _find_repo_root(cwd: str | None = None) -> str | None:
    """Find the git repository root, or None if not in a repo."""
    rc, out, _ = await _run_git("rev-parse", "--show-toplevel", cwd=cwd)
    return out if rc == 0 and out else None


# ---------------------------------------------------------------------------
# git_context — full repo status snapshot
# ---------------------------------------------------------------------------


async def git_context(cwd: str | None = None, log_count: int = 10) -> str:
    """
    Return a comprehensive git context string for the LLM.

    Includes: branch, tracking info, ahead/behind, staged/modified/untracked
    files, and recent commit log.
    """
    repo_root = await _find_repo_root(cwd)
    if not repo_root:
        return "Not a git repository."

    # Run multiple git commands in parallel
    status_task = _run_git("status", "--porcelain=v2", "--branch", cwd=repo_root)
    log_task = _run_git(
        "log",
        "--oneline",
        f"-{log_count}",
        "--format=%h %s (%ar)",
        "--no-decorate",
        cwd=repo_root,
    )
    stash_task = _run_git("stash", "list", "--format=%gd: %gs", cwd=repo_root)
    remote_task = _run_git("remote", "-v", cwd=repo_root)

    (
        (status_rc, status_out, _),
        (log_rc, log_out, _),
        (stash_rc, stash_out, _),
        (remote_rc, remote_out, _),
    ) = await asyncio.gather(status_task, log_task, stash_task, remote_task)

    # Parse status
    gs = _parse_status(status_out if status_rc == 0 else "")
    gs.is_repo = True
    gs.repo_root = repo_root

    # Build output
    lines = [f"Repository: {repo_root}"]
    lines.append(f"Branch: {gs.branch}")
    if gs.tracking:
        tracking_info = gs.tracking
        if gs.ahead or gs.behind:
            parts = []
            if gs.ahead:
                parts.append(f"ahead {gs.ahead}")
            if gs.behind:
                parts.append(f"behind {gs.behind}")
            tracking_info += f" ({', '.join(parts)})"
        lines.append(f"Tracking: {tracking_info}")

    if gs.is_clean:
        lines.append("Working tree: clean")
    else:
        if gs.staged:
            lines.append(f"\nStaged ({len(gs.staged)}):")
            for f in gs.staged[:20]:
                lines.append(f"  {f}")
        if gs.modified:
            lines.append(f"\nModified ({len(gs.modified)}):")
            for f in gs.modified[:20]:
                lines.append(f"  {f}")
        if gs.untracked:
            lines.append(f"\nUntracked ({len(gs.untracked)}):")
            for f in gs.untracked[:20]:
                lines.append(f"  {f}")

    # Remote
    if remote_out:
        remotes = set()
        for line in remote_out.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                remotes.add(f"{parts[0]} → {parts[1]}")
        if remotes:
            lines.append("\nRemotes:")
            for r in sorted(remotes):
                lines.append(f"  {r}")

    # Stash
    if stash_out:
        stash_entries = stash_out.splitlines()[:5]
        lines.append(f"\nStash ({len(stash_out.splitlines())} entries):")
        for s in stash_entries:
            lines.append(f"  {s}")

    # Recent commits
    if log_out:
        lines.append("\nRecent commits:")
        for commit_line in log_out.splitlines()[:log_count]:
            lines.append(f"  {commit_line}")

    return "\n".join(lines)


def _parse_status(raw: str) -> GitStatus:
    """Parse `git status --porcelain=v2 --branch` output."""
    gs = GitStatus()
    for line in raw.splitlines():
        if line.startswith("# branch.head "):
            gs.branch = line.split(" ", 2)[2]
        elif line.startswith("# branch.upstream "):
            gs.tracking = line.split(" ", 2)[2]
        elif line.startswith("# branch.ab "):
            parts = line.split()
            for p in parts[2:]:
                if p.startswith("+"):
                    gs.ahead = int(p[1:])
                elif p.startswith("-"):
                    gs.behind = int(p[1:])
        elif line.startswith("1 ") or line.startswith("2 "):
            # Changed entry
            parts = line.split("\t")
            info = parts[0].split()
            xy = info[1] if len(info) > 1 else ".."
            filepath = parts[-1] if "\t" in line else info[-1]

            if xy[0] != ".":
                gs.staged.append(f"{xy[0]} {filepath}")
            if xy[1] != ".":
                gs.modified.append(f"{xy[1]} {filepath}")
        elif line.startswith("? "):
            gs.untracked.append(line[2:])

    gs.is_clean = not (gs.staged or gs.modified or gs.untracked)
    return gs


# ---------------------------------------------------------------------------
# git_diff_summary — structured diff information
# ---------------------------------------------------------------------------


async def git_diff_summary(
    ref: str = "HEAD",
    staged: bool = False,
    cwd: str | None = None,
    include_diff: bool = False,
    max_diff_lines: int = 200,
) -> str:
    """
    Return a summary of changes vs. a reference.

    Args:
        ref: Git ref to diff against (default: HEAD)
        staged: If True, show staged changes only (--cached)
        cwd: Working directory
        include_diff: If True, include raw diff text
        max_diff_lines: Maximum diff lines to include
    """
    repo_root = await _find_repo_root(cwd)
    if not repo_root:
        return "Not a git repository."

    # Build diff command
    stat_args = ["diff", "--stat", "--stat-width=80"]
    diff_args = ["diff"]

    if staged:
        stat_args.append("--cached")
        diff_args.append("--cached")
    else:
        stat_args.append(ref)
        diff_args.append(ref)

    # Also get name-status for structured data
    ns_args = ["diff", "--name-status"]
    if staged:
        ns_args.append("--cached")
    else:
        ns_args.append(ref)

    # Run in parallel
    tasks = [
        _run_git(*stat_args, cwd=repo_root),
        _run_git(*ns_args, cwd=repo_root),
    ]
    if include_diff:
        tasks.append(_run_git(*diff_args, cwd=repo_root))

    results = await asyncio.gather(*tasks)
    stat_rc, stat_out, _ = results[0]
    ns_rc, ns_out, _ = results[1]
    diff_text = results[2][1] if include_diff and len(results) > 2 else ""

    if ns_rc != 0:
        return f"No changes found (or invalid ref: {ref})."

    if not ns_out.strip():
        return "No changes."

    # Parse name-status
    files: list[FileDiff] = []
    for line in ns_out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status_code = parts[0][0]  # First char: A, M, D, R, etc.
        if status_code == "R" and len(parts) >= 3:
            files.append(FileDiff(path=parts[2], status="R", old_path=parts[1]))
        else:
            files.append(FileDiff(path=parts[1], status=status_code))

    # Parse stat for insertion/deletion counts
    total_ins = 0
    total_del = 0
    for line in stat_out.splitlines():
        # Match lines like: " file.py | 10 +++---"
        m = re.match(r"\s*(.+?)\s*\|\s*(\d+)\s*(\+*)(−*|-*)", line)
        if m:
            fname = m.group(1).strip()
            changes = int(m.group(2))
            plus_chars = len(m.group(3))
            minus_chars = len(m.group(4))
            total = plus_chars + minus_chars
            if total > 0:
                ins = round(changes * plus_chars / total)
                dels = changes - ins
            else:
                ins = dels = 0

            # Match to our FileDiff
            for f in files:
                if f.path == fname or fname.endswith(f.path):
                    f.insertions = ins
                    f.deletions = dels
                    break
            total_ins += ins
            total_del += dels
        elif "Bin" in line:
            # Binary file
            fname_match = re.match(r"\s*(.+?)\s*\|", line)
            if fname_match:
                for f in files:
                    if f.path == fname_match.group(1).strip():
                        f.binary = True

    summary = DiffSummary(
        files=files,
        total_insertions=total_ins,
        total_deletions=total_del,
        total_files=len(files),
        diff_text=diff_text,
    )

    return summary.to_text(include_diff=include_diff, max_diff_lines=max_diff_lines)


# ---------------------------------------------------------------------------
# git_commit_suggest — generate commit message from staged changes
# ---------------------------------------------------------------------------


async def git_commit_suggest(cwd: str | None = None) -> str:
    """
    Analyze staged changes and suggest a commit message.

    Returns a structured string with:
      - Type classification (feat/fix/refactor/docs/test/chore/style)
      - Suggested commit message
      - File summary
    """
    repo_root = await _find_repo_root(cwd)
    if not repo_root:
        return "Not a git repository."

    # Get staged changes
    ns_rc, ns_out, _ = await _run_git(
        "diff", "--cached", "--name-status", cwd=repo_root
    )
    if ns_rc != 0 or not ns_out.strip():
        return "No staged changes. Use `git add` first."

    # Get stat
    stat_rc, stat_out, _ = await _run_git(
        "diff", "--cached", "--stat", "--stat-width=80", cwd=repo_root
    )

    # Parse files
    files: list[tuple[str, str]] = []  # (status, path)
    for line in ns_out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            files.append((parts[0][0], parts[-1]))

    # Classify commit type
    commit_type = _classify_commit_type(files)

    # Build scope from common directory
    paths = [p for _, p in files]
    scope = _find_common_scope(paths)

    # Generate subject line
    added = [p for s, p in files if s == "A"]
    modified = [p for s, p in files if s == "M"]
    deleted = [p for s, p in files if s == "D"]

    subject_parts = []
    if added:
        if len(added) == 1:
            subject_parts.append(f"add {Path(added[0]).name}")
        else:
            subject_parts.append(f"add {len(added)} files")
    if modified:
        if len(modified) == 1:
            subject_parts.append(f"update {Path(modified[0]).name}")
        else:
            subject_parts.append(f"update {len(modified)} files")
    if deleted:
        if len(deleted) == 1:
            subject_parts.append(f"remove {Path(deleted[0]).name}")
        else:
            subject_parts.append(f"remove {len(deleted)} files")

    action = " and ".join(subject_parts) if subject_parts else "update code"

    scope_prefix = f"({scope}): " if scope else ": "
    suggested = f"{commit_type}{scope_prefix}{action}"

    # Build output
    lines = [
        "Suggested commit message:",
        f"  {suggested}",
        "",
        f"Type: {commit_type}",
        f"Scope: {scope or '(root)'}",
        f"Files: {len(files)}",
    ]

    if stat_out:
        lines.append("")
        lines.append("Changes:")
        for stat_line in stat_out.splitlines():
            if stat_line.strip():
                lines.append(f"  {stat_line.strip()}")

    return "\n".join(lines)


def _classify_commit_type(files: list[tuple[str, str]]) -> str:
    """Classify the commit type based on changed files."""
    paths = [p for _, p in files]

    test_files = sum(1 for p in paths if "test" in p.lower() or p.endswith("_test.py"))
    doc_files = sum(
        1
        for p in paths
        if p.lower().endswith((".md", ".rst", ".txt"))
        or "docs/" in p.lower()
        or "doc/" in p.lower()
    )
    config_files = sum(
        1
        for p in paths
        if p.lower().endswith((".toml", ".yaml", ".yml", ".json", ".cfg", ".ini"))
        or p in ("Makefile", "Dockerfile", ".gitignore")
    )
    style_files = sum(
        1 for p in paths if p.lower().endswith((".css", ".scss", ".less"))
    )

    total = len(files)
    if total == 0:
        return "chore"

    if test_files == total:
        return "test"
    if doc_files == total:
        return "docs"
    if config_files == total:
        return "chore"
    if style_files == total:
        return "style"

    # Check for all new files
    all_added = all(s == "A" for s, _ in files)
    if all_added:
        return "feat"

    # Check for all deletes
    all_deleted = all(s == "D" for s, _ in files)
    if all_deleted:
        return "refactor"

    return "feat"


def _find_common_scope(paths: list[str]) -> str:
    """Find a common directory scope from a list of file paths."""
    if not paths:
        return ""

    # Get directory parts
    dirs = []
    for p in paths:
        parts = Path(p).parts
        if len(parts) > 1:
            dirs.append(parts[:-1])  # Exclude filename
        else:
            dirs.append(())

    if not dirs or any(len(d) == 0 for d in dirs):
        return ""

    # Find common prefix
    common = list(dirs[0])
    for d in dirs[1:]:
        new_common = []
        for a, b in zip(common, d, strict=False):
            if a == b:
                new_common.append(a)
            else:
                break
        common = new_common

    if not common:
        return ""

    # Use the last component as scope
    scope = common[-1]
    # Strip common prefixes like "src/" — use the component before it
    if scope in ("src", "lib", "app"):
        if len(common) > 1:
            scope = common[-2]
        elif len(dirs[0]) > len(common):
            return ""

    return scope


# ---------------------------------------------------------------------------
# git_find_files — search files using git index
# ---------------------------------------------------------------------------


async def git_find_files(
    pattern: str,
    cwd: str | None = None,
    max_results: int = 50,
) -> str:
    """
    Search for files in the git index matching a pattern.

    Uses git ls-files for tracked files and supports glob patterns.
    Faster than filesystem search since it uses git's index.
    """
    repo_root = await _find_repo_root(cwd)
    if not repo_root:
        return "Not a git repository."

    # Search tracked files
    rc, out, _ = await _run_git("ls-files", cwd=repo_root)
    if rc != 0:
        return "Failed to list files."

    all_files = out.splitlines()

    # Filter by pattern (case-insensitive glob-like match)
    pattern_lower = pattern.lower()
    matched = []
    for f in all_files:
        f_lower = f.lower()
        # Support multiple matching strategies
        if pattern_lower in f_lower:
            matched.append(f)
        elif _glob_match(f_lower, pattern_lower):
            matched.append(f)

    if not matched:
        return f"No files matching '{pattern}'."

    # Sort: exact filename matches first, then by path
    def sort_key(f: str) -> tuple[int, str]:
        name = Path(f).name.lower()
        if name == pattern_lower:
            return (0, f)
        if name.startswith(pattern_lower):
            return (1, f)
        if pattern_lower in name:
            return (2, f)
        return (3, f)

    matched.sort(key=sort_key)

    total = len(matched)
    if total > max_results:
        matched = matched[:max_results]

    lines = [f"Found {total} files matching '{pattern}':"]
    for f in matched:
        lines.append(f"  {f}")
    if total > max_results:
        lines.append(f"  ... and {total - max_results} more")

    return "\n".join(lines)


def _glob_match(text: str, pattern: str) -> bool:
    """Simple glob matching: * matches any sequence, ? matches single char."""
    if "*" not in pattern and "?" not in pattern:
        return False

    # Convert glob to regex
    regex = ""
    for ch in pattern:
        if ch == "*":
            regex += ".*"
        elif ch == "?":
            regex += "."
        elif ch in ".+^${}()|[]":
            regex += "\\" + ch
        else:
            regex += ch

    try:
        return bool(re.match(f"^{regex}$", text))
    except re.error:
        return False
