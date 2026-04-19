"""Git handlers: git_context, git_diff_summary, git_commit_suggest, git_find_files, git_semantic_search."""
from __future__ import annotations

from ._context import (
    ToolContext,
    invalid_param,
    missing_param,
)


async def handle_git_context(args: dict, ctx: ToolContext) -> str:
    """Get current git repository context (branch, status, recent logs)."""
    from predacore.services.git_integration import git_context

    return await git_context(
        cwd=args.get("cwd"),
        log_count=args.get("log_count", 10),
    )


async def handle_git_diff_summary(args: dict, ctx: ToolContext) -> str:
    """Summarize git diff for a given ref."""
    from predacore.services.git_integration import git_diff_summary

    ref = str(args.get("ref", "HEAD")).strip()
    # Prevent git flag injection via ref parameter
    if ref.startswith("-"):
        raise invalid_param("ref", "must not start with '-'", tool="git_diff_summary")
    return await git_diff_summary(
        ref=ref,
        staged=args.get("staged", False),
        include_diff=args.get("include_diff", False),
    )


async def handle_git_commit_suggest(args: dict, ctx: ToolContext) -> str:
    """Suggest a commit message based on staged changes."""
    from predacore.services.git_integration import git_commit_suggest

    return await git_commit_suggest()


async def handle_git_find_files(args: dict, ctx: ToolContext) -> str:
    """Find files in the git repository matching a pattern."""
    from predacore.services.git_integration import git_find_files

    pattern = args.get("pattern", "")
    if not pattern:
        raise missing_param("pattern", tool="git_find_files")
    return await git_find_files(
        pattern=pattern,
        max_results=args.get("max_results", 50),
    )


async def handle_git_semantic_search(args: dict, ctx: ToolContext) -> str:
    """Semantic code search over git-tracked files using natural language."""
    from predacore.services.code_index import semantic_code_search

    query = str(args.get("query") or "").strip()
    if not query:
        raise missing_param("query", tool="git_semantic_search")

    return await semantic_code_search(
        query=query,
        top_k=max(1, min(int(args.get("top_k") or 10), 30)),
        file_pattern=args.get("file_pattern"),
        rebuild=bool(args.get("rebuild", False)),
    )
