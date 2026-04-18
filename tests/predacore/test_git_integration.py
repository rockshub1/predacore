"""
Tests for Git Integration — Phase 8B.

Tests cover:
  - git_context: repository status, branch info, commit log
  - git_diff_summary: diff parsing, file changes, stat parsing
  - git_commit_suggest: commit type classification, scope detection
  - git_find_files: pattern matching, glob support
  - Helper functions: _parse_status, _classify_commit_type, _find_common_scope
  - _glob_match utility
"""
import subprocess
from pathlib import Path

import pytest

from src.predacore.services.git_integration import (
    DiffSummary,
    FileDiff,
    _classify_commit_type,
    _find_common_scope,
    _glob_match,
    _parse_status,
    git_commit_suggest,
    git_context,
    git_diff_summary,
    git_find_files,
)

# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def temp_git_repo(tmp_path):
    """Create a temporary git repository with some commits."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, capture_output=True)

    # Create initial commit
    (repo / "README.md").write_text("# Test Repo\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=repo, capture_output=True)

    return str(repo)


@pytest.fixture
def temp_git_repo_with_changes(temp_git_repo):
    """Git repo with staged and unstaged changes."""
    repo = temp_git_repo

    # Create a modified file
    Path(repo, "README.md").write_text("# Test Repo\n\nUpdated content.\n")

    # Create a new file and stage it
    Path(repo, "new_file.py").write_text("print('hello')\n")
    subprocess.run(["git", "add", "new_file.py"], cwd=repo, capture_output=True)

    # Create an untracked file
    Path(repo, "untracked.txt").write_text("I'm not tracked\n")

    return repo


# ── _parse_status tests ──────────────────────────────────────────────

class TestParseStatus:
    def test_empty_status(self):
        gs = _parse_status("")
        assert gs.branch == ""
        assert gs.is_clean

    def test_branch_header(self):
        gs = _parse_status("# branch.head main\n# branch.upstream origin/main")
        assert gs.branch == "main"
        assert gs.tracking == "origin/main"

    def test_ahead_behind(self):
        gs = _parse_status("# branch.head feat\n# branch.ab +3 -1")
        assert gs.ahead == 3
        assert gs.behind == 1

    def test_untracked_files(self):
        gs = _parse_status("? foo.txt\n? bar.py")
        assert len(gs.untracked) == 2
        assert "foo.txt" in gs.untracked
        assert not gs.is_clean

    def test_clean_repo(self):
        gs = _parse_status("# branch.head main")
        assert gs.is_clean


# ── _classify_commit_type tests ──────────────────────────────────────

class TestClassifyCommitType:
    def test_all_test_files(self):
        files = [("A", "tests/test_foo.py"), ("M", "tests/test_bar.py")]
        assert _classify_commit_type(files) == "test"

    def test_all_docs(self):
        files = [("A", "docs/guide.md"), ("M", "README.md")]
        assert _classify_commit_type(files) == "docs"

    def test_all_config(self):
        files = [("M", "pyproject.toml"), ("M", ".gitignore")]
        assert _classify_commit_type(files) == "chore"

    def test_all_added(self):
        files = [("A", "src/new_module.py"), ("A", "src/helpers.py")]
        assert _classify_commit_type(files) == "feat"

    def test_all_deleted(self):
        files = [("D", "src/old.py"), ("D", "src/deprecated.py")]
        assert _classify_commit_type(files) == "refactor"

    def test_mixed_files(self):
        files = [("M", "src/core.py"), ("A", "src/new.py")]
        assert _classify_commit_type(files) == "feat"

    def test_empty(self):
        assert _classify_commit_type([]) == "chore"

    def test_style_files(self):
        files = [("M", "styles/main.css"), ("M", "styles/app.scss")]
        assert _classify_commit_type(files) == "style"


# ── _find_common_scope tests ─────────────────────────────────────────

class TestFindCommonScope:
    def test_common_directory(self):
        paths = ["src/predacore/core.py", "src/predacore/config.py"]
        assert _find_common_scope(paths) == "predacore"

    def test_no_common(self):
        paths = ["foo.py", "bar.py"]
        assert _find_common_scope(paths) == ""

    def test_single_file(self):
        paths = ["src/predacore/core.py"]
        assert _find_common_scope(paths) == "predacore"

    def test_empty(self):
        assert _find_common_scope([]) == ""

    def test_deeply_nested(self):
        paths = ["a/b/c/d.py", "a/b/c/e.py"]
        assert _find_common_scope(paths) == "c"


# ── _glob_match tests ────────────────────────────────────────────────

class TestGlobMatch:
    def test_star_match(self):
        assert _glob_match("foo.py", "*.py")

    def test_star_no_match(self):
        assert not _glob_match("foo.js", "*.py")

    def test_question_mark(self):
        assert _glob_match("foo.py", "fo?.py")

    def test_no_glob_chars(self):
        assert not _glob_match("foo.py", "foo.py")

    def test_complex_glob(self):
        assert _glob_match("test_core.py", "test_*.py")


# ── FileDiff tests ───────────────────────────────────────────────────

class TestFileDiff:
    def test_summary_added(self):
        fd = FileDiff(path="new.py", status="A", insertions=10)
        assert "added" in fd.summary
        assert "+10" in fd.summary

    def test_summary_renamed(self):
        fd = FileDiff(path="new.py", status="R", old_path="old.py", insertions=5, deletions=3)
        s = fd.summary
        assert "old.py" in s
        assert "new.py" in s
        assert "renamed" in s

    def test_summary_binary(self):
        fd = FileDiff(path="image.png", status="M", binary=True)
        assert "binary" in fd.summary


# ── DiffSummary tests ────────────────────────────────────────────────

class TestDiffSummary:
    def test_to_text(self):
        ds = DiffSummary(
            files=[FileDiff(path="a.py", status="M", insertions=5, deletions=2)],
            total_insertions=5,
            total_deletions=2,
            total_files=1,
        )
        text = ds.to_text()
        assert "1 files" in text
        assert "+5/-2" in text

    def test_to_text_with_diff(self):
        ds = DiffSummary(
            files=[],
            total_files=0,
            diff_text="line1\nline2\nline3",
        )
        text = ds.to_text(include_diff=True)
        assert "Diff" in text
        assert "line1" in text


# ── Integration tests (require git) ──────────────────────────────────

class TestGitContextIntegration:
    @pytest.mark.asyncio
    async def test_not_a_repo(self, tmp_path):
        result = await git_context(cwd=str(tmp_path))
        assert "Not a git repository" in result

    @pytest.mark.asyncio
    async def test_basic_repo(self, temp_git_repo):
        result = await git_context(cwd=temp_git_repo)
        assert "Repository:" in result
        assert "Branch:" in result
        assert "Recent commits:" in result
        assert "initial commit" in result

    @pytest.mark.asyncio
    async def test_with_changes(self, temp_git_repo_with_changes):
        result = await git_context(cwd=temp_git_repo_with_changes)
        assert "Staged" in result
        assert "Modified" in result or "Untracked" in result


class TestGitDiffIntegration:
    @pytest.mark.asyncio
    async def test_not_a_repo(self, tmp_path):
        result = await git_diff_summary(cwd=str(tmp_path))
        assert "Not a git repository" in result

    @pytest.mark.asyncio
    async def test_no_changes(self, temp_git_repo):
        result = await git_diff_summary(cwd=temp_git_repo)
        assert "No changes" in result

    @pytest.mark.asyncio
    async def test_staged_changes(self, temp_git_repo_with_changes):
        result = await git_diff_summary(staged=True, cwd=temp_git_repo_with_changes)
        assert "new_file.py" in result


class TestGitCommitSuggestIntegration:
    @pytest.mark.asyncio
    async def test_not_a_repo(self, tmp_path):
        result = await git_commit_suggest(cwd=str(tmp_path))
        assert "Not a git repository" in result

    @pytest.mark.asyncio
    async def test_no_staged(self, temp_git_repo):
        result = await git_commit_suggest(cwd=temp_git_repo)
        assert "No staged changes" in result

    @pytest.mark.asyncio
    async def test_with_staged(self, temp_git_repo_with_changes):
        result = await git_commit_suggest(cwd=temp_git_repo_with_changes)
        assert "Suggested commit message" in result
        assert "new_file.py" in result


class TestGitFindFilesIntegration:
    @pytest.mark.asyncio
    async def test_not_a_repo(self, tmp_path):
        result = await git_find_files("*.py", cwd=str(tmp_path))
        assert "Not a git repository" in result

    @pytest.mark.asyncio
    async def test_find_readme(self, temp_git_repo):
        result = await git_find_files("README", cwd=temp_git_repo)
        assert "README.md" in result

    @pytest.mark.asyncio
    async def test_no_match(self, temp_git_repo):
        result = await git_find_files("nonexistent_xyz", cwd=temp_git_repo)
        assert "No files matching" in result

    @pytest.mark.asyncio
    async def test_find_with_extension(self, temp_git_repo):
        result = await git_find_files(".md", cwd=temp_git_repo)
        assert "README.md" in result
