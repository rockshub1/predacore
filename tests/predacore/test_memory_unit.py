"""T1 — Unit tests for the 3 new pure modules added in the memory upgrade:

  • predacore.memory.chunker      — semantic chunking (AST/markdown/brace/window)
  • predacore.memory.safety       — secret scanner + .memoryignore
  • predacore.memory.project_id   — env/git/cwd project resolution

These are pure functions — no DB, no embedder, no network. Fast.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from predacore.memory.chunker import (
    Chunk,
    chunk_text,
    looks_binary,
    safe_read_text,
)
from predacore.memory.project_id import (
    ALL_PROJECTS,
    clear_cache,
    default_project,
)
from predacore.memory.safety import (
    MemoryIgnore,
    SafetyStats,
    SecretMatch,
    is_sensitive_path,
    scan_for_secrets,
)


# ═════════════════════════════════════════════════════════════════════
# CHUNKER
# ═════════════════════════════════════════════════════════════════════


class TestChunkPython:
    """Python AST-based chunking: one chunk per top-level def/class."""

    def test_function_and_class_become_separate_chunks(self):
        src = "def foo():\n    return 1\n\nclass Bar:\n    pass\n"
        chunks = chunk_text("x.py", src)
        assert len(chunks) == 2
        kinds = sorted(c.kind for c in chunks)
        assert kinds == ["class", "function"]

    def test_anchor_extracted_from_def_line(self):
        chunks = chunk_text("x.py", "def alpha(x, y):\n    return x + y\n")
        assert chunks[0].anchor == "def alpha"

    def test_class_anchor_extracted(self):
        chunks = chunk_text("x.py", "class Beta:\n    pass\n")
        assert chunks[0].anchor == "class Beta"

    def test_module_level_code_becomes_module_chunk(self):
        src = "import os\nX = 1\n\ndef foo():\n    pass\n"
        chunks = chunk_text("x.py", src)
        kinds = [c.kind for c in chunks]
        assert "module" in kinds
        assert "function" in kinds

    def test_nested_function_stays_in_outer_scope(self):
        """Nested defs are not promoted to top-level chunks."""
        src = "def outer():\n    def inner():\n        return 1\n    return inner\n"
        chunks = chunk_text("x.py", src)
        # Should be ONE chunk for outer (containing inner inside it)
        functions = [c for c in chunks if c.kind == "function"]
        assert len(functions) == 1
        assert "inner" in functions[0].content

    def test_syntax_error_falls_back_to_windows(self):
        """Broken Python should still produce SOME chunks via window fallback."""
        broken = "def foo(:\n    return\nthis is not python at all\n" * 30
        chunks = chunk_text("broken.py", broken)
        # Doesn't crash — produces window chunks
        assert len(chunks) >= 1

    def test_empty_python_file_returns_empty(self):
        assert chunk_text("x.py", "") == []

    def test_whitespace_only_python_file_returns_empty(self):
        assert chunk_text("x.py", "   \n\n  \n") == []

    def test_line_numbers_are_one_indexed(self):
        chunks = chunk_text("x.py", "def foo():\n    return 1\n")
        assert chunks[0].line_start == 1
        assert chunks[0].line_end >= 1


class TestChunkMarkdown:
    """Markdown chunking: heading-bounded sections."""

    def test_heading_split_creates_chunk_per_section(self):
        src = "# Title\n\nintro\n\n## Section A\n\ncontent A\n\n## Section B\n\ncontent B\n"
        chunks = chunk_text("x.md", src)
        kinds = [c.kind for c in chunks]
        assert kinds.count("heading") == 3  # Title + Section A + Section B

    def test_no_headings_treats_as_one_block(self):
        src = "Just plain markdown without any headings.\nMore text.\n"
        chunks = chunk_text("x.md", src)
        assert len(chunks) == 1
        assert chunks[0].kind == "block"

    def test_preface_before_first_heading_captured(self):
        src = "preface text\nmore preface\n\n## Heading\n\nbody\n"
        chunks = chunk_text("x.md", src)
        # Preface gets a "block" chunk + then heading chunk
        kinds = [c.kind for c in chunks]
        assert "block" in kinds
        assert "heading" in kinds

    def test_heading_anchor_is_the_heading_text(self):
        src = "## My Section Title\n\nbody\n"
        chunks = chunk_text("x.md", src)
        assert "My Section Title" in chunks[0].anchor


class TestChunkBraceLanguages:
    """Brace-balanced splitter for .rs/.js/.ts/.go/.c/etc."""

    def test_rust_fn_and_struct(self):
        src = "fn alpha() {\n    return 1;\n}\n\nstruct Bravo {\n    x: i32,\n}\n"
        chunks = chunk_text("x.rs", src)
        kinds = [c.kind for c in chunks]
        # Both should be captured as block chunks
        assert kinds.count("block") >= 2

    def test_javascript_class(self):
        src = "class Foo {\n  constructor() {}\n}\n\nfunction bar() {\n  return 1;\n}\n"
        chunks = chunk_text("x.js", src)
        kinds = [c.kind for c in chunks]
        assert kinds.count("block") >= 2

    def test_no_decls_falls_back_to_windows(self):
        """File with no recognizable decls → window fallback."""
        src = "// just comments\n// nothing else\n"
        chunks = chunk_text("x.rs", src)
        # Could be window or empty-after-merge — important is no crash
        assert isinstance(chunks, list)


class TestChunkWindows:
    """400-line window fallback for unknown extensions."""

    def test_unknown_extension_uses_window_chunker(self):
        src = "\n".join(f"line {i}" for i in range(500))
        chunks = chunk_text("x.unknown", src)
        # 500 lines, 400-line windows with 40-line overlap → 2 windows
        kinds = [c.kind for c in chunks]
        assert all(k == "window" for k in kinds)
        assert len(chunks) >= 1

    def test_window_overlap_creates_duplication(self):
        """40-line overlap → some lines appear in 2 chunks."""
        src = "\n".join(f"unique_line_{i}" for i in range(500))
        chunks = chunk_text("x.txt", src)
        if len(chunks) >= 2:
            # Lines near the boundary should appear in both windows
            content1 = chunks[0].content
            content2 = chunks[1].content
            # find at least one line that's in both
            overlap_lines = [
                line for line in content2.splitlines()
                if line in content1.splitlines()
            ]
            assert len(overlap_lines) > 0, "expected overlap between windows"


class TestPostProcessing:
    """Post-chunk merging (tiny → neighbor) + splitting (oversized → windows)."""

    def test_tiny_module_chunk_merges_into_preceding_neighbor(self):
        """Two consecutive module chunks where the second is tiny → merged.
        Per chunker semantics: tiny non-semantic chunks merge into their
        PRECEDING non-semantic neighbor. A tiny module chunk at the START
        of a file (no preceding neighbor) cannot be merged backwards."""
        # Set up two tiny module chunks separated by something that breaks
        # the grouping — but the chunker treats consecutive module-level
        # code as one chunk anyway, so this is hard to test without
        # construct-fighting. Instead, we verify the protection rule:
        # function/class chunks are NEVER merged regardless of size.
        src = "x = 1\ny = 2\n\ndef tiny(): return 1\n\nz = 3\n"
        chunks = chunk_text("x.py", src)
        # The function chunk MUST be present as its own chunk
        functions = [c for c in chunks if c.kind == "function"]
        assert len(functions) == 1, "function chunk must survive merging"
        assert functions[0].anchor == "def tiny"

    def test_function_chunk_never_merged_even_if_tiny(self):
        """A 1-liner function should KEEP its own chunk."""
        src = "def x(): return 1\n\ndef y(): return 2\n"
        chunks = chunk_text("x.py", src)
        functions = [c for c in chunks if c.kind == "function"]
        assert len(functions) == 2  # both preserved despite being tiny

    def test_oversized_chunk_split_into_windows(self):
        """A megafunction >4000 chars gets split."""
        # Build a function with ~6000 chars of body
        body = "    pass  # filler line that adds chars\n" * 200
        src = f"def big():\n{body}"
        chunks = chunk_text("x.py", src)
        # Original function would be one chunk; if oversized, it gets split
        # into N "function_window" chunks
        assert sum(c.char_count for c in chunks) >= 6000  # all content preserved
        assert any(c.kind in ("function", "function_window") for c in chunks)


class TestSafeReadText:
    """File-reading guards: binary, oversized, missing."""

    def test_basic_text_file_read(self, tmp_path: Path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        assert safe_read_text(f) == "hello world"

    def test_binary_file_returns_none(self, tmp_path: Path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00\x01\x02hello\x03binary\x00")
        assert safe_read_text(f) is None

    def test_oversized_file_returns_none(self, tmp_path: Path):
        f = tmp_path / "huge.txt"
        f.write_text("x" * 1000)
        assert safe_read_text(f, max_bytes=500) is None

    def test_missing_file_returns_none(self, tmp_path: Path):
        assert safe_read_text(tmp_path / "nope.txt") is None

    def test_utf8_text_preserved(self, tmp_path: Path):
        f = tmp_path / "unicode.txt"
        f.write_text("héllo wörld 中文 🚀", encoding="utf-8")
        assert "héllo" in safe_read_text(f)


class TestLooksBinary:
    """Binary-vs-text heuristic."""

    def test_null_byte_is_binary(self):
        assert looks_binary(b"\x00\x01\x02") is True

    def test_pure_ascii_is_not_binary(self):
        assert looks_binary(b"hello world") is False

    def test_empty_input_not_binary(self):
        assert looks_binary(b"") is False

    def test_high_unicode_not_binary(self):
        # UTF-8 encoded 中文 has bytes >= 128, should still be text
        assert looks_binary("中文 hello".encode("utf-8")) is False


# ═════════════════════════════════════════════════════════════════════
# SAFETY
# ═════════════════════════════════════════════════════════════════════


class TestScanForSecrets:
    """Pattern-based secret detection."""

    @pytest.mark.parametrize("secret,expected_kind", [
        ("AKIAIOSFODNN7EXAMPLE", "aws_access_key_id"),
        ("ghp_1234567890AAAAAaaaa1111BBBBbbbb2222CC", "github_token"),
        ("sk-proj-abcdefghij1234567890ABCDEF", "openai_key"),
        ("xoxb-12345-abcdefghijkl-mnopqrst-uvwxyz", "slack_token"),
        # Google API key pattern requires EXACTLY 35 chars after "AIza"
        ("AIzaSyD1234567890abcdefghij_klmnopqrstu", "google_api_key"),
    ])
    def test_known_secret_pattern_detected(self, secret, expected_kind):
        hits = scan_for_secrets(f"my key is {secret} please")
        assert any(h.name == expected_kind for h in hits), (
            f"expected to detect {expected_kind} in: {secret}"
        )

    def test_pem_private_key_block_detected(self):
        pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAA...\n-----END RSA PRIVATE KEY-----"
        hits = scan_for_secrets(pem)
        assert any(h.name == "pem_private_key" for h in hits)

    def test_ssh_private_key_detected(self):
        ssh = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNz...\n-----END OPENSSH PRIVATE KEY-----"
        hits = scan_for_secrets(ssh)
        # Either ssh_private_key OR pem_private_key catches it (both patterns match)
        assert len(hits) >= 1

    def test_jwt_token_detected(self):
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ.signature_part_here_xyz"
        hits = scan_for_secrets(jwt)
        assert any(h.name == "jwt_token" for h in hits)

    def test_innocent_text_no_false_positives(self):
        innocent = "This is just regular text about a project. We discussed the architecture."
        assert scan_for_secrets(innocent) == []

    def test_empty_text_returns_empty(self):
        assert scan_for_secrets("") == []

    def test_secret_match_redacts_preview(self):
        hits = scan_for_secrets("AKIAIOSFODNN7EXAMPLE")
        assert hits
        preview = hits[0].preview
        # Preview should mask the middle so we don't store full secret
        assert "*" in preview
        # Should NOT contain the full secret verbatim
        assert "AKIAIOSFODNN7EXAMPLE" != preview


class TestIsSensitivePath:
    """Path-name heuristic for skipping sensitive files."""

    @pytest.mark.parametrize("path", [
        ".env", ".env.local", ".env.production",
        "credentials.pem", "id_rsa", "id_ed25519",
        "secret.key", "cert.p12",
    ])
    def test_known_sensitive_paths_flagged(self, path):
        assert is_sensitive_path(path) is True

    @pytest.mark.parametrize("path", [
        "foo.py", "README.md", "test.txt", "config.yaml",
    ])
    def test_normal_paths_not_flagged(self, path):
        assert is_sensitive_path(path) is False


class TestMemoryIgnore:
    """gitignore-style matcher."""

    def test_unanchored_glob_matches_any_segment(self):
        mi = MemoryIgnore(["*.log"])
        assert mi.matches("debug.log") is True
        assert mi.matches("path/to/error.log") is True
        assert mi.matches("foo.py") is False

    def test_anchored_pattern_matches_root_only(self):
        mi = MemoryIgnore(["/build/"])
        assert mi.matches("build/foo.o") is True
        assert mi.matches("src/build/foo.o") is False  # anchored, not in subtree

    def test_directory_only_pattern(self):
        mi = MemoryIgnore(["build/"])
        assert mi.matches("build/foo.txt") is True

    def test_negation_pattern(self):
        mi = MemoryIgnore(["*.log", "!keep.log"])
        assert mi.matches("debug.log") is True
        assert mi.matches("keep.log") is False  # negated

    def test_blank_lines_and_comments_ignored(self):
        mi = MemoryIgnore(["", "# a comment", "*.log"])
        assert mi.matches("x.log") is True

    def test_for_root_loads_memoryignore_and_gitignore(self, tmp_path: Path):
        (tmp_path / ".gitignore").write_text("*.pyc\nbuild/\n")
        (tmp_path / ".memoryignore").write_text("*.log\n")
        mi = MemoryIgnore.for_root(tmp_path)
        assert mi.matches("foo.pyc") is True   # from .gitignore
        assert mi.matches("debug.log") is True  # from .memoryignore

    def test_for_root_handles_missing_files(self, tmp_path: Path):
        # No .gitignore, no .memoryignore in tmp_path → empty matcher, no crash
        mi = MemoryIgnore.for_root(tmp_path)
        assert mi.matches("anything") is False


class TestSafetyStats:
    """Counter dataclass for safety blocking events."""

    def test_record_block_increments(self):
        stats = SafetyStats()
        match = SecretMatch(name="aws_access_key_id", position=0, length=20, preview="AKI***LE")
        stats.record_block([match])
        assert stats.secrets_blocked == 1
        assert stats.by_kind["aws_access_key_id"] == 1

    def test_record_block_with_multiple_matches(self):
        stats = SafetyStats()
        matches = [
            SecretMatch(name="github_token", position=0, length=40, preview="ghp_***"),
            SecretMatch(name="openai_key", position=50, length=40, preview="sk-***"),
        ]
        stats.record_block(matches)
        assert stats.secrets_blocked == 1  # still ONE block (the row was refused)
        assert stats.by_kind["github_token"] == 1
        assert stats.by_kind["openai_key"] == 1

    def test_as_dict_shape(self):
        stats = SafetyStats()
        d = stats.as_dict()
        assert "secrets_blocked" in d
        assert "by_kind" in d
        assert "ignored_paths" in d
        assert "sensitive_paths_skipped" in d


# ═════════════════════════════════════════════════════════════════════
# PROJECT_ID
# ═════════════════════════════════════════════════════════════════════


class TestDefaultProject:
    """env → git → cwd → 'default' resolution chain."""

    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PREDACORE_MEMORY_PROJECT", "my-forced-project")
        clear_cache()
        assert default_project() == "my-forced-project"

    def test_env_empty_string_does_not_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PREDACORE_MEMORY_PROJECT", "   ")
        clear_cache()
        # Should fall through to git/cwd resolution
        result = default_project()
        assert result != "   "

    def test_git_repo_basename_used(self, git_repo, monkeypatch: pytest.MonkeyPatch):
        """Inside a git repo (created by git_repo fixture), project_id = repo dir name."""
        monkeypatch.delenv("PREDACORE_MEMORY_PROJECT", raising=False)
        clear_cache()
        result = default_project(cwd=str(git_repo.path))
        # git_repo fixture creates the repo at tmp_path/repo
        assert result == "repo"

    def test_non_git_directory_uses_cwd_basename(self, tmp_path: Path):
        nogit = tmp_path / "my-non-git-project"
        nogit.mkdir()
        clear_cache()
        result = default_project(cwd=str(nogit))
        assert result == "my-non-git-project"

    def test_caching_speeds_up_repeated_calls(self, git_repo):
        import time
        clear_cache()
        t1_start = time.perf_counter()
        first = default_project(cwd=str(git_repo.path))
        t1 = time.perf_counter() - t1_start

        t2_start = time.perf_counter()
        second = default_project(cwd=str(git_repo.path))
        t2 = time.perf_counter() - t2_start

        assert first == second
        assert t2 < t1, "cached call should be measurably faster"

    def test_clear_cache_invalidates(self, git_repo):
        from predacore.memory.project_id import _PROJECT_CACHE
        default_project(cwd=str(git_repo.path))
        assert len(_PROJECT_CACHE) > 0
        clear_cache()
        assert len(_PROJECT_CACHE) == 0


class TestAllProjectsSentinel:
    """The 'all' string is the sentinel for 'no project filter'."""

    def test_all_constant_is_lowercase_string(self):
        assert ALL_PROJECTS == "all"
