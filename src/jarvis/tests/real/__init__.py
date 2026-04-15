"""
Real integration tests — hit live external APIs.

Every test in this package is marked ``@pytest.mark.real`` and skipped
by default. Enable with ``pytest --real``. Requires credentials (API
keys or Claude Code OAuth keychain entries). Costs real dollars per run
— do not run on every commit.

Purpose: catch the class of bugs that mock-based tests can't catch —
vendor contract enforcement (thinking signatures, tool_use_id linking,
beta flag acceptance, strict tool schema enforcement, etc.). The
content_blocks round-trip bug that broke JARVIS on Telegram in
April 2026 would have been caught on day one by these tests.
"""
