"""Root conftest.py — fixes PYTHONPATH and defines the --real flag."""
import sys
from pathlib import Path

import pytest

# Add src/ to sys.path so bare imports like `from common.protos import ...` work
src_dir = str(Path(__file__).parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


# ── --real flag support ───────────────────────────────────────────────
#
# Tests marked with ``@pytest.mark.real`` hit live external APIs
# (Anthropic, OpenAI, Gemini). They cost real dollars per run and
# require credentials (either env vars or macOS Keychain entries).
#
# They are SKIPPED by default. To run them explicitly:
#
#     pytest --real               # enables real tests
#     pytest -m real --real       # runs ONLY real tests
#     pytest -m "not real"        # explicitly excludes (same as default)
#
# Per-test credential checks live in src/jarvis/tests/real/conftest.py
# and skip individual tests when their specific credential is missing
# (e.g. a Claude Code keychain entry for claude_dev tests).


def pytest_addoption(parser):
    parser.addoption(
        "--real",
        action="store_true",
        default=False,
        help=(
            "Run tests marked @pytest.mark.real that hit live external APIs "
            "(Anthropic, OpenAI, Gemini). Requires credentials; costs real "
            "dollars per run. Skipped by default."
        ),
    )


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.real tests unless --real is passed."""
    if config.getoption("--real"):
        return
    skip_marker = pytest.mark.skip(
        reason="needs --real flag (hits live external APIs, costs money)"
    )
    for item in items:
        if "real" in item.keywords:
            item.add_marker(skip_marker)
