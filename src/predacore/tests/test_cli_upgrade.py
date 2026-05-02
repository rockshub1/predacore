"""Tests for ``predacore upgrade`` CLI subcommand.

Three guarantees:
  1. dry-run prints the pip command but doesn't execute it (no network)
  2. --pre injects ``--pre`` into the pip args list
  3. wraps subprocess errors so a pip failure surfaces as a non-zero exit
     with a clear message rather than a Python traceback
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from predacore.cli import _run_upgrade


def test_upgrade_dry_run_does_not_call_subprocess(capsys):
    """--dry-run should print the pip command but skip execution."""
    with patch("predacore.cli._subprocess.check_call") as mock_call:
        _run_upgrade(dry_run=True)
        mock_call.assert_not_called()
    out = capsys.readouterr().out
    assert "pip install -U" in out
    assert "predacore" in out
    assert "predacore_core" in out
    assert "dry-run" in out


def test_upgrade_executes_pip_when_not_dry_run():
    """Real run calls subprocess.check_call with the right command shape."""
    with patch("predacore.cli._subprocess.check_call") as mock_call:
        _run_upgrade(dry_run=False)
        mock_call.assert_called_once()
        cmd = mock_call.call_args.args[0]
        assert cmd[0].endswith(("python", "python3"))
        assert "pip" in cmd
        assert "install" in cmd
        assert "-U" in cmd
        assert "predacore" in cmd
        assert "predacore_core" in cmd
        assert "--no-cache-dir" in cmd


def test_upgrade_pre_adds_flag():
    """--pre should add the ``--pre`` argument to pip install."""
    with patch("predacore.cli._subprocess.check_call") as mock_call:
        _run_upgrade(allow_pre=True, dry_run=False)
        cmd = mock_call.call_args.args[0]
        assert "--pre" in cmd


def test_upgrade_pip_failure_exits_with_code():
    """pip failure surfaces as sys.exit(N), not a raw traceback."""
    import subprocess as _sp
    with patch(
        "predacore.cli._subprocess.check_call",
        side_effect=_sp.CalledProcessError(returncode=42, cmd=["pip"]),
    ):
        with pytest.raises(SystemExit) as exc_info:
            _run_upgrade(dry_run=False)
    assert exc_info.value.code == 42
