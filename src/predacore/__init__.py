"""
PredaCore — The unified PredaCore AI agent.

Wraps the entire PredaCore stack (CSC, DAF, WIL, EGM, KN) into a
single-process conversational agent that can be run with:

    predacore chat     – interactive terminal session
    predacore start    – 24/7 daemon mode
    predacore setup    – guided onboarding wizard
"""
# v1.6.2 fix: read version from installed package metadata so
# `predacore --version` is ALWAYS truthful, even after a pip upgrade.
# Previously this was a hardcoded string that drifted across releases
# (got stuck at 1.5.6 through 1.5.7, 1.6.0, 1.6.1). The metadata path
# is canonical and updated by pip atomically with the install itself.
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("predacore")
except Exception:  # noqa: BLE001 — source install or metadata stripped
    # Fallback for git checkouts or unusual install layouts. Bump this
    # in lockstep with pyproject.toml version so it stays roughly current
    # for source-install users (who are rare and tend to know git).
    __version__ = "1.6.8"
