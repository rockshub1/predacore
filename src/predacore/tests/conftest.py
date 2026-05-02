"""Project-level pytest fixtures for the PredaCore test suite.

The single most important thing this file does: **block real DNS by default**.

After PR 5 (per-redirect SSRF validation), every HTTP path runs through
``validate_url_ssrf_async`` → ``socket.getaddrinfo``. Tests that exercise
those paths without explicitly mocking DNS will issue real lookups, which
is slow at best and can hang for minutes on captive-network or firewalled
CI. We mock ``socket.getaddrinfo`` by default so a leaked-DNS test fails
loudly instead of silently stalling.

Tests that genuinely need real network can opt out::

    @pytest.mark.real_network
    def test_something_against_real_dns():
        ...
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "real_network: test legitimately uses real DNS / network. "
        "Skip the auto-mock; reviewer must justify in the PR description.",
    )


@pytest.fixture(autouse=True)
def _block_real_dns(request: pytest.FixtureRequest):
    """Block ``socket.getaddrinfo`` everywhere unless ``real_network`` marker is set.

    The mock returns a fixed public IP so SSRF validators see a "safe"
    address and proceed. Tests that want to assert SSRF *blocks* a private
    IP must set their own mock inside the test body — that override wins
    over this autouse fixture because it's a deeper context manager scope.
    """
    if request.node.get_closest_marker("real_network"):
        yield None
        return

    # Patch the import sites we know about. Each ``patch`` is independent —
    # if a module hasn't imported ``socket`` yet, the patch is a no-op,
    # which is fine.
    public_ip = [(2, 1, 6, "", ("93.184.216.34", 0))]

    with (
        patch("predacore.auth.security.socket.getaddrinfo",
              return_value=public_ip) as security_mock,
        patch("socket.getaddrinfo",
              return_value=public_ip) as global_mock,
    ):
        yield (security_mock, global_mock)
