"""PKCE (RFC 7636) verifier + challenge generation.

Used by OAuth 2.1 / 2.0+PKCE flows. The verifier is a high-entropy
secret kept on the client; the challenge is its SHA-256 hash sent up
front in the authorization request. After the user approves, we send
the verifier to the token endpoint — the server hashes it and matches
against the challenge it stored. Defeats authorization-code interception
attacks because an attacker who only sees the redirect can't redeem it
without the verifier.

Implementation notes:

  * RFC 7636 §4.1 specifies length 43-128 chars from
    ``[A-Z][a-z][0-9]-._~`` — base64url-safe. ``secrets.token_urlsafe(64)``
    yields ~86 chars in that alphabet, so we trim trailing ``=`` padding
    and slice to a safe length.
  * S256 (recommended) is the only method we generate. The plain method
    is allowed by the RFC but explicitly NOT secure — we never use it.
"""
from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass


@dataclass(frozen=True)
class PKCEPair:
    """Verifier (kept secret) + challenge (sent in auth URL)."""
    verifier: str
    challenge: str
    method: str = "S256"


def generate_pkce_pair(verifier_length: int = 64) -> PKCEPair:
    """Build a fresh PKCE pair for one OAuth round-trip.

    ``verifier_length`` is the byte count fed to ``secrets.token_urlsafe``
    — the resulting URL-safe verifier is roughly ``ceil(length * 4/3)``
    chars. Default 64 → ~86 chars, comfortably inside RFC 7636's 43-128
    range with extra entropy.
    """
    if verifier_length < 32:
        # Below 32 bytes the RFC's 43-char minimum is at risk; bump.
        verifier_length = 32
    verifier = secrets.token_urlsafe(verifier_length).rstrip("=")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return PKCEPair(verifier=verifier, challenge=challenge, method="S256")
