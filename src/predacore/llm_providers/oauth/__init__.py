"""OAuth flow primitives for LLM providers.

Used today by OpenAI Codex (subscription-based access via PKCE) and
designed to be the shared substrate when we add Gemini direct OAuth or
similar provider flows. See ``providers/openai_codex.py`` for the
consumer.

The package is small on purpose:

  ``pkce``      — RFC 7636 verifier + S256 challenge generation
  ``callback``  — aiohttp listener that captures the OAuth code from
                  ``http://localhost:PORT/callback``
  ``store``     — atomic + chmod 600 + per-provider JSON file under
                  ``~/.predacore/oauth/``
  ``refresh``   — auto-refresh access tokens when they're within 5
                  minutes of expiry; serialized via file lock so two
                  concurrent agents don't double-refresh
  ``flow``      — full end-to-end PKCE orchestration: build URL → open
                  browser → wait for callback → exchange for tokens →
                  store
"""
from __future__ import annotations

from .callback import wait_for_authorization_code
from .flow import OAuthFlow, OAuthFlowConfig, OAuthGrant
from .pkce import generate_pkce_pair
from .refresh import refresh_access_token, with_auto_refresh
from .store import OAuthGrantStore, load_grant, save_grant

__all__ = [
    "generate_pkce_pair",
    "wait_for_authorization_code",
    "OAuthFlow",
    "OAuthFlowConfig",
    "OAuthGrant",
    "OAuthGrantStore",
    "load_grant",
    "save_grant",
    "refresh_access_token",
    "with_auto_refresh",
]
