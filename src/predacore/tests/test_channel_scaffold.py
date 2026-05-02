"""Tests for the channel scaffold generator (T10a).

Five guarantees:
  1. invalid channel_name → ValueError
  2. existing target file + overwrite=False → FileExistsError
  3. happy path writes a file that can be IMPORTED
  4. the imported class has the correct channel_name and inherits ChannelAdapter
  5. token-from-env wiring works (the SCAFFOLDED adapter reads its token from
     the right env var without us touching the file)
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from predacore.channels.scaffold import generate_scaffold
from predacore.gateway import ChannelAdapter


def _import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_invalid_name_raises(tmp_path):
    with pytest.raises(ValueError):
        generate_scaffold("Bad-Name", plugin_dir=tmp_path)
    with pytest.raises(ValueError):
        generate_scaffold("123starts_with_digit", plugin_dir=tmp_path)
    with pytest.raises(ValueError):
        generate_scaffold("", plugin_dir=tmp_path)


def test_overwrite_protection(tmp_path):
    target = generate_scaffold("matrix", plugin_dir=tmp_path)
    assert target.exists()
    with pytest.raises(FileExistsError):
        generate_scaffold("matrix", plugin_dir=tmp_path)
    # overwrite=True succeeds and produces fresh content
    target2 = generate_scaffold("matrix", plugin_dir=tmp_path, overwrite=True)
    assert target2 == target


def test_scaffolded_module_is_importable(tmp_path):
    target = generate_scaffold("mattermost", plugin_dir=tmp_path)
    module = _import_module_from_path("scaffold_mattermost", target)
    cls = module.MattermostAdapter
    assert cls.channel_name == "mattermost"
    assert issubclass(cls, ChannelAdapter)


def test_class_name_handles_underscores(tmp_path):
    target = generate_scaffold("ms_teams", plugin_dir=tmp_path)
    module = _import_module_from_path("scaffold_ms_teams", target)
    # Snake → Pascal: "ms_teams" → "MsTeamsAdapter"
    assert hasattr(module, "MsTeamsAdapter")
    assert module.MsTeamsAdapter.channel_name == "ms_teams"


def test_token_read_from_env(tmp_path, monkeypatch):
    """The scaffolded adapter pulls its token from the auto-derived env var.
    Verifies that setting MATRIX_TOKEN before instantiation populates _token,
    so the user only has to fill the 3 TODOs — token plumbing is automatic."""
    target = generate_scaffold("matrix", plugin_dir=tmp_path)
    module = _import_module_from_path("scaffold_matrix_env", target)
    monkeypatch.setenv("MATRIX_TOKEN", "tok-abc")
    # Minimal duck-typed config — the scaffold reads getattr(config.channels, name).
    config = SimpleNamespace(channels=SimpleNamespace(matrix=None))
    adapter = module.MatrixAdapter(config)
    assert adapter._token == "tok-abc"


def test_capabilities_dict_shape(tmp_path):
    """Capabilities must be a dict with the standard supports_* + max_message_length
    keys — gateway code does dict-style lookup on these, not attribute access."""
    target = generate_scaffold("rocket", plugin_dir=tmp_path)
    module = _import_module_from_path("scaffold_rocket_caps", target)
    caps = module.RocketAdapter.channel_capabilities
    assert isinstance(caps, dict)
    assert "supports_media" in caps
    assert "supports_markdown" in caps
    assert "max_message_length" in caps
    assert isinstance(caps["max_message_length"], int)


# ── /channel slash command (T10b) ────────────────────────────────────


def _gateway_for_test(tmp_path, enabled: list[str] | None = None):
    """Build a real Gateway with a real PredaCoreConfig pointed at tmp_path."""
    from unittest.mock import AsyncMock
    from predacore.config import ChannelConfig, PredaCoreConfig
    from predacore.gateway import Gateway

    config = PredaCoreConfig(
        name="test",
        home_dir=str(tmp_path),
        channels=ChannelConfig(enabled=enabled or []),
    )
    return Gateway(config=config, process_fn=AsyncMock(return_value="ok"))


def test_channel_slash_list_returns_enabled_and_available(tmp_path):
    """`/channel list` enumerates enabled + discoverable adapters."""
    gw = _gateway_for_test(tmp_path, enabled=["telegram"])
    out = gw._handle_channel_command("/channel list", "user1", "cli")
    assert out is not None
    text = out.text
    assert "Channels" in text
    assert "telegram" in text
    # At least one of the always-discovered built-ins should be listed
    assert "discord" in text or "slack" in text


def test_channel_slash_info_lists_required_secrets(tmp_path):
    """`/channel info telegram` describes the env var the user must set."""
    gw = _gateway_for_test(tmp_path)
    out = gw._handle_channel_command("/channel info telegram", "u", "cli")
    assert out is not None
    assert "TELEGRAM_BOT_TOKEN" in out.text
    assert "BotFather" in out.text


def test_channel_slash_scaffold_writes_file(tmp_path):
    """`/channel scaffold matrix` writes ~/.predacore/channels/matrix.py."""
    gw = _gateway_for_test(tmp_path)
    out = gw._handle_channel_command("/channel scaffold matrix", "u", "cli")
    assert out is not None
    assert "Wrote" in out.text or "wrote" in out.text.lower()
    assert (tmp_path / "channels" / "matrix.py").is_file()


def test_channel_slash_returns_none_for_non_channel_text(tmp_path):
    """Plain user messages are passed through (handler returns None)."""
    gw = _gateway_for_test(tmp_path)
    assert gw._handle_channel_command("hello there", "u", "cli") is None
    # /channelXYZ shouldn't match — must be exact /channel or /channel <args>
    assert gw._handle_channel_command("/channelxyz list", "u", "cli") is None
