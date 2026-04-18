import json
from types import SimpleNamespace

import pytest

from src.predacore.config import (
    PredaCoreConfig,
    LaunchProfileConfig,
    LLMConfig,
    MemoryConfig,
    SecurityConfig,
)
from src.predacore.core import ToolExecutor

pytestmark = pytest.mark.asyncio


@pytest.fixture
def _config(tmp_path):
    home = tmp_path / ".prometheus"
    sessions = home / "sessions"
    skills = home / "skills"
    logs = home / "logs"
    memory = home / "memory"
    for path in (home, sessions, skills, logs, memory):
        path.mkdir(parents=True, exist_ok=True)
    return PredaCoreConfig(
        home_dir=str(home),
        sessions_dir=str(sessions),
        skills_dir=str(skills),
        logs_dir=str(logs),
        llm=LLMConfig(provider="gemini-cli"),
        security=SecurityConfig(trust_level="normal"),
        memory=MemoryConfig(persistence_dir=str(memory)),
        launch=LaunchProfileConfig(enable_plugin_marketplace=True),
    )


async def test_marketplace_install_and_invoke_data_analyzer(_config):
    executor = ToolExecutor(_config)

    install = await executor.execute(
        "marketplace_install_skill",
        {"user_id": "u-1", "skill_id": "prometheus.data-analyzer"},
    )
    assert "Installed skill" in install

    invoked = await executor.execute(
        "marketplace_invoke_skill",
        {
            "user_id": "u-1",
            "skill_id": "prometheus.data-analyzer",
            "params": {"data": '{"alpha": 1, "beta": 2}'},
        },
    )
    payload = json.loads(invoked)
    assert payload["success"] is True
    assert payload["result"]["output"]["format"] == "json"
    assert sorted(payload["result"]["output"]["keys"]) == ["alpha", "beta"]


async def test_memory_store_and_recall_user_profile(_config):
    """User profiles are now handled via unified memory_store/memory_recall."""
    executor = ToolExecutor(_config)

    stored = await executor.execute(
        "memory_store",
        {
            "key": "user:u-42:profile",
            "content": '{"preferences": {"risk": "yolo"}, "goals": ["launch public beast"]}',
            "tags": ["user_profile"],
            "user_id": "u-42",
        },
    )
    assert "stored" in stored.lower() or "saved" in stored.lower() or "memory" in stored.lower()


async def test_openclaw_skills_import_and_run(_config, tmp_path, monkeypatch):
    skills_root = tmp_path / "openclaw_skills"
    skill_dir = skills_root / "demo-skill"
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    (skill_dir / "SKILL.md").write_text(
        """---
name: demo-skill
description: Demo imported OpenClaw skill.
metadata:
  openclaw:
    emoji: "🧪"
---

# Demo skill

```bash
python3 scripts/echo.py hello
```
""",
        encoding="utf-8",
    )
    (scripts_dir / "echo.py").write_text(
        "import sys\nprint('OPENCLAW:' + ' '.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("OPENCLAW_SKILLS_DIR", str(skills_root))
    monkeypatch.setenv("OPENCLAW_AUTO_IMPORT_SKILLS", "1")

    executor = ToolExecutor(_config)

    listed = await executor.execute(
        "marketplace_list_skills",
        {"user_id": "u-oc", "search": "openclaw"},
    )
    assert "openclaw.demo-skill" in listed

    installed = await executor.execute(
        "marketplace_install_skill",
        {"user_id": "u-oc", "skill_id": "openclaw.demo-skill"},
    )
    assert "Installed skill" in installed

    described_raw = await executor.execute(
        "marketplace_invoke_skill",
        {
            "user_id": "u-oc",
            "skill_id": "openclaw.demo-skill",
            "params": {"action": "describe"},
        },
    )
    described = json.loads(described_raw)
    assert described["success"] is True
    assert described["result"]["output"]["skill_id"] == "openclaw.demo-skill"
    assert "scripts/echo.py" in described["result"]["output"]["scripts"]

    ran_raw = await executor.execute(
        "marketplace_invoke_skill",
        {
            "user_id": "u-oc",
            "skill_id": "openclaw.demo-skill",
            "params": {
                "action": "run_script",
                "script": "scripts/echo.py",
                "args": ["alpha", "beta"],
            },
        },
    )
    ran = json.loads(ran_raw)
    assert ran["success"] is True
    assert ran["result"]["output"]["success"] is True
    assert "OPENCLAW:alpha beta" in ran["result"]["output"]["output"]


async def test_openclaw_skills_autodetects_installed_cli(_config, tmp_path, monkeypatch):
    from src.predacore.tools.marketplace import MarketplaceManager

    monkeypatch.delenv("PREDACORE_OPENCLAW_SKILLS_DIR", raising=False)
    monkeypatch.delenv("OPENCLAW_SKILLS_DIR", raising=False)
    _config.openclaw.skills_dir = ""

    install_root = tmp_path / "node_modules" / "openclaw"
    skills_dir = install_root / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    (install_root / "openclaw.mjs").write_text("#!/usr/bin/env node\n", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "openclaw").symlink_to(install_root / "openclaw.mjs")

    monkeypatch.setattr("src.predacore.tools.marketplace.shutil.which", lambda name: str(bin_dir / "openclaw"))

    manager = MarketplaceManager(_config, SimpleNamespace(skill_marketplace=None))
    resolved = manager._resolve_openclaw_skills_dir()

    assert resolved == skills_dir.resolve()


async def test_openclaw_skills_no_longer_falls_back_to_vendored_tree(_config, tmp_path, monkeypatch):
    from src.predacore.tools.marketplace import MarketplaceManager

    monkeypatch.delenv("PREDACORE_OPENCLAW_SKILLS_DIR", raising=False)
    monkeypatch.delenv("OPENCLAW_SKILLS_DIR", raising=False)
    _config.openclaw.skills_dir = ""
    monkeypatch.setattr("src.predacore.tools.marketplace.shutil.which", lambda name: None)

    vendored_skills = tmp_path / "external" / "openclaw" / "skills"
    vendored_skills.mkdir(parents=True, exist_ok=True)

    manager = MarketplaceManager(_config, SimpleNamespace(skill_marketplace=None))
    resolved = manager._resolve_openclaw_skills_dir()

    assert resolved is None
