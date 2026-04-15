import pytest

import jarvis._vendor.ethical_governance_module.rule_engine as rule_engine_module
from jarvis._vendor.ethical_governance_module.rule_engine import BasicRuleEngine


@pytest.fixture
def rule_engine():
    return BasicRuleEngine()

def test_check_compliance_email_non_approved_domain(rule_engine: BasicRuleEngine):
    """Test SEND_EMAIL action to a non-approved domain is blocked."""
    item = {
        "action_type": "SEND_EMAIL",
        "id": "email-001",
        "parameters": {
            "to": "user@unapproved.com",
            "subject": "Test",
            "body": "Hello"
        }
    }
    context = {"approved_domains": ["example.com", "company.org"]}
    result = rule_engine.check_compliance(item, context)
    assert result.is_compliant is False
    assert any("non-approved email domain" in v.description for v in result.violations)

def test_check_compliance_slack_non_approved_channel(rule_engine: BasicRuleEngine):
    """Test SLACK_BOT action to a non-approved channel is blocked."""
    item = {
        "action_type": "SLACK_BOT",
        "id": "slack-001",
        "parameters": {
            "channel": "#random",
            "message": "This is a test"
        }
    }
    context = {"approved_slack_channels": ["#general", "#alerts"]}
    result = rule_engine.check_compliance(item, context)
    assert result.is_compliant is False
    assert any("non-approved Slack channel" in v.description for v in result.violations)


def test_modify_filesystem_outside_allowed(monkeypatch, tmp_path):
    """MODIFY_FILESYSTEM outside allowed base should be blocked."""
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    monkeypatch.setattr(rule_engine_module, "DEFAULT_ALLOWED_FS_BASE_PATH", str(allowed_dir))
    engine = rule_engine_module.BasicRuleEngine()
    item = {
        "action_type": "MODIFY_FILESYSTEM",
        "id": "fs-001",
        "parameters": {"path": str(tmp_path / "other" / "data.txt")},
    }
    result = engine.check_compliance(item, {})
    assert result.is_compliant is False
    assert any("outside allowed path" in v.description for v in result.violations)


def test_modify_filesystem_missing_path(monkeypatch, tmp_path):
    """MODIFY_FILESYSTEM without path should be blocked."""
    monkeypatch.setattr(rule_engine_module, "DEFAULT_ALLOWED_FS_BASE_PATH", str(tmp_path))
    engine = rule_engine_module.BasicRuleEngine()
    item = {"action_type": "MODIFY_FILESYSTEM", "id": "fs-002", "parameters": {}}
    result = engine.check_compliance(item, {})
    assert result.is_compliant is False
    assert any("Missing 'path'" in v.description for v in result.violations)


def test_browser_automation_non_approved_domain(rule_engine: BasicRuleEngine):
    """BROWSER_AUTOMATION to non-approved domain should be blocked."""
    item = {
        "action_type": "BROWSER_AUTOMATION",
        "id": "browser-001",
        "parameters": {"url": "https://evil.com/page"},
    }
    context = {"approved_domains": ["example.com", "trusted.org"]}
    result = rule_engine.check_compliance(item, context)
    assert result.is_compliant is False
    assert any("non-approved domain" in v.description for v in result.violations)


def test_no_violations_returns_compliant(rule_engine: BasicRuleEngine):
    """A benign item should be compliant."""
    item = {
        "description": "Summarize document",
        "action_type": "GENERIC_PROCESS",
        "id": "ok-001",
    }
    result = rule_engine.check_compliance(item, {})
    assert result.is_compliant is True
    assert result.violations == []
