"""
Initial implementation of the Rule Engine for the EGM.
"""
import logging
import os
from typing import Any, Optional
from urllib.parse import urlparse

from jarvis._vendor.common.protos import egm_pb2

# Define a simple list of forbidden keywords for demonstration
FORBIDDEN_KEYWORDS = {
    "delete_user_data",
    "disable_safety",
    "ignore_ethics",
    "harm_human",
    "overwrite_system_files",
}

# Define potentially risky action types (examples)
RISKY_ACTION_TYPES = {
    "EXECUTE_ARBITRARY_CODE",
    "MODIFY_FILESYSTEM",
    "SEND_EMAIL",  # Could be misused
    "ACCESS_SENSITIVE_DATA",
}

DEFAULT_ALLOWED_FS_BASE_PATH = os.getenv("ALLOWED_FS_BASE_PATH", os.getcwd())


class BasicRuleEngine:
    """
    A very basic rule engine that checks for forbidden keywords and simple policy constraints.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("BasicRuleEngine initialized.")

    def _violation(
        self,
        principle: str,
        severity: egm_pb2.SeverityLevel,
        description: str,
        component_source: str,
    ) -> egm_pb2.EthicalViolationMessage:
        return egm_pb2.EthicalViolationMessage(
            principle_violated=principle,
            severity=severity,
            description=description,
            component_source=component_source,
        )

    def check_compliance(
        self, item_to_check: Any, context: dict | None = None
    ) -> egm_pb2.ComplianceCheckResultMessage:
        """
        Checks a plan step or action description for forbidden keywords and basic policy rules.
        """
        context = context or {}
        text_to_check = ""
        action_type = None
        item_id = "unknown_item"
        parameters: dict[str, Any] = {}

        if hasattr(item_to_check, "description"):
            text_to_check = getattr(item_to_check, "description", "") or ""
            action_type = getattr(item_to_check, "action_type", None)
            item_id = getattr(item_to_check, "id", item_id)
        elif isinstance(item_to_check, dict):
            text_to_check = (
                item_to_check.get(
                    "description",
                    item_to_check.get("action", item_to_check.get("command", "")),
                )
                or ""
            )
            action_type = item_to_check.get("action_type")
            item_id = item_to_check.get("id", item_id)
            params = item_to_check.get("parameters")
            if isinstance(params, dict):
                parameters = params
        elif isinstance(item_to_check, str):
            text_to_check = item_to_check

        self.logger.debug(
            f"Checking compliance for item '{item_id}', text: '{text_to_check[:100]}...', action_type={action_type}"
        )

        violations: list[egm_pb2.EthicalViolationMessage] = []
        warnings: list[str] = []

        # 1. Forbidden keywords in description/text
        if text_to_check:
            text_lower = text_to_check.lower()
            for keyword in FORBIDDEN_KEYWORDS:
                if keyword.lower() in text_lower:
                    violations.append(
                        self._violation(
                            "NonMaleficence",
                            egm_pb2.SeverityLevel.SEVERITY_HIGH,
                            f"Detected forbidden keyword: '{keyword}'",
                            f"RuleEngineCheck({item_id})",
                        )
                    )
                    self.logger.warning(
                        f"Compliance violation (keyword) for item '{item_id}': '{keyword}'"
                    )

        # 2. Risky action types
        if action_type and action_type in RISKY_ACTION_TYPES:
            violations.append(
                self._violation(
                    "RiskMitigation",
                    egm_pb2.SeverityLevel.SEVERITY_MEDIUM,
                    f"Detected potentially risky action type: '{action_type}'",
                    f"RuleEngineCheck({item_id})",
                )
            )
            self.logger.warning(
                f"Compliance violation (action type) for item '{item_id}': '{action_type}'"
            )

        # 3. Parameterized rules
        # Filesystem mutations
        if action_type == "MODIFY_FILESYSTEM":
            target_path = parameters.get("path")
            if target_path:
                abs_target = os.path.abspath(target_path)
                abs_allowed = os.path.abspath(DEFAULT_ALLOWED_FS_BASE_PATH)
                if not abs_target.startswith(abs_allowed):
                    violations.append(
                        self._violation(
                            "Safety",
                            egm_pb2.SeverityLevel.SEVERITY_HIGH,
                            f"Attempt to modify filesystem outside allowed path: '{target_path}' (resolved to '{abs_target}')",
                            f"RuleEngineParamCheck({item_id})",
                        )
                    )
                    self.logger.error(
                        f"Path '{target_path}' outside allowed base '{DEFAULT_ALLOWED_FS_BASE_PATH}'"
                    )
            else:
                violations.append(
                    self._violation(
                        "Safety",
                        egm_pb2.SeverityLevel.SEVERITY_MEDIUM,
                        "Missing 'path' parameter for MODIFY_FILESYSTEM action.",
                        f"RuleEngineParamCheck({item_id})",
                    )
                )
                warnings.append("Missing path parameter for filesystem modification.")

        # Email domain allowlist
        if action_type == "SEND_EMAIL":
            to_addr = parameters.get("to", "")
            approved_domains = context.get("approved_domains", [])
            if to_addr and approved_domains:
                domain = to_addr.split("@")[-1].lower()
                if not any(domain == d.lower() for d in approved_domains):
                    violations.append(
                        self._violation(
                            "DataPrivacy",
                            egm_pb2.SeverityLevel.SEVERITY_HIGH,
                            f"Attempt to send email to non-approved email domain: '{domain}'",
                            f"RuleEngineParamCheck({item_id})",
                        )
                    )
                    self.logger.warning(
                        f"Email domain '{domain}' not approved for item '{item_id}'"
                    )

        # Slack channel allowlist
        if action_type == "SLACK_BOT":
            channel = parameters.get("channel", "")
            approved_channels = context.get("approved_slack_channels", [])
            if channel and approved_channels and channel not in approved_channels:
                violations.append(
                    self._violation(
                        "DataPrivacy",
                        egm_pb2.SeverityLevel.SEVERITY_MEDIUM,
                        f"Attempt to send Slack message to non-approved Slack channel: '{channel}'",
                        f"RuleEngineParamCheck({item_id})",
                    )
                )
                self.logger.warning(
                    f"Slack channel '{channel}' not approved for item '{item_id}'"
                )

        # Browser domain allowlist
        if action_type == "BROWSER_AUTOMATION":
            url = parameters.get("url", "")
            approved_domains = context.get("approved_domains", [])
            if url and approved_domains:
                domain = urlparse(url).netloc.lower()
                if not any(
                    domain == d.lower() or domain.endswith(f".{d.lower()}")
                    for d in approved_domains
                ):
                    violations.append(
                        self._violation(
                            "DataPrivacy",
                            egm_pb2.SeverityLevel.SEVERITY_MEDIUM,
                            f"Browser navigation to non-approved domain: '{domain}'",
                            f"RuleEngineParamCheck({item_id})",
                        )
                    )
                    self.logger.warning(
                        f"Browser domain '{domain}' not approved for item '{item_id}'"
                    )

        if violations:
            justification = "Action or step failed compliance checks."
            return egm_pb2.ComplianceCheckResultMessage(
                is_compliant=False,
                violations=violations,
                warnings=warnings,
                justification=justification,
            )

        return egm_pb2.ComplianceCheckResultMessage(
            is_compliant=True,
            warnings=warnings,
            justification="No compliance violations detected.",
        )
