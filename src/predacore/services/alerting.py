"""
Alerting — PagerDuty, Slack, Email, Discord, and webhook alerting for PredaCore.

Provides a unified alerting interface that dispatches to multiple
notification channels when critical events occur.
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import socket
import time
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Optional
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

logger = logging.getLogger("predacore.services.alerting")


# ── Data Models ──────────────────────────────────────────────────────


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    RESOLVED = "resolved"


class AlertChannel(str, Enum):
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    EMAIL = "email"
    DISCORD = "discord"
    LOG = "log"


@dataclass
class Alert:
    """An individual alert event."""

    title: str
    message: str
    severity: AlertSeverity = AlertSeverity.WARNING
    source: str = "predacore"
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    dedup_key: str = ""  # For PagerDuty deduplication

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source,
            "labels": self.labels,
            "timestamp": self.timestamp,
            "dedup_key": self.dedup_key,
        }


# ── Dispatchers ──────────────────────────────────────────────────────


class SlackDispatcher:
    """Send alerts to a Slack channel via webhook."""

    SEVERITY_COLORS = {
        AlertSeverity.INFO: "#36a64f",
        AlertSeverity.WARNING: "#ffcc00",
        AlertSeverity.CRITICAL: "#ff0000",
        AlertSeverity.RESOLVED: "#2eb886",
    }

    SEVERITY_EMOJI = {
        AlertSeverity.INFO: "\u2139\ufe0f",
        AlertSeverity.WARNING: "\u26a0\ufe0f",
        AlertSeverity.CRITICAL: "\U0001f6a8",
        AlertSeverity.RESOLVED: "\u2705",
    }

    def __init__(self, webhook_url: str = "") -> None:
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL", "")

    @property
    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    def send(self, alert: Alert) -> bool:
        """Send an alert to Slack."""
        if not self.is_configured:
            logger.warning("Slack webhook not configured")
            return False

        emoji = self.SEVERITY_EMOJI.get(alert.severity, "\U0001f4e2")
        color = self.SEVERITY_COLORS.get(alert.severity, "#cccccc")

        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"{emoji}  {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {"title": k, "value": v, "short": True}
                        for k, v in alert.labels.items()
                    ],
                    "footer": f"PredaCore | {alert.source}",
                    "ts": int(alert.timestamp),
                }
            ],
        }

        return self._post(payload)

    def _post(self, payload: dict) -> bool:
        try:
            data = json.dumps(payload).encode("utf-8")
            req = Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urlopen(req, timeout=5)
            return True
        except (URLError, OSError) as e:
            logger.error("Slack dispatch failed: %s", e)
            return False


class PagerDutyDispatcher:
    """Send alerts to PagerDuty Events API v2."""

    EVENTS_URL = "https://events.pagerduty.com/v2/enqueue"

    SEVERITY_MAP = {
        AlertSeverity.INFO: "info",
        AlertSeverity.WARNING: "warning",
        AlertSeverity.CRITICAL: "critical",
        AlertSeverity.RESOLVED: "info",
    }

    def __init__(self, routing_key: str = "") -> None:
        self.routing_key = routing_key or os.getenv("PAGERDUTY_ROUTING_KEY", "")

    @property
    def is_configured(self) -> bool:
        return bool(self.routing_key)

    def send(self, alert: Alert) -> bool:
        """Send an alert to PagerDuty."""
        if not self.is_configured:
            logger.warning("PagerDuty routing key not configured")
            return False

        event_action = (
            "resolve" if alert.severity == AlertSeverity.RESOLVED else "trigger"
        )

        payload = {
            "routing_key": self.routing_key,
            "event_action": event_action,
            "dedup_key": alert.dedup_key or f"predacore-{alert.title}",
            "payload": {
                "summary": f"[{alert.severity.value.upper()}] {alert.title}: {alert.message}",
                "severity": self.SEVERITY_MAP.get(alert.severity, "warning"),
                "source": alert.source,
                "component": "predacore",
                "custom_details": alert.labels,
            },
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = Request(
                self.EVENTS_URL,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urlopen(req, timeout=5)
            return True
        except (URLError, Exception) as e:
            logger.error(f"PagerDuty dispatch failed: {e}")
            return False


class WebhookDispatcher:
    """Send alerts to a generic webhook endpoint."""

    def __init__(self, url: str = "", headers: dict[str, str] | None = None) -> None:
        self.url = url or os.getenv("ALERT_WEBHOOK_URL", "")
        self.headers = headers or {}

    @property
    def is_configured(self) -> bool:
        return bool(self.url)

    def send(self, alert: Alert) -> bool:
        if not self.is_configured:
            return False
        return self._post(alert.to_dict())

    def _post(self, payload: dict) -> bool:
        if not _is_safe_url(self.url):
            logger.warning("Blocked webhook to private/internal URL: %s", self.url)
            return False
        try:
            data = json.dumps(payload).encode("utf-8")
            h = {"Content-Type": "application/json", **self.headers}
            req = Request(self.url, data=data, headers=h)
            urlopen(req, timeout=5)
            return True
        except (OSError, ConnectionError, ValueError) as e:
            logger.error(f"Webhook dispatch failed: {e}")
            return False


def _is_safe_url(url: str) -> bool:
    """Block requests to private/internal IP ranges (SSRF prevention)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    try:
        for info in socket.getaddrinfo(parsed.hostname, None):
            addr = info[4][0]
            _BLOCKED = (
                "10.", "172.16.", "172.17.", "172.18.", "172.19.",
                "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
                "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
                "172.30.", "172.31.", "192.168.", "127.", "0.", "169.254.",
            )
            if any(addr.startswith(p) for p in _BLOCKED):
                return False
            # Also check IPv6 loopback
            if addr == "::1":
                return False
    except socket.gaierror:
        return False
    return True


class EmailDispatcher:
    """Send alerts via email (SMTP with TLS)."""

    def __init__(
        self,
        smtp_host: str = "",
        smtp_port: int = 587,
        from_addr: str = "",
        to_addrs: list[str] | None = None,
        username: str = "",
        password: str = "",
    ) -> None:
        self.smtp_host = smtp_host or os.getenv("PREDACORE_ALERT_SMTP_HOST", "")
        self.smtp_port = int(os.getenv("PREDACORE_ALERT_SMTP_PORT", str(smtp_port)))
        self.from_addr = from_addr or os.getenv("PREDACORE_ALERT_SMTP_FROM", "")
        to_env = os.getenv("PREDACORE_ALERT_SMTP_TO", "")
        self.to_addrs = to_addrs or ([a.strip() for a in to_env.split(",") if a.strip()] if to_env else [])
        self.username = username or os.getenv("PREDACORE_ALERT_SMTP_USER", "")
        self.password = password or os.getenv("PREDACORE_ALERT_SMTP_PASS", "")

    @property
    def is_configured(self) -> bool:
        return bool(self.smtp_host and self.from_addr and self.to_addrs)

    def send(self, alert: Alert) -> bool:
        """Send an alert email."""
        if not self.is_configured:
            logger.warning("Email alerting not configured")
            return False

        severity = alert.severity.value.upper()
        subject = f"[{severity}] {alert.title}"

        label_lines = "".join(
            f"  {k}: {v}\n" for k, v in alert.labels.items()
        )
        body = (
            f"Severity: {severity}\n"
            f"Source:   {alert.source}\n"
            f"Time:     {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(alert.timestamp))}\n"
            f"\n{alert.message}\n"
        )
        if label_lines:
            body += f"\nLabels:\n{label_lines}"

        payload = {"subject": subject, "body": body}
        return self._post(payload)

    def _post(self, payload: dict) -> bool:
        try:
            msg = MIMEMultipart()
            msg["Subject"] = payload["subject"]
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)
            msg.attach(MIMEText(payload["body"], "plain"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            return True
        except (smtplib.SMTPException, OSError) as e:
            logger.error("Email dispatch failed: %s", e)
            return False


class DiscordDispatcher:
    """Send alerts to a Discord channel via webhook."""

    SEVERITY_COLORS = {
        AlertSeverity.INFO: 0x2ECC71,       # Green
        AlertSeverity.WARNING: 0xF1C40F,    # Yellow
        AlertSeverity.CRITICAL: 0xE74C3C,   # Red
        AlertSeverity.RESOLVED: 0x3498DB,   # Blue
    }

    def __init__(self, webhook_url: str = "") -> None:
        self.webhook_url = webhook_url or os.getenv("PREDACORE_ALERT_DISCORD_URL", "")

    @property
    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    def send(self, alert: Alert) -> bool:
        """Send an alert to Discord."""
        if not self.is_configured:
            logger.warning("Discord webhook not configured")
            return False

        color = self.SEVERITY_COLORS.get(alert.severity, 0x95A5A6)

        embed = {
            "title": f"[{alert.severity.value.upper()}] {alert.title}",
            "description": alert.message,
            "color": color,
            "footer": {"text": f"PredaCore | {alert.source}"},
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(alert.timestamp)
            ),
        }

        if alert.labels:
            embed["fields"] = [
                {"name": k, "value": v, "inline": True}
                for k, v in list(alert.labels.items())[:10]
            ]

        payload = {"embeds": [embed]}
        return self._post(payload)

    def _post(self, payload: dict) -> bool:
        try:
            data = json.dumps(payload).encode("utf-8")
            req = Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urlopen(req, timeout=5)
            return True
        except (URLError, OSError) as e:
            logger.error("Discord dispatch failed: %s", e)
            return False


# ── Alert Manager ────────────────────────────────────────────────────


class AlertManager:
    """
    Central alert management system.

    Dispatches alerts to configured channels, with:
    - Severity-based routing (e.g., only page for CRITICAL)
    - Deduplication within cooldown windows
    - Alert history and statistics
    """

    def __init__(
        self,
        slack_url: str = "",
        pagerduty_key: str = "",
        webhook_url: str = "",
        smtp_host: str = "",
        smtp_from: str = "",
        smtp_to: str = "",
        discord_url: str = "",
    ) -> None:
        # Parse comma-separated email recipients
        to_addrs = [a.strip() for a in smtp_to.split(",") if a.strip()] if smtp_to else None

        self._dispatchers: dict[AlertChannel, Any] = {
            AlertChannel.SLACK: SlackDispatcher(slack_url),
            AlertChannel.PAGERDUTY: PagerDutyDispatcher(pagerduty_key),
            AlertChannel.WEBHOOK: WebhookDispatcher(webhook_url),
            AlertChannel.EMAIL: EmailDispatcher(
                smtp_host=smtp_host or os.getenv("PREDACORE_ALERT_SMTP_HOST", ""),
                from_addr=smtp_from or os.getenv("PREDACORE_ALERT_SMTP_FROM", ""),
                to_addrs=to_addrs,
            ),
            AlertChannel.DISCORD: DiscordDispatcher(
                webhook_url=discord_url or os.getenv("PREDACORE_ALERT_DISCORD_URL", ""),
            ),
        }

        # Cooldown tracking: dedup_key -> last_sent_time
        self._cooldowns: dict[str, float] = {}
        self._cooldown_seconds: float = 300  # 5 minute default
        self._history: list[dict[str, Any]] = []
        self._max_history: int = 1000

        # Routing: which channels get which severities
        self._routing: dict[AlertSeverity, list[AlertChannel]] = {
            AlertSeverity.INFO: [
                AlertChannel.LOG,
                AlertChannel.SLACK,
                AlertChannel.DISCORD,
            ],
            AlertSeverity.WARNING: [
                AlertChannel.LOG,
                AlertChannel.SLACK,
                AlertChannel.DISCORD,
                AlertChannel.EMAIL,
            ],
            AlertSeverity.CRITICAL: [
                AlertChannel.LOG,
                AlertChannel.SLACK,
                AlertChannel.PAGERDUTY,
                AlertChannel.DISCORD,
                AlertChannel.EMAIL,
            ],
            AlertSeverity.RESOLVED: [
                AlertChannel.LOG,
                AlertChannel.SLACK,
                AlertChannel.PAGERDUTY,
                AlertChannel.DISCORD,
                AlertChannel.EMAIL,
            ],
        }

    def fire(self, alert: Alert) -> dict[str, bool]:
        """
        Fire an alert to configured channels.
        Returns dict of channel -> success.
        """
        dedup = alert.dedup_key or alert.title
        now = time.time()

        # Check cooldown (skip for RESOLVED)
        if alert.severity != AlertSeverity.RESOLVED:
            last_sent = self._cooldowns.get(dedup, 0)
            if now - last_sent < self._cooldown_seconds:
                logger.debug(f"Alert {dedup} still in cooldown")
                return {"cooldown": True}

        self._cooldowns[dedup] = now

        # Dispatch to channels
        channels = self._routing.get(alert.severity, [AlertChannel.LOG])
        results: dict[str, bool] = {}

        for channel in channels:
            if channel == AlertChannel.LOG:
                log_func = (
                    logger.critical
                    if alert.severity == AlertSeverity.CRITICAL
                    else logger.warning
                )
                log_func(
                    f"ALERT [{alert.severity.value}] {alert.title}: {alert.message}"
                )
                results["log"] = True
            else:
                dispatcher = self._dispatchers.get(channel)
                if dispatcher and dispatcher.is_configured:
                    results[channel.value] = dispatcher.send(alert)

        # Record history
        self._history.append(
            {
                **alert.to_dict(),
                "dispatched_to": list(results.keys()),
                "results": results,
            }
        )
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        # Evict expired cooldowns to prevent unbounded growth
        if len(self._cooldowns) > 100:
            cutoff = time.time() - self._cooldown_seconds
            self._cooldowns = {k: v for k, v in self._cooldowns.items() if v > cutoff}

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get alerting statistics."""
        severity_counts: dict[str, int] = {}
        for entry in self._history:
            sev = entry.get("severity", "unknown")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return {
            "total_alerts": len(self._history),
            "by_severity": severity_counts,
            "cooldown_seconds": self._cooldown_seconds,
            "channels_configured": {
                name.value: d.is_configured for name, d in self._dispatchers.items()
            },
        }

    def get_recent_alerts(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent alert history."""
        return self._history[-limit:]
