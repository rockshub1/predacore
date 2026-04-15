"""
Prometheus Notification Service - Multi-channel notification delivery

Supports Email, Slack, Discord, and Push notifications.
Inspired by Moltbot's proactive communication capabilities.
"""
import asyncio
import html as _html
import logging
import os
import socket
from abc import ABC, abstractmethod
from urllib.parse import urlparse
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

# Optional imports for different channels
try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Supported notification channels."""

    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    PUSH = "push"  # Web push notifications
    WEBHOOK = "webhook"  # Generic webhook


class NotificationPriority(Enum):
    """Notification priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Notification:
    """Represents a notification to be sent."""

    id: UUID = field(default_factory=uuid4)
    channel: NotificationChannel = NotificationChannel.SLACK
    recipient: str = ""  # Email, channel ID, webhook URL, etc.
    title: str = ""
    message: str = ""
    priority: NotificationPriority = NotificationPriority.NORMAL
    metadata: dict[str, Any] = field(default_factory=dict)
    user_id: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    sent_at: datetime | None = None
    status: str = "pending"  # pending, sent, failed
    error: str | None = None


class NotificationProvider(ABC):
    """Base class for notification providers."""

    @abstractmethod
    async def send(self, notification: Notification) -> bool:
        """Send a notification. Returns True on success."""
        pass


class SlackProvider(NotificationProvider):
    """Send notifications via Slack webhook."""

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        if not self.webhook_url:
            logger.warning("Slack webhook URL not configured")

    async def send(self, notification: Notification) -> bool:
        if not self.webhook_url or not _HTTPX_AVAILABLE:
            return False

        # Map priority to emoji
        priority_emoji = {
            NotificationPriority.LOW: "📋",
            NotificationPriority.NORMAL: "📌",
            NotificationPriority.HIGH: "⚠️",
            NotificationPriority.URGENT: "🚨",
        }

        emoji = priority_emoji.get(notification.priority, "📌")

        payload = {
            "text": f"{emoji} *{notification.title}*\n{notification.message}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} {notification.title}",
                    },
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": notification.message},
                },
            ],
        }

        # Add metadata as context if present
        if notification.metadata:
            context_text = " | ".join(
                f"*{k}:* {v}" for k, v in notification.metadata.items()
            )
            payload["blocks"].append(
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": context_text}],
                }
            )

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(self.webhook_url, json=payload)
                return resp.status_code == 200
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            return False


class DiscordProvider(NotificationProvider):
    """Send notifications via Discord webhook."""

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        if not self.webhook_url:
            logger.warning("Discord webhook URL not configured")

    async def send(self, notification: Notification) -> bool:
        if not self.webhook_url or not _HTTPX_AVAILABLE:
            return False

        # Map priority to color
        priority_colors = {
            NotificationPriority.LOW: 0x3498DB,  # Blue
            NotificationPriority.NORMAL: 0x2ECC71,  # Green
            NotificationPriority.HIGH: 0xF39C12,  # Orange
            NotificationPriority.URGENT: 0xE74C3C,  # Red
        }

        embed = {
            "title": notification.title,
            "description": notification.message,
            "color": priority_colors.get(notification.priority, 0x2ECC71),
            "timestamp": notification.created_at.isoformat(),
            "footer": {"text": f"Priority: {notification.priority.value}"},
        }

        # Add fields from metadata
        if notification.metadata:
            embed["fields"] = [
                {"name": k, "value": str(v), "inline": True}
                for k, v in list(notification.metadata.items())[:10]  # Max 10 fields
            ]

        payload = {"embeds": [embed]}

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(self.webhook_url, json=payload)
                return resp.status_code in (200, 204)
        except Exception as e:
            logger.error(f"Discord notification failed: {e}")
            return False


def _is_safe_url(url: str) -> bool:
    """Block requests to private/internal IP ranges (SSRF prevention)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    try:
        for info in socket.getaddrinfo(parsed.hostname, None):
            addr = info[4][0]
            _BLOCKED = ("10.", "172.16.", "172.17.", "172.18.", "172.19.",
                        "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
                        "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
                        "172.30.", "172.31.", "192.168.", "127.", "0.", "169.254.")
            if any(addr.startswith(p) for p in _BLOCKED):
                return False
    except socket.gaierror:
        return False
    return True


class WebhookProvider(NotificationProvider):
    """Send notifications to a generic webhook endpoint."""

    async def send(self, notification: Notification) -> bool:
        if not notification.recipient or not _HTTPX_AVAILABLE:
            return False

        if not _is_safe_url(notification.recipient):
            logger.warning("Blocked webhook to private/internal URL: %s", notification.recipient)
            return False

        payload = {
            "id": str(notification.id),
            "title": notification.title,
            "message": notification.message,
            "priority": notification.priority.value,
            "metadata": notification.metadata,
            "timestamp": notification.created_at.isoformat(),
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(notification.recipient, json=payload)
                return 200 <= resp.status_code < 300
        except Exception as e:
            logger.error(
                f"Webhook notification to {notification.recipient} failed: {e}"
            )
            return False


class EmailProvider(NotificationProvider):
    """Send notifications via email (SMTP or API)."""

    def __init__(
        self,
        smtp_host: str | None = None,
        smtp_port: int = 587,
        smtp_user: str | None = None,
        smtp_pass: str | None = None,
        sender_email: str | None = None,
    ):
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST")
        self.smtp_port = int(os.getenv("SMTP_PORT", smtp_port))
        self.smtp_user = smtp_user or os.getenv("SMTP_USER")
        self.smtp_pass = smtp_pass or os.getenv("SMTP_PASS")
        self.sender_email = sender_email or os.getenv(
            "SENDER_EMAIL", "prometheus@localhost"
        )

        if not self.smtp_host:
            logger.warning("SMTP not configured - email notifications disabled")

    async def send(self, notification: Notification) -> bool:
        if not self.smtp_host:
            logger.warning("Email notification skipped - SMTP not configured")
            return False

        # Using aiosmtplib would be ideal; fallback to sync smtplib in executor
        try:
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[Prometheus] {notification.title}"
            msg["From"] = self.sender_email
            msg["To"] = notification.recipient

            # HTML version of message
            safe_title = _html.escape(notification.title)
            safe_message = _html.escape(notification.message).replace(chr(10), '<br>')
            html = f"""
            <html>
            <body>
                <h2>{safe_title}</h2>
                <p>{safe_message}</p>
                <hr>
                <small>Priority: {notification.priority.value} | Sent: {notification.created_at.isoformat()}</small>
            </body>
            </html>
            """
            msg.attach(MIMEText(notification.message, "plain"))
            msg.attach(MIMEText(html, "html"))

            # Run SMTP in executor to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._send_email, msg)
            return True

        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return False

    def _send_email(self, msg):
        """Synchronous email send for use in executor."""
        import smtplib

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            if self.smtp_user and self.smtp_pass:
                server.login(self.smtp_user, self.smtp_pass)
            server.send_message(msg)


class NotificationService:
    """
    Main notification service for managing and sending notifications.
    """

    def __init__(self, history_path: str | None = None):
        self._providers: dict[NotificationChannel, NotificationProvider] = {}
        self._history: list[Notification] = []
        self._max_history = 1000
        self._history_path = (
            Path(history_path) if history_path else Path("data/notifications")
        )
        self._history_path.mkdir(parents=True, exist_ok=True)

        # Initialize default providers from environment
        self._init_default_providers()
        logger.info("NotificationService initialized")

    def _init_default_providers(self):
        """Initialize providers based on environment configuration."""
        # Slack
        if os.getenv("SLACK_WEBHOOK_URL"):
            self.register_provider(NotificationChannel.SLACK, SlackProvider())

        # Discord
        if os.getenv("DISCORD_WEBHOOK_URL"):
            self.register_provider(NotificationChannel.DISCORD, DiscordProvider())

        # Email
        if os.getenv("SMTP_HOST"):
            self.register_provider(NotificationChannel.EMAIL, EmailProvider())

        # Webhook is always available
        self.register_provider(NotificationChannel.WEBHOOK, WebhookProvider())

    def register_provider(
        self, channel: NotificationChannel, provider: NotificationProvider
    ):
        """Register a notification provider for a channel."""
        self._providers[channel] = provider
        logger.info(f"Registered notification provider for {channel.value}")

    async def send(self, notification: Notification) -> bool:
        """Send a notification through the appropriate channel."""
        provider = self._providers.get(notification.channel)

        if not provider:
            logger.warning(
                f"No provider registered for channel {notification.channel.value}"
            )
            notification.status = "failed"
            notification.error = f"No provider for {notification.channel.value}"
            self._add_to_history(notification)
            return False

        try:
            success = await provider.send(notification)
            notification.sent_at = datetime.utcnow()
            notification.status = "sent" if success else "failed"
            if not success:
                notification.error = "Provider returned failure"

        except Exception as e:
            logger.error(f"Notification send failed: {e}", exc_info=True)
            notification.status = "failed"
            notification.error = str(e)

        self._add_to_history(notification)
        return notification.status == "sent"

    async def send_multi(
        self,
        channels: list[NotificationChannel],
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        recipients: dict[NotificationChannel, str] | None = None,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> dict[NotificationChannel, bool]:
        """Send a notification to multiple channels."""
        results = {}
        recipients = recipients or {}

        for channel in channels:
            notification = Notification(
                channel=channel,
                recipient=recipients.get(channel, ""),
                title=title,
                message=message,
                priority=priority,
                metadata=metadata or {},
                user_id=user_id,
            )
            results[channel] = await self.send(notification)

        return results

    def _add_to_history(self, notification: Notification):
        """Add notification to history, pruning if necessary."""
        self._history.append(notification)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

    def get_history(
        self,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[Notification]:
        """Get notification history, optionally filtered."""
        results = self._history
        if user_id:
            results = [n for n in results if n.user_id == user_id]
        if status:
            results = [n for n in results if n.status == status]
        return results[-limit:]


# Convenience functions
async def notify_slack(
    title: str,
    message: str,
    priority: NotificationPriority = NotificationPriority.NORMAL,
):
    """Quick helper to send a Slack notification."""
    service = NotificationService()
    notification = Notification(
        channel=NotificationChannel.SLACK,
        title=title,
        message=message,
        priority=priority,
    )
    return await service.send(notification)


async def notify_user(
    user_id: str,
    title: str,
    message: str,
    channels: list[NotificationChannel] | None = None,
    priority: NotificationPriority = NotificationPriority.NORMAL,
):
    """Send notification to a user on their preferred channels."""
    service = NotificationService()
    channels = channels or [NotificationChannel.SLACK]
    return await service.send_multi(
        channels=channels,
        title=title,
        message=message,
        priority=priority,
        user_id=user_id,
    )
