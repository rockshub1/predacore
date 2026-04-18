"""
Telegram Channel Adapter — connects PredaCore to Telegram via Bot API.

Uses python-telegram-bot (async) with long polling.
No webhook needed — ideal for personal assistant use.

Setup:
  1. Create a bot via @BotFather on Telegram
  2. Add the token to config.yaml:
       channels:
         telegram:
           token: "BOT_TOKEN_HERE"
  3. Enable telegram in channels.enabled list
  4. Start the daemon: predacore start --daemon

Features:
  - Long polling (no public URL needed)
  - Markdown formatting for responses
  - Message chunking for long responses (>4096 chars)
  - Image/document support
  - Inline buttons for tool confirmations
  - Continuous typing indicator during long responses
  - User allow-list enforcement
"""
from __future__ import annotations

import asyncio
import logging
import re
import time

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

# Max Telegram message length
TG_MAX_LENGTH = 4096

# Telegram Bot API connection timeouts (seconds)
TG_CONNECT_TIMEOUT = 30.0
TG_READ_TIMEOUT = 30.0
TG_WRITE_TIMEOUT = 30.0
TG_POLL_READ_TIMEOUT = 42.0  # Long-poll read timeout (> standard to keep connection alive)

# Typing indicator refresh interval (Telegram typing expires after ~5s)
_TYPING_INTERVAL = 4.0
_RECENT_UPDATE_TTL = 30.0
_RECENT_UPDATE_MAX = 512
_RECENT_START_TTL = 10.0


def _chunk_message(text: str, max_len: int = TG_MAX_LENGTH) -> list[str]:
    """Split a long message into chunks that fit Telegram's limits."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break

        # Try to split at a newline
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len

        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")

    return chunks


def _md_to_telegram(text: str) -> str:
    """Convert standard markdown to Telegram MarkdownV2 format.

    Handles code blocks (``` and inline `), bold (**), italic (_),
    and escapes all MarkdownV2 special chars outside of those.
    """
    # Characters that need escaping in MarkdownV2 (outside code/bold/italic)
    _ESCAPE_RE = re.compile(r'([_\[\]()~>#\+\-=|{}.!\\])')

    lines = text.split("\n")
    result = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            result.append(line)
        elif in_code_block:
            result.append(line)
        else:
            # Protect inline code spans from escaping
            parts = line.split("`")
            escaped_parts = []
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    # Outside inline code — escape special chars
                    # But preserve ** for bold
                    segments = part.split("**")
                    escaped_segments = [_ESCAPE_RE.sub(r'\\\1', s) for s in segments]
                    escaped_parts.append("**".join(escaped_segments))
                else:
                    # Inside inline code — leave as-is
                    escaped_parts.append(part)
            result.append("`".join(escaped_parts))

    return "\n".join(result)


class TelegramAdapter(ChannelAdapter):
    """
    Telegram Bot API adapter using long polling.

    Receives messages from Telegram, routes them through the Gateway,
    and sends responses back to the user.
    """

    channel_name = "telegram"
    channel_capabilities = {
        "supports_media": True,
        "supports_buttons": True,
        "supports_markdown": True,
        "max_message_length": TG_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        self.config = config
        self._message_handler = None
        self._token = getattr(
            config.channels, "telegram_token", None
        ) or config.channels.__dict__.get("telegram", {}).get("token", "")
        self._bot = None
        self._app = None
        # User allow-list: if non-empty, only these Telegram user IDs can interact
        tg_config = config.channels.__dict__.get("telegram", {})
        allowed = tg_config.get("allowed_users", [])
        self._allowed_users: set[int] = set(int(u) for u in allowed) if allowed else set()
        self._recent_update_keys: dict[str, float] = {}
        self._recent_start_keys: dict[str, float] = {}

    async def start(self) -> None:
        """Start the Telegram bot with long polling."""
        if not self._token:
            logger.error("Telegram token not configured. Set channels.telegram.token")
            return

        try:
            from telegram import Update
            from telegram.ext import (
                ApplicationBuilder,
                CommandHandler,
                MessageHandler,
                filters,
            )
        except ImportError:
            logger.error(
                "python-telegram-bot not installed. "
                "Run: pip install python-telegram-bot"
            )
            return

        self._app = (
            ApplicationBuilder()
            .token(self._token)
            .connect_timeout(TG_CONNECT_TIMEOUT)
            .read_timeout(TG_READ_TIMEOUT)
            .write_timeout(TG_WRITE_TIMEOUT)
            .get_updates_read_timeout(TG_POLL_READ_TIMEOUT)
            .build()
        )

        # Register handlers
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        # Session + cross-channel commands — route through gateway as text
        self._app.add_handler(CommandHandler("new", self._cmd_gateway))
        self._app.add_handler(CommandHandler("resume", self._cmd_gateway))
        self._app.add_handler(CommandHandler("sessions", self._cmd_gateway))
        self._app.add_handler(CommandHandler("cancel", self._cmd_gateway))
        self._app.add_handler(CommandHandler("link", self._cmd_gateway))
        self._app.add_handler(CommandHandler("brain", self._cmd_gateway))
        self._app.add_handler(CommandHandler("model", self._cmd_gateway))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )

        # Start polling
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        if self._allowed_users:
            logger.info("✅ Telegram adapter started (long polling, %d allowed users)", len(self._allowed_users))
        else:
            logger.info("✅ Telegram adapter started (long polling, all users allowed)")

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("Telegram adapter stopped")

    def _check_allowed(self, user_id: int) -> bool:
        """Return True if the user is allowed to interact (or if no allow-list is set)."""
        if not self._allowed_users:
            return True
        return user_id in self._allowed_users

    def _claim_update(self, update) -> bool:
        """Return True once for a Telegram update/message, False for near-term duplicates."""
        now = time.monotonic()
        if self._recent_update_keys:
            expired = [
                key for key, seen_at in self._recent_update_keys.items()
                if now - seen_at > _RECENT_UPDATE_TTL
            ]
            for key in expired:
                self._recent_update_keys.pop(key, None)

        message = getattr(update, "message", None)
        keys: list[str] = []

        update_id = getattr(update, "update_id", None)
        if update_id is not None:
            keys.append(f"u:{update_id}")

        if message is not None:
            chat_id = getattr(message, "chat_id", None)
            message_id = getattr(message, "message_id", None)
            if chat_id is not None and message_id is not None:
                keys.append(f"m:{chat_id}:{message_id}")

        if any(key in self._recent_update_keys for key in keys):
            logger.info(
                "Suppressing duplicate Telegram update: %s",
                ", ".join(keys) or "<unknown>",
            )
            return False

        for key in keys:
            self._recent_update_keys[key] = now

        if len(self._recent_update_keys) > _RECENT_UPDATE_MAX:
            for key, _ in sorted(
                self._recent_update_keys.items(),
                key=lambda item: item[1],
            )[: len(self._recent_update_keys) - _RECENT_UPDATE_MAX]:
                self._recent_update_keys.pop(key, None)

        return True

    def _claim_start(self, chat_id: int | None, user_id: int | None) -> bool:
        """Return True unless /start was already handled very recently for this chat/user."""
        if chat_id is None or user_id is None:
            return True

        now = time.monotonic()
        expired = [
            key for key, seen_at in self._recent_start_keys.items()
            if now - seen_at > _RECENT_START_TTL
        ]
        for key in expired:
            self._recent_start_keys.pop(key, None)

        key = f"{chat_id}:{user_id}"
        if key in self._recent_start_keys:
            logger.info("Suppressing repeated /start for chat=%s user=%s", chat_id, user_id)
            return False

        self._recent_start_keys[key] = now
        return True

    async def send(self, message: OutgoingMessage) -> None:
        """Send a response message to a Telegram user."""
        text = message.text or ""
        if not text.strip():
            logger.warning("Telegram send received empty text — using fallback message")
            text = (
                "I completed the request, but the model returned an empty reply. "
                "Please ask me to summarize the last step again."
            )
        if not self._app or not self._app.bot:
            logger.error("Telegram bot not initialized")
            return

        # Prefer chat_id from metadata (original Telegram ID), fall back to user_id
        raw_chat_id = (message.metadata or {}).get("chat_id") or message.user_id
        try:
            chat_id = int(raw_chat_id)
        except (ValueError, TypeError):
            logger.error("Invalid chat_id for Telegram: %s (user_id=%s)", raw_chat_id, message.user_id)
            return
        chunks = _chunk_message(text)

        for chunk in chunks:
            try:
                await self._app.bot.send_message(
                    chat_id=chat_id,
                    text=_md_to_telegram(chunk),
                    parse_mode="MarkdownV2",
                )
            except Exception as fmt_err:
                # Fallback: send without markdown if formatting fails
                if "parse" not in str(fmt_err).lower() and "markdown" not in str(fmt_err).lower() and "can't" not in str(fmt_err).lower():
                    # Not a formatting error — log and skip fallback
                    logger.error("Telegram send failed (non-formatting): %s", fmt_err)
                    continue
                try:
                    await self._app.bot.send_message(
                        chat_id=chat_id,
                        text=chunk,
                    )
                except Exception as e:
                    logger.error("Failed to send Telegram message: %s", e)

    async def _cmd_start(self, update, context) -> None:
        """Handle /start command."""
        if not update.message or not self._claim_update(update):
            return
        if not self._claim_start(
            getattr(update.message, "chat_id", None),
            getattr(update.message.from_user, "id", None),
        ):
            return
        if not self._check_allowed(update.message.from_user.id):
            return
        await update.message.reply_text(
            "🔥 *PredaCore Online*\n\n"
            "I'm your AI assistant powered by PredaCore.\n"
            "Send me any message and I'll help you out.\n\n"
            "Commands:\n"
            "  /status — Check my status\n"
            "  /help — Show help\n"
            "  /sessions — List your sessions\n"
            "  /new — Start a fresh session\n"
            "  /resume — Resume a previous session\n"
            "  /link — Link this chat to another channel\n"
            "  /model — Show/switch LLM provider\n",
            parse_mode="Markdown",
        )

    async def _cmd_help(self, update, context) -> None:
        """Handle /help command."""
        if not update.message or not self._claim_update(update):
            return
        if not self._check_allowed(update.message.from_user.id):
            return
        await update.message.reply_text(
            "💡 *PredaCore Help*\n\n"
            "Just send me any message — I can:\n"
            "• Answer questions\n"
            "• Execute code\n"
            "• Manage files\n"
            "• Search the web\n"
            "• Run scheduled tasks\n\n"
            "I remember our conversation across restarts.\n\n"
            "*Session commands:*\n"
            "  /sessions — List your sessions\n"
            "  /new — Start a fresh session\n"
            "  /resume — Resume most recent session\n"
            "  /resume 2 — Resume session #2\n\n"
            "*Cross-channel:*\n"
            "  /link — Get a code to link this chat to webchat\n"
            "  /link CODE — Link to another channel using a code",
            parse_mode="Markdown",
        )

    async def _cmd_status(self, update, context) -> None:
        """Handle /status command."""
        if not update.message or not self._claim_update(update):
            return
        if not self._check_allowed(update.message.from_user.id):
            return
        await update.message.reply_text(
            "✅ *PredaCore Status: Online*\n"
            f"Channel: Telegram\n"
            f"Mode: {self.config.mode}\n"
            f"Trust: {self.config.security.trust_level}\n",
            parse_mode="Markdown",
        )

    async def _cmd_gateway(self, update, context) -> None:
        """Route commands through gateway: /new, /resume, /sessions, /cancel, /link, /model."""
        if not update.message or not self._claim_update(update):
            return
        if not self._check_allowed(update.message.from_user.id):
            return
        text = update.message.text or ""
        user_id = str(update.message.from_user.id)

        if self._message_handler:
            outgoing = await self._message_handler(
                IncomingMessage(
                    channel=self.channel_name,
                    user_id=user_id,
                    text=text,
                    metadata={
                        "chat_id": update.message.chat_id,
                        "username": update.message.from_user.username,
                        "raw_user_id": user_id,
                    },
                )
            )
            if outgoing:
                await self.send(outgoing)

    async def _keep_typing(self, chat, stop_event: asyncio.Event) -> None:
        """Send typing indicator every 4s until stop_event is set."""
        while not stop_event.is_set():
            try:
                await chat.send_action("typing")
            except (OSError, ConnectionError):
                pass
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=_TYPING_INTERVAL)
                return  # Event was set
            except asyncio.TimeoutError:
                continue  # Send another typing indicator

    async def _on_message(self, update, context) -> None:
        """Handle incoming text messages."""
        if not update.message or not update.message.text or not self._claim_update(update):
            return

        tg_user_id = update.message.from_user.id
        if not self._check_allowed(tg_user_id):
            logger.debug("Ignoring message from unauthorized user %d", tg_user_id)
            return

        user_id = str(tg_user_id)
        message = update.message.text
        chat = update.effective_chat

        # Classify chat type and social context
        is_private = chat.type == "private"
        is_mentioned = self._is_bot_mentioned(update, context)
        is_reply_to_bot = self._is_reply_to_bot(update, context)

        # In group chats, only respond when mentioned or replied to
        if not is_private and not is_mentioned and not is_reply_to_bot:
            logger.debug(
                "Ignoring group message (not mentioned): %s", message[:50]
            )
            return

        logger.info("Telegram message from %s: %s", user_id, message[:100])

        # Start continuous typing indicator
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(self._keep_typing(update.message.chat, stop_typing))

        try:
            if self._message_handler:
                outgoing = await self._message_handler(
                    IncomingMessage(
                        channel=self.channel_name,
                        user_id=user_id,
                        text=message,
                        metadata={
                            "chat_id": update.message.chat_id,
                            "username": update.message.from_user.username,
                            "raw_user_id": user_id,
                            "channel_type": "private" if is_private else "group",
                            "chat_title": getattr(chat, "title", None),
                            "is_mentioned": is_mentioned,
                            "is_reply_to_bot": is_reply_to_bot,
                        },
                    )
                )

                if outgoing:
                    await self.send(outgoing)
        finally:
            stop_typing.set()
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

    @staticmethod
    def _is_bot_mentioned(update, context) -> bool:
        """Check if the bot was @mentioned in the message."""
        if not update.message or not update.message.text:
            return False
        bot_username = context.bot.username
        if bot_username and f"@{bot_username}" in update.message.text:
            return True
        if update.message.entities:
            for entity in update.message.entities:
                if entity.type == "mention":
                    mentioned = update.message.text[
                        entity.offset : entity.offset + entity.length
                    ]
                    if bot_username and mentioned.lower() == f"@{bot_username}".lower():
                        return True
        return False

    @staticmethod
    def _is_reply_to_bot(update, context) -> bool:
        """Check if the message is a reply to one of the bot's messages."""
        if not update.message or not update.message.reply_to_message:
            return False
        reply_from = update.message.reply_to_message.from_user
        return reply_from is not None and reply_from.id == context.bot.id
