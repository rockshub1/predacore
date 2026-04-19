"""
Discord Channel Adapter — connects PredaCore to Discord via discord.py.

DM-only mode for privacy (personal assistant doesn't need server access).

Setup:
  1. Create a bot at https://discord.com/developers/applications
  2. Enable "Message Content Intent" in bot settings
  3. Add token to config.yaml:
       channels:
         discord:
           token: "BOT_TOKEN_HERE"
  4. Enable discord in channels.enabled list

Features:
  - DM-only (privacy-first)
  - Rich embeds for structured responses
  - Message chunking (>2000 chars)
  - Code block formatting
"""
from __future__ import annotations

import asyncio
import logging

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

DISCORD_MAX_LENGTH = 2000


def _chunk_message(text: str, max_len: int = DISCORD_MAX_LENGTH) -> list[str]:
    """Split a long message into Discord-safe chunks."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break

        # Try to split at a newline or code block boundary
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len

        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")

    return chunks


class DiscordAdapter(ChannelAdapter):
    """
    Discord bot adapter — DM-only for personal assistant use.
    """

    channel_name = "discord"
    channel_capabilities = {
        "supports_media": True,
        "supports_buttons": False,  # Buttons need interactions
        "supports_embeds": True,
        "supports_markdown": True,
        "max_message_length": DISCORD_MAX_LENGTH,
    }

    def __init__(self, config: PredaCoreConfig):
        self.config = config
        self._message_handler = None
        self._token = getattr(
            config.channels, "discord_token", None
        ) or config.channels.__dict__.get("discord", {}).get("token", "")
        self._client = None
        self._ready_event = asyncio.Event()
        self._bot_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the Discord bot."""
        if not self._token:
            logger.error("Discord token not configured. Set channels.discord.token")
            return

        try:
            import discord
        except ImportError:
            logger.error("discord.py not installed. Run: pip install discord.py")
            return

        intents = discord.Intents.default()
        intents.message_content = True
        intents.dm_messages = True

        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready():
            logger.info("✅ Discord adapter ready as %s", self._client.user)
            self._ready_event.set()

        @self._client.event
        async def on_message(message):
            # Ignore own messages
            if message.author == self._client.user:
                return

            # DM-only: ignore server messages
            if message.guild is not None:
                return

            await self._handle_message(message)

        # Start the client in the background
        self._bot_task = asyncio.create_task(
            self._client.start(self._token),
            name="discord-bot",
        )
        logger.info("Discord adapter starting...")

    async def stop(self) -> None:
        """Stop the Discord bot."""
        if self._client:
            await self._client.close()
            logger.info("Discord adapter stopped")

    async def send(self, message: OutgoingMessage) -> None:
        """Send a DM to a Discord user."""
        if not self._client:
            logger.error("Discord client not initialized")
            return

        try:
            user = await self._client.fetch_user(int(message.user_id))
            dm_channel = await user.create_dm()

            chunks = _chunk_message(message.text)
            for chunk in chunks:
                await dm_channel.send(chunk)

        except (OSError, ConnectionError, ValueError) as e:
            logger.error("Failed to send Discord DM to %s: %s", message.user_id, e)

    async def _handle_message(self, message) -> None:
        """Process an incoming Discord DM."""
        user_id = str(message.author.id)
        content = message.content

        logger.info("Discord DM from %s: %s", message.author.name, content[:100])

        # Show typing indicator
        async with message.channel.typing():
            if self._message_handler:
                outgoing = await self._message_handler(
                    IncomingMessage(
                        channel=self.channel_name,
                        user_id=user_id,
                        text=content,
                        metadata={
                            "username": str(message.author),
                            "channel_id": message.channel.id,
                        },
                    )
                )

                if outgoing:
                    chunks = _chunk_message(outgoing.text)
                    for chunk in chunks:
                        await message.channel.send(chunk)
