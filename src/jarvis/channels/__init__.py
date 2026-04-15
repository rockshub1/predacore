"""
JARVIS Channel Adapters — multi-platform message routing.

Each adapter implements the ChannelAdapter interface from gateway.py
and connects JARVIS to a specific messaging platform.

Available channels:
  - telegram   — Telegram Bot API (long polling)
  - discord    — Discord bot (DM-only)
  - whatsapp   — WhatsApp Business Cloud API
  - webchat    — WebSocket-based browser chat UI
"""
