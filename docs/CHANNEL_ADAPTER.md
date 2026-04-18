# Writing a PredaCore Channel Adapter

Channels are how users talk to a PredaCore agent: terminal, webchat, Telegram,
Discord, WhatsApp — and anything else you want to add. This doc walks through
building a new channel and shipping it as an installable package.

## The interface

Every channel subclasses `predacore.gateway.ChannelAdapter` and implements
four things:

```python
from predacore.gateway import (
    ChannelAdapter, IncomingMessage, OutgoingMessage,
)


class SlackAdapter(ChannelAdapter):
    # Required: unique name matching the value users put in
    # config.channels.enabled and that tools like channel_configure use.
    channel_name = "slack"

    # Required: capability metadata. Format — Slack supports markdown but
    # trims messages to 4000 chars, etc.
    channel_capabilities = {
        "supports_media":        True,
        "supports_buttons":      True,
        "supports_markdown":     True,
        "max_message_length":    39000,
    }

    def __init__(self, config):
        self.config = config
        self._message_handler = None
        # Read your secret from the environment. The `secret_set` tool and
        # the setup wizard write to ~/.predacore/.env; config.channels
        # holds per-channel fields for structured access.
        self._token = os.getenv("SLACK_BOT_TOKEN", "")

    async def start(self) -> None:
        """Open a connection / begin polling / bind a webhook."""
        ...

    async def stop(self) -> None:
        """Clean shutdown — idempotent."""
        ...

    async def send(self, message: OutgoingMessage) -> None:
        """Deliver an outgoing message to the channel."""
        ...

    async def _handle_incoming(self, user_id: str, text: str) -> None:
        """Route an inbound message through the gateway."""
        if self._message_handler is None:
            return
        outgoing = await self._message_handler(IncomingMessage(
            channel=self.channel_name,
            user_id=user_id,
            text=text,
        ))
        if outgoing is not None:
            await self.send(outgoing)
```

The base class takes care of `set_message_handler` — the gateway wires that
up during registration. Your job is `start/stop/send` plus whatever inbound
loop your platform needs.

## Distribution

### Option 1 — publish to PyPI (canonical)

Create a standalone package, e.g. `predacore-slack`:

```
predacore-slack/
├── pyproject.toml
├── README.md
└── predacore_slack/
    ├── __init__.py
    └── adapter.py          # contains SlackAdapter
```

Declare the entry point so the gateway auto-discovers it:

```toml
# pyproject.toml
[project]
name = "predacore-slack"
version = "0.1.0"
dependencies = [
  "predacore>=0.1",
  "slack-sdk>=3.27",
]

[project.entry-points."predacore.channels"]
slack = "predacore_slack.adapter:SlackAdapter"
```

End users:

```
pip install predacore-slack
predacore bootstrap     # no-op for channels, but a good sanity check
# Then in chat:
#   "add slack with token xoxb-..."
# The agent calls channel_configure and Slack is live after a daemon restart.
```

### Option 2 — drop a file for local experiments

For a one-off adapter, save the class to
`~/.predacore/channels/my_adapter.py`:

```python
# ~/.predacore/channels/my_adapter.py
from predacore.gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

class MyAdapter(ChannelAdapter):
    channel_name = "myadapter"
    channel_capabilities = {"max_message_length": 2000}

    def __init__(self, config): ...
    async def start(self): ...
    async def stop(self): ...
    async def send(self, message: OutgoingMessage): ...
```

The registry imports every top-level class ending in `Adapter` from any
`.py` in that directory on startup. Restart the daemon and the channel
is registered; enable it via `channel_configure action=add
channel=myadapter`.

## Discovery precedence

The gateway scans three sources for adapter classes (later wins on name clash):

1. **Built-in** — `predacore.channels.{telegram, discord, whatsapp, webchat}`
2. **Entry points** — any installed package declaring `predacore.channels`
3. **User directory** — `~/.predacore/channels/*.py`

This means you can shadow a built-in adapter locally (e.g. fork Telegram)
without modifying core.

## Capabilities schema

`channel_capabilities` is a dict the gateway can consult when chunking
messages or deciding what to render. Keys used today:

| Key | Type | Used by |
|---|---|---|
| `max_message_length` | int | gateway's message chunker |
| `supports_media` | bool | attachment handling |
| `supports_buttons` | bool | tool-confirmation UX |
| `supports_markdown` | bool | prompt formatting decisions |
| `supports_embeds` | bool | rich-response rendering |

Add your own keys freely — the gateway ignores unknown ones.

## Testing

Two things you want to cover:

1. **Adapter class construction** — instantiating `SlackAdapter(config)`
   shouldn't raise when the token is missing; it should log + no-op on
   `start()`. This is how PredaCore degrades gracefully when a user has
   `slack` enabled but hasn't set `SLACK_BOT_TOKEN` yet.
2. **Message round-trip** — mock `set_message_handler` to capture the
   `IncomingMessage` the adapter emits on a fake inbound, and assert
   `send()` produces the expected payload for a given `OutgoingMessage`.

## Good adapter traits

- **Lazy-import platform SDKs.** Do `import slack_sdk` inside `start()`,
  not at module top level. Users who install `predacore-slack` and then
  never enable slack shouldn't pay import cost; users who forgot to
  install `slack-sdk` should see a clear error only when they try to use
  it.
- **Honor `max_message_length`.** The gateway will chunk for you if you
  declare it; don't trust unlimited.
- **Keep tokens in `.env`.** Never read from `config.yaml` for secrets —
  only from `os.environ` (the config loader merges them in). That way
  `secret_set` and `channel_configure` can update live secrets without
  touching config.
- **Fail quietly on a missing token.** Log a warning and stop; don't
  crash the gateway.
- **Forward `user_id` stably.** Use the platform's user ID (Slack user
  ID, Telegram chat ID). The gateway's IdentityService resolves that to
  a canonical user across channels.

## Reference implementations

Read these before writing your own:

- `src/predacore/channels/telegram.py` — long-polling pattern, most
  common for bot APIs.
- `src/predacore/channels/discord.py` — event-driven client pattern.
- `src/predacore/channels/whatsapp.py` — webhook receiver pattern
  (good template for any webhook-based platform).
- `src/predacore/channels/webchat.py` — embedded HTTP+WebSocket server
  pattern.

Pick the one closest to your platform, copy it, swap the SDK.
