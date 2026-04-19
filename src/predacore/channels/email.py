"""
Email Channel Adapter — IMAP in, SMTP out.

The async agent-over-email channel. Inbound messages are pulled from
any IMAP mailbox (Gmail, Outlook/Microsoft 365, Fastmail, self-hosted).
Outbound responses go via SMTP, threaded into the same conversation
using ``In-Reply-To`` + ``References`` headers so mail clients group
them naturally.

Setup
-----
Gmail (via app passwords)
~~~~~~~~~~~~~~~~~~~~~~~~~
1. Turn on 2FA on your Google account.
2. Create an app password for "Mail": https://myaccount.google.com/apppasswords
3. Store config::

       EMAIL_USERNAME=me@gmail.com
       EMAIL_PASSWORD=<16-char app password>
       EMAIL_FROM_ADDRESS=me@gmail.com          # defaults to EMAIL_USERNAME
       # IMAP + SMTP defaults are correct for Gmail, no further config needed.

Custom / self-hosted
~~~~~~~~~~~~~~~~~~~~
Override these env vars as needed::

    IMAP_HOST=imap.example.com    IMAP_PORT=993
    SMTP_HOST=smtp.example.com    SMTP_PORT=587

Enable the channel::

    channels.enabled += ["email"]

Features
--------
- IDLE-based inbound (low latency, no polling loop)
- Automatic threading: ``In-Reply-To`` + ``References`` preserved
- Auto-reads the ``Reply-To`` header if present so replies go to the
  right address (important for lists)
- Skips messages from the bot's own address to avoid loops
- Lazy-imports ``aioimaplib`` + ``aiosmtplib`` so the core install
  doesn't need them on startup
"""
from __future__ import annotations

import asyncio
import email
import logging
import os
import ssl
from email.message import EmailMessage
from email.utils import make_msgid, parseaddr

from ..config import PredaCoreConfig
from ..gateway import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

# Common hosts — good defaults so Gmail users don't touch config.
_DEFAULT_IMAP_HOST = "imap.gmail.com"
_DEFAULT_IMAP_PORT = 993
_DEFAULT_SMTP_HOST = "smtp.gmail.com"
_DEFAULT_SMTP_PORT = 587


class EmailAdapter(ChannelAdapter):
    """IMAP (inbound) + SMTP (outbound). Threaded, async."""

    channel_name = "email"
    channel_capabilities = {
        "supports_media": True,   # attachments via MIME; we don't forward yet
        "supports_buttons": False,
        "supports_embeds": False,
        "supports_markdown": False,
        "max_message_length": 100_000,
    }

    def __init__(self, config: PredaCoreConfig):
        self.config = config
        self._message_handler = None

        e = lambda k, d="": os.getenv(k, d)  # noqa: E731
        cfg = getattr(config.channels, "__dict__", {}).get("email", {}) or {}

        self._username = e("EMAIL_USERNAME") or cfg.get("username", "")
        self._password = e("EMAIL_PASSWORD") or cfg.get("password", "")
        self._from_addr = e("EMAIL_FROM_ADDRESS") or cfg.get("from_address", "") or self._username
        self._imap_host = e("IMAP_HOST") or cfg.get("imap_host", "") or _DEFAULT_IMAP_HOST
        self._imap_port = int(e("IMAP_PORT", "0") or cfg.get("imap_port") or _DEFAULT_IMAP_PORT)
        self._smtp_host = e("SMTP_HOST") or cfg.get("smtp_host", "") or _DEFAULT_SMTP_HOST
        self._smtp_port = int(e("SMTP_PORT", "0") or cfg.get("smtp_port") or _DEFAULT_SMTP_PORT)

        # Messaging state per conversation — maps sender address →
        # (last Message-Id, list of prior Message-Ids) so we can thread.
        self._thread_state: dict[str, tuple[str, list[str]]] = {}
        self._idle_task: asyncio.Task | None = None
        self._running = False
        self._imap_client = None

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        if not self._username or not self._password:
            logger.error(
                "Email: missing EMAIL_USERNAME / EMAIL_PASSWORD — add them "
                "to ~/.predacore/.env (or use secret_set)"
            )
            return

        try:
            import aioimaplib  # type: ignore[import-not-found]  # noqa: F401
            import aiosmtplib  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            logger.error(
                "Email: aioimaplib + aiosmtplib not installed. "
                "pip install aioimaplib aiosmtplib (or: pip install predacore[email])"
            )
            return

        self._running = True
        self._idle_task = asyncio.create_task(self._imap_idle_loop(), name="email-idle")
        logger.info(
            "Email adapter started (user=%s, imap=%s:%d, smtp=%s:%d)",
            self._username, self._imap_host, self._imap_port,
            self._smtp_host, self._smtp_port,
        )

    async def stop(self) -> None:
        self._running = False
        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()
            try:
                await self._idle_task
            except (asyncio.CancelledError, Exception):
                pass
        logger.info("Email adapter stopped")

    # ── Outbound ────────────────────────────────────────────────────

    async def send(self, message: OutgoingMessage) -> None:
        import aiosmtplib  # type: ignore[import-not-found]

        recipient = message.user_id
        prior = self._thread_state.get(recipient)
        msg = EmailMessage()
        msg["From"] = self._from_addr
        msg["To"] = recipient
        subj = (message.metadata or {}).get("subject") or "Re: PredaCore reply"
        if not subj.lower().startswith("re:"):
            subj = "Re: " + subj
        msg["Subject"] = subj
        msg["Message-ID"] = make_msgid(domain=self._from_addr.split("@")[-1] or "predacore")
        if prior is not None:
            last_id, refs = prior
            msg["In-Reply-To"] = last_id
            msg["References"] = " ".join(refs + [last_id])
        msg.set_content(message.text)

        try:
            await aiosmtplib.send(
                msg,
                hostname=self._smtp_host,
                port=self._smtp_port,
                username=self._username,
                password=self._password,
                start_tls=self._smtp_port == 587,
                use_tls=self._smtp_port == 465,
                tls_context=ssl.create_default_context(),
            )
        except Exception as exc:
            logger.error("Email send failed: %s", exc)
            return

        # Record our outbound Message-ID for future threading.
        new_id = msg["Message-ID"]
        refs = (prior[1] + [prior[0]]) if prior else []
        self._thread_state[recipient] = (new_id, refs)

    # ── Inbound (IMAP IDLE) ─────────────────────────────────────────

    async def _imap_idle_loop(self) -> None:
        """Stay connected via IMAP IDLE; reconnect on any error."""
        import aioimaplib  # type: ignore[import-not-found]

        backoff = 2.0
        while self._running:
            try:
                client = aioimaplib.IMAP4_SSL(
                    host=self._imap_host, port=self._imap_port, timeout=60,
                )
                await client.wait_hello_from_server()
                await client.login(self._username, self._password)
                await client.select("INBOX")

                # Process anything unseen right now.
                await self._process_unseen(client)
                backoff = 2.0  # reset after a good connect

                # Then wait for new deliveries.
                while self._running:
                    idle_task = await client.idle_start(timeout=29 * 60)
                    try:
                        await client.wait_server_push()
                    finally:
                        client.idle_done()
                        try:
                            await asyncio.wait_for(idle_task, timeout=5)
                        except asyncio.TimeoutError:
                            pass
                    if not self._running:
                        break
                    await self._process_unseen(client)

            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning("Email IMAP error (retrying in %.0fs): %s", backoff, exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def _process_unseen(self, client) -> None:
        """Pull and handle every UNSEEN message, marking them read."""
        try:
            typ, data = await client.search("UNSEEN")
        except Exception as exc:
            logger.debug("Email search failed: %s", exc)
            return
        if typ != "OK" or not data:
            return
        ids = (data[0] or b"").split()
        for raw_id in ids:
            try:
                typ2, payload = await client.fetch(raw_id.decode(), "(RFC822)")
            except Exception as exc:
                logger.debug("Email fetch failed for %s: %s", raw_id, exc)
                continue
            if typ2 != "OK" or not payload:
                continue
            raw_bytes = self._extract_rfc822_bytes(payload)
            if not raw_bytes:
                continue
            try:
                msg = email.message_from_bytes(raw_bytes)
            except Exception:
                continue
            await self._handle_incoming(msg)
            try:
                await client.store(raw_id.decode(), "+FLAGS", "(\\Seen)")
            except Exception:
                pass

    @staticmethod
    def _extract_rfc822_bytes(payload: list) -> bytes:
        """aioimaplib returns a list of literal tuples — pick the bytes blob."""
        for item in payload:
            if isinstance(item, (bytes, bytearray)):
                if len(item) > 40:  # skip the short metadata line
                    return bytes(item)
        return b""

    async def _handle_incoming(self, msg) -> None:
        from_raw = msg.get("From", "")
        reply_to = msg.get("Reply-To", "")
        _, from_addr = parseaddr(reply_to or from_raw)
        if not from_addr or from_addr.lower() == (self._from_addr or "").lower():
            return
        subject = msg.get("Subject", "") or ""
        message_id = msg.get("Message-ID", "")
        in_reply_to = msg.get("In-Reply-To", "")
        references = [r for r in (msg.get("References") or "").split() if r]

        # Prefer text/plain; fall back to stripped HTML.
        body = self._extract_body(msg)
        if not body:
            return

        # Record this message in the thread state so our reply threads right.
        if message_id:
            prior_refs = references or []
            if in_reply_to and in_reply_to not in prior_refs:
                prior_refs = prior_refs + [in_reply_to]
            self._thread_state[from_addr] = (message_id, prior_refs)

        logger.info("Email from %s: %s", from_addr, body[:100])
        if self._message_handler is None:
            return
        outgoing = await self._message_handler(
            IncomingMessage(
                channel=self.channel_name,
                user_id=from_addr,
                text=body,
                metadata={
                    "subject": subject,
                    "message_id": message_id,
                    "in_reply_to": in_reply_to,
                },
            )
        )
        if outgoing is not None:
            outgoing.metadata.setdefault("subject", subject)
            await self.send(outgoing)

    @staticmethod
    def _extract_body(msg) -> str:
        """Return the first text/plain body, or stripped text/html as fallback."""
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/plain":
                    payload = part.get_payload(decode=True) or b""
                    try:
                        return payload.decode(part.get_content_charset() or "utf-8", "replace")
                    except (LookupError, UnicodeDecodeError):
                        return payload.decode("utf-8", "replace")
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    html = (part.get_payload(decode=True) or b"").decode(
                        part.get_content_charset() or "utf-8", "replace",
                    )
                    return _strip_html(html)
            return ""
        payload = msg.get_payload(decode=True) or b""
        text = payload.decode(msg.get_content_charset() or "utf-8", "replace")
        if msg.get_content_type() == "text/html":
            text = _strip_html(text)
        return text


def _strip_html(html: str) -> str:
    """Dumb-but-good-enough HTML → text fallback. Drops tags, decodes entities."""
    import html as htmllib
    import re

    no_scripts = re.sub(r"<(script|style).*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    no_tags = re.sub(r"<[^>]+>", " ", no_scripts)
    collapsed = re.sub(r"[ \t]+", " ", no_tags)
    return htmllib.unescape(collapsed).strip()
