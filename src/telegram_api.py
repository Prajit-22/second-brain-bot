"""Pure Telegram Bot API helpers: update parsing, authorization and payload
building. No network calls happen here, so everything is unit tested; the
Worker (entry.py) does the actual HTTPS with js.fetch."""

import hmac
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# /ask some question  ->  ("ask", "some question"); tolerates /ask@MyBot and
# multi-line arguments.
_COMMAND_RE = re.compile(r"^/(\w+)(?:@\w+)?(?:\s+(.*))?$", re.S)
_URL_RE = re.compile(r"https?://[^\s<>\"'()\[\],;!?]+", re.I)

TELEGRAM_MAX_MESSAGE = 4096


@dataclass
class Update:
    """The parts of a Telegram update this bot cares about."""

    update_id: int
    chat_id: int
    kind: str  # "text" | "voice" | "unsupported"
    text: str = ""
    file_id: Optional[str] = None
    caption: str = ""
    is_forward: bool = False
    sender_name: str = ""


def parse_update(payload: dict) -> Optional[Update]:
    """Normalize a raw webhook payload. Returns None for updates we can
    safely ignore (channel posts, service messages, callbacks...)."""
    message = payload.get("message") or payload.get("edited_message")
    if not isinstance(message, dict):
        return None
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return None
    sender = message.get("from") or {}
    name = " ".join(
        part for part in (sender.get("first_name", ""), sender.get("last_name", "")) if part
    ).strip()
    is_forward = bool(
        message.get("forward_origin")
        or message.get("forward_from")
        or message.get("forward_from_chat")
        or message.get("forward_sender_name")
    )
    update_id = int(payload.get("update_id", 0))
    media = message.get("voice") or message.get("audio")
    if isinstance(media, dict) and media.get("file_id"):
        return Update(
            update_id=update_id,
            chat_id=int(chat_id),
            kind="voice",
            file_id=media["file_id"],
            caption=message.get("caption", "") or "",
            is_forward=is_forward,
            sender_name=name,
        )
    text = message.get("text") or ""
    if text.strip():
        return Update(
            update_id=update_id,
            chat_id=int(chat_id),
            kind="text",
            text=text.strip(),
            is_forward=is_forward,
            sender_name=name,
        )
    return Update(update_id=update_id, chat_id=int(chat_id), kind="unsupported")


def parse_command(text: str) -> Optional[Tuple[str, str]]:
    """'/ask what is RRF' -> ('ask', 'what is RRF'); plain text -> None."""
    match = _COMMAND_RE.match(text.strip())
    if not match:
        return None
    return match.group(1).lower(), (match.group(2) or "").strip()


def find_urls(text: str) -> List[str]:
    """HTTP(S) links in a message, with trailing punctuation stripped."""
    return [url.rstrip(".,;:!?)\"'") for url in _URL_RE.findall(text)]


def is_authorized(chat_id: int, allowed: Optional[Set[int]]) -> bool:
    """None means 'no whitelist configured' (allow anyone who finds the bot);
    a set restricts the bot to those chat ids."""
    return allowed is None or chat_id in allowed


def verify_secret(provided: Optional[str], expected: Optional[str]) -> bool:
    """Compare Telegram's X-Telegram-Bot-Api-Secret-Token header against the
    secret we registered in setWebhook, in constant time."""
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided, expected)


def api_url(token: str, method: str) -> str:
    return "https://api.telegram.org/bot{token}/{method}".format(token=token, method=method)


def file_url(token: str, file_path: str) -> str:
    return "https://api.telegram.org/file/bot{token}/{path}".format(token=token, path=file_path)


def build_send_payload(chat_id: int, text: str) -> Dict:
    return {"chat_id": chat_id, "text": text, "disable_web_page_preview": False}


def split_message(text: str, limit: int = TELEGRAM_MAX_MESSAGE) -> List[str]:
    """Split an over-long reply on newlines (then hard cuts) so it fits
    Telegram's 4096-char message cap."""
    if len(text) <= limit:
        return [text]
    parts: List[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(remaining[:cut].rstrip("\n"))
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        parts.append(remaining)
    return parts
