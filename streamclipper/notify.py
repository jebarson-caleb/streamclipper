"""Discord / Telegram notifications (both optional)."""
from __future__ import annotations

import logging

import httpx

from .config import get_settings

log = logging.getLogger(__name__)


async def notify(text: str) -> None:
    n = get_settings().notify
    log.info("notify: %s", text)
    if not (n.discord_webhook or (n.telegram_bot_token and n.telegram_chat_id)):
        return
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            if n.discord_webhook:
                await c.post(n.discord_webhook, json={"content": text[:1900]})
            if n.telegram_bot_token and n.telegram_chat_id:
                await c.post(f"https://api.telegram.org/bot{n.telegram_bot_token}/sendMessage",
                             json={"chat_id": n.telegram_chat_id, "text": text[:4000]})
    except httpx.HTTPError as e:
        log.warning("notification failed: %s", e)
