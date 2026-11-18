"""Titles, descriptions, tags and chapters for uploads."""
from __future__ import annotations

import logging
import re

from ..detect.llm import LLMError, chat_json
from ..models import Channel

log = logging.getLogger(__name__)

SYSTEM = """You write YouTube metadata for stream clips. The stream/video title tells you what the content
is (sport, game, etc.): keep titles accurate to it and never invent a different topic. Reply ONLY with JSON:
{"title": "max 70 chars, punchy, accurate", "summary": "1-2 sentence description",
 "tags": ["8-15 lowercase search tags"], "hashtags": ["3-5 hashtags without #"]}"""


def credit_line(ch: Channel) -> str:
    if ch.credit_text:
        return ch.credit_text
    if ch.permission == "own":
        return ""
    return f"Clip from {ch.name}: {ch.url}"


def fmt_ts(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def chapters(items: list[tuple[float, str]]) -> str:
    """YouTube chapters: first must be 0:00, at least 3, each >= 10 s."""
    if len(items) < 3:
        return ""
    return "\n".join(f"{fmt_ts(t)} {title}" for t, title in items)


def as_list(v, split_spaces: bool = False) -> list[str]:
    """Small LLMs return lists, comma strings, or '#a #b' strings. Normalize all of them."""
    items = v if isinstance(v, list) else [v] if v else []
    out: list[str] = []
    for it in items:
        for part in re.split(r"[,\s]+" if split_spaces else r",", str(it)):
            part = part.strip()
            if part:
                out.append(part)
    return out


def clean_title(t: str, limit: int = 100) -> str:
    t = re.sub(r"[<>]", "", t).strip().strip('"')
    return t[:limit] or "Stream highlight"


async def build_metadata(ch: Channel, fmt: str, title_hint: str, transcript: str, category: str,
                         extra_desc: str = "", context: str = "") -> tuple[str, str, list[str]]:
    """title_hint: a title to polish. context: background only (e.g. the stream title) — the LLM must
    write a fresh title about the most exciting thing in the transcript instead of copying it."""
    hint = (f"Suggested title: {title_hint}\nStream/video it comes from: {context}\n" if title_hint else
            f"Stream title (context only, do NOT copy it): {context}\n"
            "Write a NEW title about the most exciting thing that happens in this transcript.\n")
    user = (f"Streamer: {ch.name}\nFormat: {'YouTube Short' if fmt == 'short' else 'long video'}\n"
            f"Category: {category}\n{hint}Transcript excerpt:\n{transcript[:2500]}")
    data: dict = {}
    if len(transcript.split()) >= 6:   # with no speech the LLM just invents things; use the stream title
        try:
            data = await chat_json(SYSTEM, user)
        except LLMError as e:
            log.warning("metadata LLM failed: %s", e)
    title = clean_title(str(data.get("title") or title_hint or context or f"{ch.name} stream highlight"))
    summary = str(data.get("summary") or "")
    tags = [t.lower() for t in as_list(data.get("tags"))]
    tags += [t.strip().lower() for t in ch.extra_tags.split(",") if t.strip()]
    tags += [ch.name.lower(), category] if category else [ch.name.lower()]
    hashtags = [h.lstrip("#") for h in as_list(data.get("hashtags"), split_spaces=True) if h.lstrip("#")][:5]
    if fmt == "short" and "shorts" not in [h.lower() for h in hashtags]:
        hashtags.append("shorts")
    full_title = clean_title(ch.title_template.replace("{title}", title).replace("{channel}", ch.name))
    if fmt == "short" and "#shorts" not in full_title.lower() and len(full_title) <= 90:
        full_title += " #shorts"
    desc = ch.desc_template.format_map(_Safe(
        title=title, summary=summary, credit=credit_line(ch), channel=ch.name,
        hashtags=" ".join(f"#{h}" for h in hashtags),
    ))
    if extra_desc:
        desc = f"{desc}\n\n{extra_desc}"
    # YouTube limits: tags total <= 500 chars, description <= 5000
    seen, out_tags, total = set(), [], 0
    for t in tags:
        t = re.sub(r"[<>,]", "", t)[:30]
        if t and t not in seen and total + len(t) + 1 <= 450:
            seen.add(t)
            out_tags.append(t)
            total += len(t) + 1
    return full_title, re.sub(r"\n{3,}", "\n\n", desc).strip()[:4900], out_tags


class _Safe(dict):
    def __missing__(self, key):
        return ""
