"""Ask the local LLM whether a candidate window is clip-worthy and where to trim it."""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

from .candidates import chat_key
from .llm import LLMError, chat_json
from ..publish.metadata import no_links

log = logging.getLogger(__name__)

CATEGORIES = ["funny", "clutch", "fail", "rage", "wholesome", "hot_take", "story", "reaction",
              "skill", "announcement", "other"]

SYSTEM = f"""You are an expert short-form video editor who finds viral moments in live streams.
You get a transcript with timestamps (seconds), a sample of live chat, and hype statistics.
Judge whether this window contains a moment worth posting as a standalone clip, and where it should start and end
so it makes sense without context: start just before the setup, end right after the payoff/reaction.
Reply with ONLY a JSON object:
{{"score": 0-10 (10 = guaranteed viral, 5 = mildly interesting, 0 = nothing happens),
 "category": one of {CATEGORIES},
 "start": seconds, "end": seconds,
 "title": "catchy YouTube title, max 70 chars, no clickbait lies",
 "hook": "2-6 word on-screen hook shown in the first seconds",
 "reason": "one short sentence"}}
You cannot see the video. Excited commentary, shouting, crowd/chat reactions ("oh!", "wow", "no way", emotes)
and a high hype score mean something visual happened (a play, a dunk, a kill, a fail): score those on the
reaction, not on whether the words alone are funny. Use the stream/video title to understand the content.
The transcript is everything audible: it can be game characters, an in-game radio, an ad, or a video the streamer
is watching, not the streamer. The most repeated chat messages tell you what viewers actually reacted to
(e.g. "L DRIVER" x20 = the streamer crashed): base the title and hook on that reaction, never on background
dialogue. If the transcript is only background dialogue and chat isn't reacting, score it low.
Dead air, loading screens, reading donations, or talk that needs earlier context score low."""


@dataclass
class Judgement:
    score: float
    category: str
    start: float
    end: float
    title: str
    hook: str
    reason: str


def build_prompt(start: float, end: float, words: list[tuple[float, str]], chat: list[tuple[float, str]],
                 hype: float, max_len: float, min_len: float, context: str = "") -> str:
    lines: list[str] = []
    cur_t, cur = None, []
    for t, w in words:  # group transcript into ~5 s lines
        if cur_t is None or t - cur_t >= 5:
            if cur:
                lines.append(f"[{cur_t:.0f}] {' '.join(cur)}")
            cur_t, cur = t, []
        cur.append(w)
    if cur:
        lines.append(f"[{cur_t:.0f}] {' '.join(cur)}")
    step = max(1, len(chat) // 30)
    chat_lines = [f"[{t:.0f}] {m[:120]}" for t, m in chat[::step][:30]]
    counts = Counter(chat_key(m) for _, m in chat)
    top = [f'"{k[:40]}" x{n}' for k, n in counts.most_common(8) if k and n >= 3]
    return (
        (f"Stream/video: {context}\n" if context else "") +
        f"Window: {start:.0f}s to {end:.0f}s. Clip length must be {min_len:.0f}-{max_len:.0f}s.\n"
        f"Hype z-score at peak: {hype:.1f} (>2 is unusual). Chat messages in window: {len(chat)}.\n\n"
        f"TRANSCRIPT:\n" + ("\n".join(lines) or "(no speech)") +
        "\n\nMOST REPEATED CHAT: " + (", ".join(top) or "(none)") +
        "\n\nCHAT SAMPLE:\n" + ("\n".join(chat_lines) or "(no chat)")
    )


def sanitize(data: dict, start: float, end: float, min_len: float, max_len: float) -> Judgement:
    def num(k, default):
        try:
            return float(data.get(k, default))
        except (TypeError, ValueError):
            return float(default)

    s = min(max(num("start", start), start), end)
    e = min(max(num("end", end), start), end)
    if e - s < min_len:  # grow around the middle; if one side hits the window edge, grow the other way
        mid = (s + e) / 2
        s, e = max(start, mid - min_len / 2), min(end, mid + min_len / 2)
        if e - s < min_len:
            s, e = (s, min(end, s + min_len)) if s == start else (max(start, e - min_len), e)
    if e - s > max_len:
        e = s + max_len
    cat = str(data.get("category", "other")).lower().strip()
    return Judgement(
        score=max(0.0, min(10.0, num("score", 0))),
        category=cat if cat in CATEGORIES else "other",
        start=s, end=e,
        title=no_links(str(data.get("title", "")))[:100],   # AI text gets published: no injected links
        hook=no_links(str(data.get("hook", "")))[:60],
        reason=str(data.get("reason", "")).strip()[:300],
    )


async def judge(start: float, end: float, words, chat, hype: float, min_len: float, max_len: float,
                context: str = "") -> Judgement:
    prompt = build_prompt(start, end, words, chat, hype, max_len, min_len, context)
    try:
        data = await chat_json(SYSTEM, prompt)
    except LLMError as e:
        log.warning("LLM judge failed (%s); using signal score only", e)
        j = sanitize({}, start, end, min_len, max_len)
        j.score = -1.0  # marker: no LLM opinion
        return j
    return sanitize(data, start, end, min_len, max_len)
