"""Ask the local LLM whether a candidate window is clip-worthy and where to trim it."""
from __future__ import annotations

import logging
import re
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
Answer the three checks first, then score. Reply with ONLY a JSON object:
{{"payoff": true or false (did something actually happen: a play, a fail, a joke's punchline, a surprise, a reveal,
            a big emotion? false for chatting, explaining, waiting or setting up),
 "reaction": "none", "some" or "strong" (how hard the streamer and chat react to it),
 "standalone": true or false (would a stranger get it without having watched the stream before?),
 "score": 0-10 on this scale:
   0-2 nothing happens: breaks, BRB, mic checks, setup, loading, greeting or thanking viewers, ad or sponsor reads,
       background or in-game dialogue nobody reacts to
   3-4 calm talk or an explanation (a tutorial, commentary) with no payoff
   5-6 a small moment: a mild laugh or a minor play, little reaction
   7-8 a clear moment: a payoff AND a strong reaction from the streamer or chat
   9-10 exceptional: something rare with a huge reaction
   Most windows are 0-4. Only give 7 or more when there is a payoff and a strong reaction.
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


# Emotes and words chat uses when something happens (matched case-insensitively, whole words).
_HYPE_WORDS = {"kekw", "lul", "lmao", "lmfao", "omegalul", "pog", "pogchamp", "poggers", "pogu", "monkas", "monkaw",
               "pepelaugh", "kappa", "xd", "wtf", "omg", "noway", "letsgo", "hype", "clutch", "insane", "sheesh",
               "crazy", "rip", "gg", "ez", "madge", "aware", "icant", "w", "l", "f"}
_ELONGATED = re.compile(r"(\w)\1{2,}")              # "noooo", "www", "ㅋㅋㅋ" style laughs
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
EXCITED_SHARE = 0.2      # below this share of excited messages, chat is calm
MIN_CHAT_FOR_EVIDENCE = 6
STRONG_HYPE_Z = 3.0      # audio/scene hype this high counts as a reaction even with calm chat


def is_excited(msg: str, keywords: frozenset[str] | set[str] = frozenset()) -> bool:
    """Does one chat message look like a reaction (emote, caps, !, stretched letters, emoji, hype word)?"""
    m = msg.strip()
    if not m:
        return False
    if "!" in m or _EMOJI.search(m) or _ELONGATED.search(m):
        return True
    words = re.findall(r"[^\W_]+", m)
    if any(len(w) >= 2 and w.isupper() and w.isalpha() for w in words):
        return True
    norm = {w.lower() for w in words} | {re.sub(r"\s+", "", m.lower())}
    return bool(norm & (_HYPE_WORDS | set(keywords)))


def chat_excitement(chat: list[tuple[float, str]]) -> float:
    """Share of chat messages that look excited (0-1)."""
    if not chat:
        return 0.0
    from ..config import get_settings

    kws = {k.lower().replace(" ", "") for k in get_settings().detect.keywords}
    return sum(is_excited(m, kws) for _, m in chat) / len(chat)


def evidence_cap(score: float, excitement: float, n_chat: int, hype: float) -> float:
    """Plenty of chat but it's calm, and the audio isn't spiking either: nobody reacted, so it's not a clip,
    whatever the model says (small models call 'ok', 'yes', 'nice' a strong reaction)."""
    if n_chat >= MIN_CHAT_FOR_EVIDENCE and excitement < EXCITED_SHARE and hype < STRONG_HYPE_Z:
        return min(score, 4.5)
    return score


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
    ex = chat_excitement(chat)
    level = "none" if not chat else "high" if ex >= 0.5 else "medium" if ex >= EXCITED_SHARE else "low"
    return (
        (f"Stream/video: {context}\n" if context else "") +
        f"Window: {start:.0f}s to {end:.0f}s. Clip length must be {min_len:.0f}-{max_len:.0f}s.\n"
        f"Hype z-score at peak: {hype:.1f} (>2 is unusual). Chat messages in window: {len(chat)}.\n"
        f"Chat excitement: {level} ({ex:.0%} of messages are emotes, caps, laughs or hype words). Calm chat "
        f"replies like 'ok', 'yes' or 'hi' are not a reaction.\n\n"
        f"TRANSCRIPT:\n" + ("\n".join(lines) or "(no speech)") +
        "\n\nMOST REPEATED CHAT: " + (", ".join(top) or "(none)") +
        "\n\nCHAT SAMPLE:\n" + ("\n".join(chat_lines) or "(no chat)")
    )


def _flag(v) -> bool | None:
    if isinstance(v, bool):
        return v
    t = str(v).strip().lower()
    return True if t in ("true", "yes", "1") else False if t in ("false", "no", "0") else None


def calibrate(score: float, data: dict) -> float:
    """Small models rate almost everything 7-8. Hold the score to what the model's own answers support:
    no payoff caps it at 4.5, no reaction at 4.5, some reaction at 7, and needing context costs a point."""
    score = max(0.0, min(10.0, score))
    reaction = str(data.get("reaction", "")).strip().lower()
    if _flag(data.get("payoff")) is False:
        score = min(score, 4.5)
    if reaction == "none":
        score = min(score, 4.5)
    elif reaction == "some":
        score = min(score, 7.0)
    if _flag(data.get("standalone")) is False:
        score = max(0.0, score - 1.0)
    return round(score, 2)


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
        score=calibrate(num("score", 0), data),
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
    j = sanitize(data, start, end, min_len, max_len)
    j.score = evidence_cap(j.score, chat_excitement(chat), len(chat), hype)
    return j
