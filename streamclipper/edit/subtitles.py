"""Animated "karaoke" subtitles (ASS) + plain SRT. Pure functions, easy to test."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Word:
    start: float  # seconds, relative to clip start
    end: float
    text: str


def _ts(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _srt_ts(t: float) -> str:
    t = max(0.0, t)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _esc(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def censor(text: str, bad: list[str]) -> str:
    for b in bad:
        text = re.sub(rf"(?i)\b({re.escape(b)})\w*", lambda m: m.group(0)[0] + "*" * (len(m.group(0)) - 1), text)
    return text


def clean_words(words: list[Word]) -> list[Word]:
    """Whisper sometimes stretches a word over seconds of crowd noise ("to" lasting 5.6 s), which leaves
    one stale word on screen. Cap each word to a speakable length."""
    out = []
    for w in words:
        limit = max(0.8, 0.09 * len(w.text) + 0.5)
        out.append(Word(w.start, min(w.end, w.start + limit), w.text))
    return out


def chunk_words(words: list[Word], max_words: int, max_dur: float = 2.6, gap: float = 0.6) -> list[list[Word]]:
    chunks: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if cur and (len(cur) >= max_words or w.start - cur[-1].end > gap or w.end - cur[0].start > max_dur
                    or cur[-1].text.endswith((".", "?", "!"))):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    # don't leave one orphan word ("long.") alone on screen: glue it to the chunk it belongs to
    merged: list[list[Word]] = []
    for c in chunks:
        if (merged and len(c) == 1 and len(merged[-1]) <= max_words
                and c[0].start - merged[-1][-1].end <= gap
                and not merged[-1][-1].text.endswith((".", "?", "!"))):
            merged[-1] = merged[-1] + c
        else:
            merged.append(c)
    return merged


def _legacy_style(font: str, font_size: int, highlight: str):
    """Old keyword arguments -> a CaptionStyle (highlight given as ASS &HBBGGRR)."""
    from .caption_style import CaptionStyle

    bgr = highlight.upper().replace("&H", "").replace("&", "")[-6:].rjust(6, "0")
    return CaptionStyle(font=font, size=font_size, highlight_color=f"#{bgr[4:6]}{bgr[2:4]}{bgr[0:2]}")


def build_ass(words: list[Word], duration: float, vertical: bool, *, font: str = "Arial Black",
              font_size: int = 78, highlight: str = "&H0000F0FF", hook: str = "", credit: str = "",
              profanity: list[str] | None = None, keywords: list[str] | None = None, style=None) -> str:
    """style: a validated caption_style.CaptionStyle; without it the font/font_size/highlight arguments apply."""
    from .caption_style import FONTS, ass_color

    st = style or _legacy_style(font, font_size, highlight)
    w, h = (1080, 1920) if vertical else (1920, 1080)
    size = st.size if (vertical or style is None) else max(28, int(st.size * 0.7))
    outline = st.outline if vertical else max(0, round(st.outline * 0.7))
    align, margin_v = {
        "low": (2, 560 if vertical else 70), "middle": (5, 0), "top": (8, 420 if vertical else 60),
    }[st.position]
    max_words = st.words_per_line if vertical else max(5, st.words_per_line * 2 + 1)
    border_style, shadow = (3, 0) if st.box else (1, 2)
    font_name = st.font if st.font in FONTS else "Arial Black"   # never put unvalidated text in the header
    hook_size = int(size * 0.9)
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,{font_name},{size},{ass_color(st.text_color)},{ass_color(st.text_color)},{ass_color(st.outline_color)},&H80000000,-1,0,0,0,100,100,1,{st.angle},{border_style},{outline},{shadow},{align},60,60,{margin_v},1
Style: Hook,{font_name},{hook_size},&H00000000,&H00000000,&H0000F0FF,&H0000F0FF,-1,0,0,0,100,100,1,{st.angle},3,18,0,8,80,80,{220 if vertical else 60},1
Style: Credit,Arial,{int(size * 0.38)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,0,0,0,0,100,100,0,0,1,2,1,7,30,30,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev: list[str] = []
    highlight = ass_color(st.highlight_color)[4:] + "&"     # \c wants &HBBGGRR&
    highlight = "&H" + highlight
    lead = r"{\blur3}" if st.glow else ""
    pop_on = st.pop and vertical
    bad = profanity or []
    kws = {k.upper() for k in (keywords or [])}
    chunks = chunk_words(clean_words(words), max_words)
    for ci, chunk in enumerate(chunks):
        texts = [_esc(censor(x.text, bad)) for x in chunk]
        if vertical and st.uppercase:
            texts = [t.upper() for t in texts]
        # hold the last word until the next chunk if the pause is short (avoids flicker)
        nxt = chunks[ci + 1][0].start if ci + 1 < len(chunks) else None
        for i, word in enumerate(chunk):
            start = word.start
            if i + 1 < len(chunk):
                end = chunk[i + 1].start
            else:
                end = max(word.end, word.start + 0.25)
                if nxt is not None and nxt - end < 0.6:
                    end = nxt
            parts = []
            for j, t in enumerate(texts):
                reset = r"{\r" + (r"\blur3" if st.glow else "") + "}"
                if j == i:
                    pop = r"\fscx118\fscy118\t(0,90,\fscx108\fscy108)" if pop_on else ""
                    parts.append(rf"{{\c{highlight}{pop}}}{t}{reset}")
                elif t.strip(".,!?").upper() in kws:
                    parts.append(rf"{{\c&H4CFF00&}}{t}{reset}")
                else:
                    parts.append(t)
            ev.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Sub,,0,0,0,,{lead}{' '.join(parts)}")
    if hook:
        hk = _esc(hook.upper() if vertical else hook)
        ev.append(f"Dialogue: 1,{_ts(0)},{_ts(min(2.8, duration))},Hook,,0,0,0,,"
                  rf"{{\fad(120,250)\t(0,150,\fscx104\fscy104)}}{hk}")
    if credit:
        ev.append(f"Dialogue: 2,{_ts(0)},{_ts(duration)},Credit,,0,0,0,,{_esc(credit)}")
    return head + "\n".join(ev) + "\n"


def build_srt(words: list[Word], max_words: int = 8, lines: list[str] | None = None) -> str:
    """lines: optional replacement text per chunk (translation)."""
    out = []
    for i, chunk in enumerate(chunk_words(clean_words(words), max_words, max_dur=4.0), start=1):
        text = lines[i - 1] if lines and i - 1 < len(lines) else " ".join(w.text for w in chunk)
        out.append(f"{i}\n{_srt_ts(chunk[0].start)} --> {_srt_ts(chunk[-1].end)}\n{text}\n")
    return "\n".join(out)


def srt_chunks(words: list[Word], max_words: int = 8) -> list[str]:
    return [" ".join(w.text for w in c) for c in chunk_words(clean_words(words), max_words, max_dur=4.0)]
