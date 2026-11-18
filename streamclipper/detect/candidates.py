"""Turn raw per-second signals into a hype curve, then into candidate windows. Pure functions."""
from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..signals.chat import keyword_hits

CHAT_LAG = 4  # chat reacts a few seconds after the event
SPAM_WINDOW = 10  # seconds either side
SPAM_MIN = 5      # the same message this many times nearby = chat is reacting to something


def chat_key(text: str) -> str:
    """'LLLL vape!!' and 'L VAPEE' are the same reaction."""
    t = re.sub(r"[^\w\s]", "", text.upper())
    t = re.sub(r"(.)\1+", r"\1", t)
    return " ".join(t.split())


SPAM_OVER_USUAL = 2.0   # a burst must be this many times the message's usual rate in this chat

# viewers talking about the stream itself, not what happened in it
_COMPLAINT = re.compile(r"\b(LAG|LAGING|LAGY|BUFER|BUFERING|FROZE|FROZEN|FREZE|FREZING|REFRESH|F5|"
                        r"PIXELATED|STREAM DIED|STREAM DOWN|DEAD STREAM|NO SOUND|NO AUDIO|AUDIO DESYNC)\b")


def is_complaint(key: str) -> bool:
    """key is a chat_key(): repeated letters are already collapsed ("LAGGGG" -> "LAG")."""
    return bool(_COMPLAINT.search(key))


def spam_flags(chat: list[tuple]) -> list[bool]:
    """True for each message repeated (normalized) at least SPAM_MIN times within +-SPAM_WINDOW s, and at
    least SPAM_OVER_USUAL times its usual rate in this chat ("W" in a chat that always spams "W" is
    background, not a reaction). chat items are (t, text) or (t, text, user); with users, it takes
    SPAM_MIN different people (one viewer pasting the same self-promo six times is not the audience)."""
    keys = [chat_key(m[1]) for m in chat]
    by_key: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for i, (m, k) in enumerate(zip(chat, keys)):
        if k:
            by_key[k].append((m[0], m[2] if len(m) > 2 else str(i)))
    for v in by_key.values():
        v.sort()
    times = {k: [t for t, _ in v] for k, v in by_key.items()}
    span = max(60.0, (max(m[0] for m in chat) - min(m[0] for m in chat)) if chat else 0.0)
    out = []
    for m, k in zip(chat, keys):
        if not k or is_complaint(k):
            out.append(False)
            continue
        ts, t = times[k], m[0]
        near = by_key[k][bisect_left(ts, t - SPAM_WINDOW):bisect_right(ts, t + SPAM_WINDOW)]
        usual = len(ts) * (2 * SPAM_WINDOW) / span
        out.append(len({u for _, u in near}) >= SPAM_MIN and len(near) >= SPAM_OVER_USUAL * usual)
    return out


@dataclass
class Window:
    start: float
    end: float
    peak: float
    hype: float          # z-score at the peak
    manual: bool = False


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Moving average. Edge-padded: zero padding would make the edges look like loud spikes (dBFS < 0)."""
    if k <= 1 or len(x) == 0:
        return x
    left, right = (k - 1) // 2, k // 2
    padded = np.pad(x, (left, right), mode="edge")
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def _robust_z(x: np.ndarray) -> np.ndarray:
    """z-score using median/MAD so a few huge spikes don't hide the rest."""
    if len(x) == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    if mad < 1e-6:
        std = x.std()
        if std < 1e-6:
            return np.zeros_like(x)
        return (x - x.mean()) / std
    return (x - med) / mad


def hype_curve(
    duration: int,
    audio_db: dict[int, float],
    scene: dict[int, float],
    chat: list[tuple[float, str]],
    weights: dict[str, float],
    keywords: list[str],
) -> np.ndarray:
    """Combined per-second hype z-score over [0, duration)."""
    n = max(int(duration), 0)
    if n == 0:
        return np.zeros(0)
    a = np.full(n, np.nan)
    s = np.zeros(n)
    for t, v in audio_db.items():
        if 0 <= t < n:
            a[t] = v
    for t, v in scene.items():
        if 0 <= t < n:
            s[t] = v
    if np.all(np.isnan(a)):
        a = np.full(n, -60.0)
    a = np.where(np.isnan(a), np.nanmedian(a), a)

    rate = np.zeros(n)
    kw = np.zeros(n)
    for msg, spam in zip(chat, spam_flags(chat)):
        t, text = msg[0], msg[1]
        i = int(t) - CHAT_LAG
        if 0 <= i < n and not is_complaint(chat_key(text)):   # "LAG"/"buffering" spikes aren't hype
            rate[i] += 1
            kw[i] += keyword_hits(text, keywords) + spam

    parts, wsum = [], 0.0
    crowd, cw = [], 0.0
    comp = {
        "audio": _robust_z(_smooth(a, 3)),
        "scene": np.clip(_smooth(s, 2) * 4, 0, 4),     # already 0..1, scale to z-ish
        "chat": _robust_z(_smooth(rate, 6)) if rate.any() else None,
        "keywords": _robust_z(_smooth(kw, 6)) if kw.any() else None,
    }
    for name, arr in comp.items():
        w = weights.get(name, 0.0)
        if arr is None or w <= 0:
            continue
        parts.append(w * np.clip(arr, -3, 8))
        wsum += w
        if name in ("chat", "keywords"):
            crowd.append(parts[-1])
            cw += w
    if not parts:
        return np.zeros(n)
    hype = np.sum(parts, axis=0) / wsum
    if crowd:
        # chat going off is the audience saying "that was good": game audio or a quiet streamer
        # shouldn't average that away
        hype = np.maximum(hype, np.sum(crowd, axis=0) / cw)
    return _smooth(hype, 3)


def find_peaks(hype: np.ndarray, z: float, min_gap: int = 20, lo: int = 0, hi: int | None = None) -> list[int]:
    """Greedy: highest seconds above z first, each blocking +-min_gap around it."""
    hi = len(hype) if hi is None else min(hi, len(hype))
    idx = [i for i in range(lo, hi) if hype[i] >= z]
    idx.sort(key=lambda i: hype[i], reverse=True)
    taken: list[int] = []
    for i in idx:
        if all(abs(i - j) >= min_gap for j in taken):
            taken.append(i)
    return sorted(taken)


def windows_from_peaks(
    peaks: list[int], hype: np.ndarray, duration: float, pad_before: float, pad_after: float,
    manual: list[float] | None = None, split_gap: float = 20.0, scene: np.ndarray | None = None,
) -> list[Window]:
    """Pad each peak into a window. Overlapping windows whose peaks are closer than split_gap are the
    same event and merge; peaks further apart are separate highlights, so their windows are split
    halfway between the peaks (two dunks 26 s apart stay two clips, not one long blob)."""
    wins = [Window(max(0, p - pad_before), min(duration, p + pad_after), p, float(hype[p])) for p in peaks]
    for m in manual or []:
        # "clip that" means the moment just happened: take the minute before it
        wins.append(Window(max(0, m - 50), min(duration, m + 5), max(0, m - 10),
                           float(hype[int(min(m, len(hype) - 1))]) if len(hype) else 0.0, manual=True))
    wins.sort(key=lambda w: w.peak)
    merged: list[Window] = []
    for w in wins:
        if merged and w.start <= merged[-1].end:
            last = merged[-1]
            if w.peak - last.peak < split_gap:
                keep_peak = last if last.hype >= w.hype else w
                merged[-1] = Window(last.start, max(last.end, w.end), keep_peak.peak, keep_peak.hype,
                                    last.manual or w.manual)
                continue
            mid = split_point(last.peak, w.peak, scene)
            merged[-1] = Window(last.start, max(mid, last.peak + 3), last.peak, last.hype, last.manual)
            w = Window(min(mid, w.peak - 3), w.end, w.peak, w.hype, w.manual)
        merged.append(w)
    return merged


def split_point(p1: float, p2: float, scene: np.ndarray | None, margin: int = 3) -> float:
    """Where to split two neighbouring highlights: at the strongest shot change between them (the
    transition between items, e.g. a countdown card), else halfway."""
    lo, hi = int(p1) + margin, int(p2) - margin
    if scene is not None and hi > lo and hi <= len(scene):
        seg = scene[lo:hi]
        if len(seg) and seg.max() >= 0.3:
            return float(lo + int(seg.argmax()))
    return (p1 + p2) / 2


def signal_score(z: float) -> float:
    """Map a hype z-score to 0-10."""
    return float(max(0.0, min(10.0, z * 2.5)))
