"""Shot boundaries: so clips start where a shot starts and end before the next item begins."""
from __future__ import annotations

import re
from pathlib import Path

from ..ffmpeg_utils import FFmpegError, run

_PTS = re.compile(r"pts_time:([\d.]+)")


async def detect_cuts(concat: Path, start: float, dur: float, threshold: float = 0.3) -> list[float]:
    """Scene-cut times (seconds, relative to `start`) inside [start, start+dur) of a concat list."""
    try:
        text = await run(
            ["-hide_banner", "-f", "concat", "-safe", "0", "-i", str(concat), "-an", "-vf",
             f"trim=start={start:.3f}:duration={dur:.3f},setpts=PTS-STARTPTS,fps=12,scale=256:-2,"
             f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
            timeout=240,
        )
    except FFmpegError:
        return []
    cuts = sorted({round(float(m.group(1)), 2) for m in _PTS.finditer(text)})
    # merge flashes/dissolves: keep the first cut of any burst within 0.6 s
    out: list[float] = []
    for c in cuts:
        if not out or c - out[-1] > 0.6:
            out.append(c)
    return out


def choose_edges(a: float, b: float, peak: float, cuts: list[float], *, min_len: float,
                 max_len: float | None, back: float = 6.0, fwd: float = 4.0) -> tuple[float, float, bool, bool]:
    """Snap a clip's [a, b] to shot cuts (absolute times). Returns (a, b, start_on_cut, end_on_cut).

    Start: the cut that begins the shot the clip starts in (prefer at/just before a; the setup belongs in).
    End: the cut where the next shot after the payoff begins (prefer at/just before b, never inside the
    peak's own action: >= peak + 1 s), so the clip never runs into the next item's intro or an outro card.
    """
    def best(cands: list[float], target: float, late_penalty: float) -> float | None:
        if not cands:
            return None
        return min(cands, key=lambda c: abs(c - target) + (late_penalty if c > target else 0.0))

    s_cands = [c for c in cuts if a - back <= c <= min(a + fwd, peak - 1.0)]
    na = best(s_cands, a, 1.0)
    e_lo = max(peak + 1.0, (na if na is not None else a) + min_len)
    e_cands = [c for c in cuts if max(e_lo, b - back) <= c <= b + fwd]
    if max_len is not None:
        start = na if na is not None else a
        e_cands = [c for c in e_cands if c - start <= max_len]
    nb = best(e_cands, b, 1.5)
    if nb is not None:
        # don't end on a short fragment shot (the next item's title/countdown card, a logo sting):
        # if the shot right before nb is < 2.5 s and starts after the payoff, end before it instead
        earlier = [c for c in cuts if e_lo <= c < nb and nb - c < 2.5]
        if earlier:
            nb = min(earlier)
    return (na if na is not None else a, nb if nb is not None else b, na is not None, nb is not None)


def static_tail(motion: list[float], fps: float, min_run: float = 0.8, rel: float = 0.02) -> float | None:
    """Seconds (from the start of `motion`) where a trailing motionless run begins: end cards, logos,
    black frames, BRB screens. motion = mean abs frame difference per frame. None if the tail moves."""
    if not motion:
        return None
    ref = sorted(motion)[int(len(motion) * 0.75)] or 1.0       # typical "moving" level in this window
    i = len(motion)
    while i > 0 and motion[i - 1] <= max(0.35, ref * rel):   # truly frozen, not slow motion
        i -= 1
    run = (len(motion) - i) / fps
    return i / fps if run >= min_run and i > 0 else None
