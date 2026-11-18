"""Clip-selection modes and what each one produces."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    moment_shorts: bool        # each good moment -> a Short
    compilation: bool          # best moments -> one long video at stream end
    continuous_long: bool      # stream split into long parts
    continuous_short: bool     # stream split into Shorts-length parts
    detect_moments: bool


def plan_for(mode: str, outputs: str) -> Plan:
    shorts = outputs in ("shorts", "both")
    long = outputs in ("long", "both")
    if mode == "continuous":
        return Plan(False, False, long, shorts, False)
    if mode == "hybrid":
        # best moments as Shorts + the whole stream as long parts
        return Plan(True, False, True, False, True)
    return Plan(shorts, long, False, False, True)  # best_moments


def continuous_windows(done_until: float, duration: float, part_len: float, ended: bool,
                       min_last: float = 20) -> list[tuple[float, float]]:
    """Back-to-back parts with no gaps. While live, only full parts are emitted."""
    out = []
    t = done_until
    while t + part_len <= duration:
        out.append((t, t + part_len))
        t += part_len
    if ended and duration - t >= min_last:
        out.append((t, duration))
    return out


def select_best(scored: list, threshold: float, limit: int, already: int = 0) -> list:
    """scored: objects with .score. Highest first, above threshold, respecting the per-stream cap."""
    room = max(0, limit - already)
    good = sorted((m for m in scored if m.score >= threshold), key=lambda m: m.score, reverse=True)
    return good[:room]
