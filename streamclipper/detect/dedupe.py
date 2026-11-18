"""Drop clips that overlap or say the same thing as clips we already made."""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9']+")


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def jaccard(t1: str, t2: str) -> float:
    a, b = set(_WORD.findall(t1.lower())), set(_WORD.findall(t2.lower()))
    if len(a) < 4 or len(b) < 4:
        return 0.0
    return len(a & b) / len(a | b)


def is_duplicate(span: tuple[float, float], text: str,
                 same_stream: list[tuple[float, float]], recent_texts: list[str],
                 max_iou: float = 0.3, max_jaccard: float = 0.7) -> bool:
    if any(iou(span, s) > max_iou for s in same_stream):
        return True
    return any(jaccard(text, t) > max_jaccard for t in recent_texts)
