"""Feedback loop: learn which clip categories get views and nudge future scoring (0.8x-1.2x)."""
from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict
from datetime import timedelta

from sqlmodel import select

from ..db import session
from ..models import CategoryWeight, Channel, Clip, utcnow
from . import scheduler
from .youtube_auth import youtube_client

log = logging.getLogger(__name__)


def compute_weights(rows: list[tuple[str, int]], min_samples: int = 3) -> dict[str, tuple[float, int]]:
    """rows: (category, views). Weight from mean log-views of a category vs the overall mean."""
    if not rows:
        return {}
    logs = [math.log1p(v) for _, v in rows]
    overall = sum(logs) / len(logs)
    by: dict[str, list[float]] = defaultdict(list)
    for (cat, v) in rows:
        by[cat].append(math.log1p(v))
    out = {}
    for cat, vals in by.items():
        if len(vals) < min_samples:
            continue
        diff = sum(vals) / len(vals) - overall
        out[cat] = (round(1 + 0.2 * math.tanh(diff), 3), len(vals))
    return out


def _fetch_views(yt, ids: list[str]) -> dict[str, int]:
    out = {}
    for i in range(0, len(ids), 50):
        resp = yt.videos().list(part="statistics", id=",".join(ids[i:i + 50])).execute()
        for item in resp.get("items", []):
            out[item["id"]] = int(item.get("statistics", {}).get("viewCount", 0))
    return out


async def refresh() -> None:
    cutoff = utcnow() - timedelta(hours=48)
    with session() as db:
        rows = db.exec(select(Clip, Channel.target_account).join(Channel, Channel.id == Clip.channel_id)
                       .where(Clip.status == "published", Clip.youtube_id != "",
                              Clip.publish_at <= cutoff)).all()
    by_acc: dict[str, list[Clip]] = defaultdict(list)
    for clip, acc in rows:
        by_acc[acc or "default"].append(clip)
    for acc, clips in by_acc.items():
        yt = await asyncio.to_thread(youtube_client, acc)
        if not yt:
            continue
        try:
            views = await asyncio.to_thread(_fetch_views, yt, [c.youtube_id for c in clips])
        except Exception as e:
            log.warning("analytics fetch failed: %s", e)
            continue
        scheduler.spend_quota(acc, math.ceil(len(clips) / 50))
        with session() as db:
            for c in clips:
                if c.youtube_id in views:
                    cc = db.get(Clip, c.id)
                    cc.views = views[c.youtube_id]
                    db.add(cc)
            db.commit()
    with session() as db:
        data = db.exec(select(Clip.category, Clip.views).where(Clip.status == "published",
                                                               Clip.format == "short")).all()
        for cat, (w, n) in compute_weights([(c or "other", v) for c, v in data]).items():
            cw = db.exec(select(CategoryWeight).where(CategoryWeight.category == cat)).first() \
                or CategoryWeight(category=cat)
            cw.weight, cw.samples = w, n
            db.add(cw)
        db.commit()
    log.info("analytics refreshed")
