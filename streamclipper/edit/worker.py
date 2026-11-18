"""Picks pending clips, renders them, writes metadata, marks them ready for publishing."""
from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import case
from sqlmodel import select

from ..config import get_settings
from ..db import session
from ..detect.llm import LLMError, chat_json
from ..ffmpeg_utils import duration as probe_duration
from ..models import Channel, Clip, Moment, Stream
from ..notify import notify
from ..publish.metadata import build_metadata, chapters, credit_line
from . import subtitles as subs
from .compile import compile_moments
from .render import render_span
from .thumbnail import make_thumbnail

log = logging.getLogger(__name__)
MAX_ATTEMPTS = 3


async def translate_lines(lines: list[str], lang: str) -> list[str] | None:
    if not lines:
        return None
    try:
        data = await chat_json(
            'Translate each subtitle line. Reply ONLY with JSON {"lines": [...]} with the same number of lines.',
            f"Target language: {lang}\n" + "\n".join(f"{i}. {x}" for i, x in enumerate(lines)),
        )
        out = [str(x) for x in data.get("lines", [])]
        return out if len(out) == len(lines) else None
    except LLMError as e:
        log.warning("translation failed: %s", e)
        return None


def _numbered(title: str, n: int) -> str:
    tag = " #shorts" if title.endswith(" #shorts") else ""
    base = title[: -len(tag)] if tag else title
    return f"{base[: 100 - len(tag) - 6]} ({n}){tag}"


def hottest_second(stream_id: int, a: float, b: float) -> float:
    """Most hyped second in [a, b) (thumbnail frame); middle of the span if no data."""
    from ..detect.engine import load_hype

    try:
        h = load_hype(stream_id, b)
        lo, hi = int(a), min(int(b), len(h))
        if hi - lo > 4:
            seg = h[lo + 2:hi - 2]  # avoid the very edges
            return float(lo + 2 + int(seg.argmax()))
    except Exception:
        pass
    return (a + b) / 2


def next_pending() -> Clip | None:
    with session() as db:
        clip = db.exec(select(Clip).join(Stream, Stream.id == Clip.stream_id)
                       .where(Clip.status == "pending")
                       # clips from streams that are still live render first (continuous while live)
                       .order_by(case((Stream.status == "live", 0), else_=1), Clip.score.desc(), Clip.id)).first()
        if clip:
            clip.status = "rendering"
            db.add(clip)
            db.commit()
        return clip


async def render_clip(clip: Clip) -> None:
    s = get_settings()
    with session() as db:
        ch = db.get(Channel, clip.channel_id)
        stream = db.get(Stream, clip.stream_id)
        ids = [int(x) for x in clip.moment_ids.split(",") if x]
        moments = [db.get(Moment, i) for i in ids]
    moments = [m for m in moments if m]
    if not moments:
        raise RuntimeError("clip has no moments")
    out_dir = s.renders_dir / f"stream_{stream.id}"
    out = out_dir / f"clip_{clip.id}_{clip.format}.mp4"
    credit = credit_line(ch)
    extra_desc = ""
    thumb = ""

    if len(moments) == 1:
        m = moments[0]
        res = await render_span(stream.id, m.start, m.end, out, vertical=clip.format == "short",
                                hook=m.hook, credit=credit if clip.format == "short" or ch.permission != "own" else "",
                                trim=m.kind != "continuous", peak=m.peak or None)
        words, dur = res.words, res.end - res.start
        transcript = " ".join(w.text for w in words)
        # continuous parts: let the LLM title the content; "Part N" is appended afterwards
        title_hint = "" if m.kind == "continuous" else (clip.title or m.title)
        category = m.category or clip.category
    else:
        path, chaps, words = await compile_moments(stream.id, moments, out, credit=credit)
        dur = await probe_duration(path)
        transcript = " | ".join(m.transcript for m in moments)[:3000]
        title_hint = clip.title
        category = "compilation"
        extra_desc = chapters(chaps)
        best = max(moments, key=lambda m: m.score)
        bi = moments.index(best)
        at = (chaps[bi][0] if bi < len(chaps) else 0) + 3
        thumb = await make_thumbnail(path, at, best.hook or best.title, out.with_suffix(".jpg"))

    # captions (original + optional translation) as sidecar SRT files
    srt = out.with_suffix(".srt")
    srt.write_text(subs.build_srt(words), encoding="utf-8")
    if ch.translate_to:
        tr = await translate_lines(subs.srt_chunks(words), ch.translate_to)
        if tr:
            out.with_suffix(f".{ch.translate_to}.srt").write_text(subs.build_srt(words, lines=tr), encoding="utf-8")

    title, desc, tags = await build_metadata(ch, clip.format, title_hint, transcript, category, extra_desc,
                                             context=stream.title or ch.name)
    if len(moments) == 1 and moments[0].kind == "continuous":
        suffix = f" | Part {moments[0].part_no}"
        base = title[: -len(" #shorts")] if title.endswith(" #shorts") else title
        title = (base[: 100 - len(suffix) - 8] + suffix + (" #shorts" if clip.format == "short" else ""))

    # several Shorts from one stream must not share a title
    with session() as db:
        same = db.exec(select(Clip).where(Clip.stream_id == clip.stream_id, Clip.id != clip.id,
                                          Clip.title == title)).first()
    if same:
        n = 2
        with session() as db:
            while db.exec(select(Clip).where(Clip.stream_id == clip.stream_id,
                                             Clip.title == _numbered(title, n))).first():
                n += 1
        title = _numbered(title, n)

    if clip.format == "long" and len(moments) == 1:
        m = moments[0]
        at = hottest_second(stream.id, res.start, res.end) - res.start
        text = m.hook or title.replace(" #shorts", "").split(" | ")[0]
        thumb = await make_thumbnail(res.path, at, text, out.with_suffix(".jpg"))

    with session() as db:
        c = db.get(Clip, clip.id)
        c.path, c.srt_path, c.thumb_path = str(out), str(srt), thumb
        c.duration = float(dur or 0)
        c.title, c.description, c.tags = title, desc, ",".join(tags)
        c.category = category
        c.status = "ready"
        c.error = ""
        db.add(c)
        for m in moments:
            mm = db.get(Moment, m.id)
            mm.status = "rendered"
            db.add(mm)
        db.commit()
    await notify(f"🎬 Clip ready: {title} ({clip.format}, score {clip.score:.1f})")


async def render_one() -> bool:
    clip = next_pending()
    if not clip:
        return False
    try:
        await render_clip(clip)
    except Exception as e:
        log.exception("render failed for clip %s", clip.id)
        with session() as db:
            c = db.get(Clip, clip.id)
            c.attempts += 1
            c.error = str(e)[:1000]
            c.status = "failed" if c.attempts >= MAX_ATTEMPTS else "pending"
            db.add(c)
            db.commit()
        if clip.attempts + 1 >= MAX_ATTEMPTS:
            await notify(f"⚠️ Render failed for clip {clip.id}: {str(e)[:200]}")
    return True


def recover_interrupted() -> None:
    """After a crash/restart, clips stuck in 'rendering' go back to 'pending'."""
    with session() as db:
        for c in db.exec(select(Clip).where(Clip.status == "rendering")).all():
            c.status = "pending"
            db.add(c)
        db.commit()
