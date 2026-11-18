"""Per-stream processing: segments -> signals/transcript -> moments -> pending clips."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from datetime import timedelta

from sqlmodel import select

from ..config import get_settings
from ..db import session
from ..gpu import gpu_lock
from ..models import (CategoryWeight, ChatMsg, Channel, Clip, HypePoint, Moment, Segment, Stream,
                      Word, utcnow)
from ..capture.twitch_ads import load_ad_spans, overlaps_ad
from ..signals.audio import load_pcm, loudness_per_second
from ..signals.scene import scene_per_second
from ..transcribe.whisper import transcribe_pcm
from . import candidates as cand
from .dedupe import is_duplicate
from .llm_judge import judge, sanitize
from .modes import continuous_windows, plan_for, select_best

log = logging.getLogger(__name__)

WARMUP_S = 120  # seconds of stream needed before live moment detection trusts the hype baseline
GLITCH_LOOKBACK_S = 20  # a broadcast glitch this long before a chat spike is what chat is reacting to


def combine(llm: float, z: float, peak_z: float, cat_weight: float = 1.0,
            bonus_per_z: float = 0.5, max_bonus: float = 1.5) -> float:
    """Final 0-10 score. The LLM's rating is the base (so threshold 7 = "the LLM says it's a 7");
    passing the hype trigger is what made this a candidate, so hype only ADDS a bonus for spikes
    above the trigger (+0 at the trigger, up to +max_bonus). Without the LLM, the hype score alone."""
    if llm < 0:  # LLM unavailable
        return round(cand.signal_score(z), 2)
    bonus = min(max_bonus, max(0.0, (z - peak_z) * bonus_per_z))
    return round(min(10.0, llm * cat_weight + bonus), 2)


# --------------------------------------------------------------------------- segments
async def process_segment(seg: Segment, language: str) -> None:
    import asyncio

    try:
        pcm = await load_pcm(seg.path)
    except Exception as e:
        log.warning("segment %s unreadable: %s", seg.path, e)
        pcm = None
    loud = loudness_per_second(pcm) if pcm is not None else []
    if seg.end - seg.start > 3 and len(loud) < (seg.end - seg.start) * 0.8:
        log.warning("segment %s: audio decoded for only %ds of %.0fs (damaged audio?); moments in this "
                    "part can't be detected from sound", Path(seg.path).name, len(loud), seg.end - seg.start)
    scene = await scene_per_second(seg.path, len(loud)) if loud else []
    words = []
    if pcm is not None and len(pcm):
        async with gpu_lock():
            try:
                words = await asyncio.to_thread(transcribe_pcm, pcm, language)
            except Exception as e:
                log.error("transcription failed for %s: %s", seg.path, e)
    base = seg.start
    with session() as db:
        for i, db_v in enumerate(loud):
            db.add(HypePoint(stream_id=seg.stream_id, t=int(base) + i, audio=db_v,
                             scene=scene[i] if i < len(scene) else 0.0))
        for w in words:
            db.add(Word(stream_id=seg.stream_id, start=base + w.start, end=base + w.end, text=w.text))
        s = db.get(Segment, seg.id)
        s.processed = True
        db.add(s)
        db.commit()


def processed_until(stream_id: int) -> float:
    """End offset of the contiguous run of processed segments."""
    with session() as db:
        segs = db.exec(select(Segment).where(Segment.stream_id == stream_id).order_by(Segment.idx)).all()
    t = 0.0
    for s in segs:
        if not s.processed:
            break
        t = s.end
    return t


def all_processed(stream_id: int) -> bool:
    with session() as db:
        return db.exec(select(Segment).where(Segment.stream_id == stream_id,
                                             Segment.processed == False)).first() is None  # noqa: E712


# --------------------------------------------------------------------------- helpers
def words_between(stream_id: int, a: float, b: float) -> list[Word]:
    with session() as db:
        return list(db.exec(select(Word).where(Word.stream_id == stream_id, Word.start >= a, Word.start < b)
                            .order_by(Word.start)).all())


def chat_between(stream_id: int, a: float, b: float) -> list[ChatMsg]:
    with session() as db:
        return list(db.exec(select(ChatMsg).where(ChatMsg.stream_id == stream_id, ChatMsg.t >= a, ChatMsg.t < b)
                            .order_by(ChatMsg.t)).all())


def load_hype(stream_id: int, duration: float):
    cfg = get_settings().detect
    with session() as db:
        pts = db.exec(select(HypePoint).where(HypePoint.stream_id == stream_id)).all()
        chat = db.exec(select(ChatMsg.t, ChatMsg.text, ChatMsg.user).where(ChatMsg.stream_id == stream_id)).all()
    audio = {p.t: p.audio for p in pts}
    scene = {p.t: p.scene for p in pts}
    return cand.hype_curve(int(duration), audio, scene, [(t, x, u) for t, x, u in chat], cfg.weights, cfg.keywords)


def load_scene(stream_id: int, duration: float):
    """Per-second scene-change score (0..1) for choosing split points between moments."""
    import numpy as np

    n = max(int(duration), 0)
    arr = np.zeros(n)
    with session() as db:
        for t, sc in db.exec(select(HypePoint.t, HypePoint.scene).where(HypePoint.stream_id == stream_id)).all():
            if 0 <= t < n:
                arr[t] = sc
    return arr


def trim_overlaps(wins: list, spans: list[tuple[float, float]], min_len: float) -> list:
    """Shrink new windows so they never overlap moments we already have; drop ones left too short.
    Manual requests are kept as they are (the user asked for that exact moment)."""
    out = []
    for w in wins:
        if not w.manual:
            for a, b in spans:
                if a < w.end and w.start < b:
                    if w.peak < a:
                        w = cand.Window(w.start, min(w.end, a), w.peak, w.hype, w.manual)
                    else:
                        w = cand.Window(max(w.start, b), w.end, w.peak, w.hype, w.manual)
        if w.manual or w.end - w.start >= min_len:
            out.append(w)
    return out


def ensure_min_len(a: float, b: float, min_len: float, lo: float, hi: float) -> tuple[float, float]:
    """Widen [a, b] to min_len using the footage around it (within [lo, hi]); a too-short window, an
    LLM trim or a split between close peaks must never produce a Short below the minimum."""
    if b - a >= min_len:
        return a, b
    need = min_len - (b - a)
    a2 = max(lo, a - need / 2)
    b2 = min(hi, b + need - (a - a2))
    a2 = max(lo, b2 - min_len) if b2 - a2 < min_len else a2
    return a2, b2


def is_dead_air(stream_id: int, a: float, b: float) -> bool:
    """Frozen AND silent for most of [a, b): signal loss slates, BRB screens, ad-break cards, a paused
    stream. Never worth a clip. (A quiet but moving scene, or a still image with talking, is fine.)
    Also true when [a, b) overlaps a Twitch ad break recorded by the yt-dlp capture (loud, not frozen)."""
    with session() as db:
        st = db.get(Stream, stream_id)
        if st and overlaps_ad(load_ad_spans(st.dir), a, b):
            return True
        pts = db.exec(select(HypePoint.audio, HypePoint.scene).where(
            HypePoint.stream_id == stream_id, HypePoint.t >= int(a), HypePoint.t < int(b))).all()
    if len(pts) < max(5, (b - a) * 0.5):
        return False
    dead = sum(1 for audio, scene in pts if audio < -50 and scene < 0.02)
    return dead / len(pts) >= 0.85


def broken_frames(edge_maps) -> int:
    """How many frames lost the stream's static overlay (webcam border, HUD, chat box, counters).
    Those edges sit in the same place in nearly every frame; a broadcast glitch (corrupted video from the
    streamer's encoder, which decodes without errors) wipes them out. A blurred pause menu keeps them.
    edge_maps: boolean edge masks of equal shape. 0 when there's no overlay to judge by."""
    import numpy as np

    if len(edge_maps) < 5:
        return 0
    e = np.array(edge_maps)
    static = e.mean(axis=0) >= 0.7
    if static.sum() < max(20, 0.0013 * static.size):   # ~300 px at 640x360: too little overlay to judge
        return 0
    kept = (e & static).sum(axis=(1, 2)) / static.sum()
    return int((kept < 0.4).sum())


def has_broken_video(stream_id: int, a: float, b: float) -> bool:
    """Sample [a, b) at 1 fps; True if the broadcast itself glitched for part of it (>= 3 frames and
    >= 10% of them). Viewers spam "WTF"/"F" at a glitch, so it looks like hype."""
    import cv2
    import numpy as np

    with session() as db:
        segs = db.exec(select(Segment).where(Segment.stream_id == stream_id, Segment.end > a,
                                             Segment.start < b).order_by(Segment.start)).all()
    maps = []
    for seg in segs:
        cap = cv2.VideoCapture(seg.path)
        t = max(a, seg.start)
        while t < min(b, seg.end):
            cap.set(cv2.CAP_PROP_POS_MSEC, (t - seg.start) * 1000)
            ok, f = cap.read()
            if ok:
                g = cv2.cvtColor(cv2.resize(f, (640, 360)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
                maps.append(mag > np.percentile(mag, 90))
            t += 1.0
        cap.release()
    bad = broken_frames(maps)
    return bad >= 3 and bad >= 0.1 * len(maps)


def cat_weight(category: str) -> float:
    with session() as db:
        cw = db.exec(select(CategoryWeight).where(CategoryWeight.category == category)).first()
    return cw.weight if cw else 1.0


# --------------------------------------------------------------------------- detection
def _max_len(plan) -> float:
    cfg = get_settings().detect
    return cfg.max_short if plan.moment_shorts else cfg.max_long_moment


async def detect_step(stream: Stream, ch: Channel) -> None:
    cfg = get_settings().detect
    plan = plan_for(ch.mode, ch.outputs)
    upto = processed_until(stream.id)
    ended = stream.status != "live" and all_processed(stream.id)

    if plan.detect_moments:
        lo = stream.detected_until
        hi = upto if ended else upto - cfg.pad_after
        manual_raw = [float(x) for x in stream.manual_requests.split(",") if x.strip()]
        manual_ready = [m for m in manual_raw if m <= upto]
        # z-scores need a baseline: in the first minutes any bump looks extreme. Wait (the first pass
        # then covers the start too) unless the stream is already over or someone asked for a clip.
        warm = ended or upto >= WARMUP_S or manual_ready
        if warm and (hi - lo >= 15 or (ended and hi > lo) or manual_ready):
            hype = load_hype(stream.id, upto)
            peaks = cand.find_peaks(hype, cfg.peak_z, cfg.min_peak_gap, lo=int(lo), hi=int(max(lo, hi)))
            # live detection runs in passes: an event's hype often keeps rising past the previous pass's
            # edge. A peak that close to an already-judged moment is the same event, not a new one.
            with session() as db:
                prev = db.exec(select(Moment).where(Moment.stream_id == stream.id,
                                                    Moment.kind.in_(["moment", "manual"]))).all()
            peaks = [p for p in peaks if all(abs(p - m.peak) >= cfg.min_peak_gap for m in prev)]
            wins = cand.windows_from_peaks(peaks, hype, upto, cfg.pad_before, cfg.pad_after, manual_ready,
                                           split_gap=cfg.min_peak_gap, scene=load_scene(stream.id, upto))
            wins = trim_overlaps(wins, [(m.start, m.end) for m in prev], cfg.min_clip)
            await _judge_windows(stream, ch, plan, wins)
            with session() as db:
                st = db.get(Stream, stream.id)
                st.detected_until = max(st.detected_until, hi)
                left = [m for m in manual_raw if m > upto]
                st.manual_requests = ",".join(f"{m:.1f}" for m in left)
                db.add(st)
                db.commit()
                stream.detected_until = st.detected_until

        if ended:
            await _second_pass(stream, ch, plan, upto)

    else:
        stream.detected_until = upto  # nothing to detect; lets retention free old segments
        with session() as db:
            st = db.get(Stream, stream.id)
            st.detected_until = upto
            db.add(st)
            db.commit()

    if plan.continuous_long:
        _continuous(stream, ch, upto, ended, ch.continuous_len_s, "long", "continuous_until")
    if plan.continuous_short:
        _continuous(stream, ch, upto, ended, min(ch.continuous_short_len_s, cfg.max_short),
                    "short", "continuous_short_until")

    if ended:
        await _finalize(stream, ch, plan)


async def _judge_windows(stream: Stream, ch: Channel, plan, wins: list[cand.Window]) -> None:
    cfg = get_settings().detect
    max_len = _max_len(plan)
    since = utcnow() - timedelta(days=7)
    with session() as db:
        existing = db.exec(select(Moment).where(Moment.stream_id == stream.id,
                                                Moment.status.in_(["accepted", "rendered"]))).all()
        recent = db.exec(select(Moment.transcript).join(Stream, Stream.id == Moment.stream_id)
                         .where(Stream.channel_id == ch.id, Moment.created_at >= since,
                                Moment.status.in_(["accepted", "rendered"]))).all()
    spans = [(m.start, m.end) for m in existing]
    texts = list(recent)
    scored: list[Moment] = []
    for w in wins:
        if not w.manual and is_dead_air(stream.id, w.start, w.end):
            log.info("%s: skipped %.0f-%.0fs (frozen and silent)", ch.name, w.start, w.end)
            continue
        # look back too: viewers watch with a delay, so their "F"/"WTF" lands after the glitch itself
        if not w.manual and await asyncio.to_thread(has_broken_video, stream.id,
                                                    max(0.0, w.start - GLITCH_LOOKBACK_S), w.end):
            log.info("%s: skipped %.0f-%.0fs (the broadcast glitched: corrupted video)", ch.name, w.start, w.end)
            continue
        words = words_between(stream.id, w.start, w.end)
        chat = chat_between(stream.id, w.start, w.end + cand.CHAT_LAG)
        if len(words) < 5 and len(chat) < 5 and not w.manual:
            # nothing for a text model to read: its score would be a guess based on the title alone.
            # Let the hype signal decide (only clearly hyped visual moments pass).
            j = sanitize({}, w.start, w.end, cfg.min_clip, max_len)
            j.score = -1.0
            j.reason = "no speech or chat in this window: scored on hype only"
        else:
            j = await judge(w.start, w.end, [(x.start, x.text) for x in words], [(c.t, c.text) for c in chat],
                            w.hype, cfg.min_clip, max_len, context=f"{stream.title or ch.name} (channel: {ch.name})")
        j.start, j.end = ensure_min_len(j.start, j.end, cfg.min_clip, 0.0, processed_until(stream.id))
        sig = cand.signal_score(w.hype)
        score = combine(j.score, w.hype, cfg.peak_z, cat_weight(j.category), cfg.hype_bonus_per_z,
                        cfg.hype_bonus_max)
        if w.manual:
            score = max(score, ch.threshold)
        text = " ".join(x.text for x in words if j.start <= x.start < j.end)
        m = Moment(stream_id=stream.id, kind="manual" if w.manual else "moment", start=j.start, end=j.end,
                   peak=w.peak, signal_score=sig, llm_score=j.score, score=score, category=j.category,
                   title=j.title, hook=j.hook, reason=j.reason, transcript=text)
        if is_duplicate((m.start, m.end), text, spans, texts):
            m.status = "rejected"
            m.reason = f"duplicate. {m.reason}"
        else:
            scored.append(m)
        with session() as db:
            db.add(m)
            db.commit()
        log.info("%s moment %.0f-%.0fs score %.1f (llm %.1f sig %.1f) %s", ch.name, m.start, m.end,
                 m.score, j.score, sig, m.title)

    already = len(existing)
    manual = [m for m in scored if m.kind == "manual"]
    best = select_best([m for m in scored if m.kind != "manual"], ch.threshold,
                       ch.max_clips_per_stream, already + len(manual))
    chosen = {id(m) for m in manual + best}
    with session() as db:
        for m in scored:
            if id(m) in chosen:
                m.status = "accepted"
                if plan.moment_shorts:
                    db.add(Clip(stream_id=stream.id, channel_id=ch.id, moment_ids=str(m.id), format="short",
                                title=m.title, category=m.category, score=m.score))
            elif m.score < ch.threshold:
                m.status = "rejected"
            # else: stays "candidate" (good but over the per-stream cap; may join a compilation)
            db.add(m)
        db.commit()


async def _second_pass(stream: Stream, ch: Channel, plan, upto: float, top_n: int = 5) -> None:
    """When a stream ends, look again at borderline peaks. Hype is relative to the whole stream, so a
    real highlight can sit just under the trigger when the rest is loud (or, for calm streams with no
    chat, nothing crosses it at all). Judge up to top_n relative peaks (z >= 0.5 when nothing was
    found, else >= 1.0) that don't overlap existing moments; the score threshold still decides."""
    cfg = get_settings().detect
    hype = load_hype(stream.id, upto)
    if len(hype) < cfg.min_clip * 2:
        return
    with session() as db:
        spans = [(m.start, m.end) for m in db.exec(select(Moment).where(
            Moment.stream_id == stream.id, Moment.kind.in_(["moment", "manual"]))).all()]
    z = 1.0 if spans else 0.5
    peaks = [p for p in cand.find_peaks(hype, z, cfg.min_peak_gap)
             if hype[p] < cfg.peak_z or not spans]                       # main pass already judged >= peak_z
    peaks = [p for p in peaks if not any(a - 3 <= p <= b + 3 for a, b in spans)]
    peaks = sorted(sorted(peaks, key=lambda p: hype[p], reverse=True)[:top_n])
    if not peaks:
        return
    log.info("%s: second pass, judging %d borderline peaks", ch.name, len(peaks))
    wins = cand.windows_from_peaks(peaks, hype, upto, cfg.pad_before, cfg.pad_after,
                                   split_gap=cfg.min_peak_gap, scene=load_scene(stream.id, upto))
    trimmed = trim_overlaps(wins, spans, cfg.min_clip)
    if trimmed:
        await _judge_windows(stream, ch, plan, trimmed)


def _continuous(stream: Stream, ch: Channel, upto: float, ended: bool, part_len: float, fmt: str,
                cursor: str) -> None:
    done = getattr(stream, cursor)
    wins = continuous_windows(done, upto, part_len, ended)
    if not wins:
        return
    queued = 0
    with session() as db:
        # number only the parts that are kept, so published parts never skip ("Part 1, Part 3")
        n = len(db.exec(select(Moment).where(Moment.stream_id == stream.id, Moment.kind == "continuous",
                                             Moment.status != "rejected",
                                             Moment.title.like(f"[{fmt}]%"))).all())
        for a, b in wins:
            dead = is_dead_air(stream.id, a, b)
            if dead:
                db.add(Moment(stream_id=stream.id, kind="continuous", start=a, end=b, status="rejected",
                              title=f"[{fmt}] skipped", score=0.0,
                              reason="frozen and silent (signal loss / BRB / paused)"))
                log.info("%s: %.0f-%.0fs skipped: frozen and silent", ch.name, a, b)
                continue
            n += 1
            m = Moment(stream_id=stream.id, kind="continuous", start=a, end=b, part_no=n,
                       status="accepted", title=f"[{fmt}] Part {n}", score=5.0)
            db.add(m)
            db.flush()
            db.add(Clip(stream_id=stream.id, channel_id=ch.id, moment_ids=str(m.id), format=fmt,
                        title=f"{stream.title or ch.name} - Part {n}", score=5.0, category="stream"))
            queued += 1
        st = db.get(Stream, stream.id)
        setattr(st, cursor, wins[-1][1])
        setattr(stream, cursor, wins[-1][1])
        db.add(st)
        db.commit()
    log.info("%s: %d continuous %s part(s) queued, %d skipped", ch.name, queued, fmt, len(wins) - queued)


async def _finalize(stream: Stream, ch: Channel, plan) -> None:
    rcfg = get_settings().render
    with session() as db:
        if plan.compilation:
            ms = db.exec(select(Moment).where(Moment.stream_id == stream.id,
                                              Moment.kind.in_(["moment", "manual"]),
                                              Moment.status.in_(["accepted", "rendered", "candidate"]))
                         .order_by(Moment.score.desc())).all()
            top = sorted(ms[: rcfg.compile_top_n], key=lambda m: m.start)
            if len(top) >= 3:
                db.add(Clip(stream_id=stream.id, channel_id=ch.id, format="long",
                            moment_ids=",".join(str(m.id) for m in top), category="compilation",
                            title=f"{ch.name} - Best moments of {stream.title or 'the stream'}",
                            score=sum(m.score for m in top) / len(top)))
                log.info("%s: compilation of %d moments queued", ch.name, len(top))
        st = db.get(Stream, stream.id)
        st.status = "finalized"
        db.add(st)
        db.commit()
    stream.status = "finalized"
