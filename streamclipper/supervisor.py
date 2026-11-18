"""Runs every background loop: monitor, capture, signals, detection, render, publish, cleanup."""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
from datetime import timedelta
from pathlib import Path

from sqlalchemy import case
from sqlmodel import select

from .capture.recorder import Recorder
from .config import get_settings
from .db import session
from .detect import engine
from .edit import worker
from .models import Channel, Clip, Segment, Stream, utcnow
from .monitor.live_check import check_live
from .monitor.platforms import CHAT_PLATFORMS, detect_platform, guess_name, normalize_url
from .notify import notify
from .publish import analytics, uploader
from .signals.chat import ChatReader

log = logging.getLogger(__name__)


class Supervisor:
    def __init__(self) -> None:
        self.active: dict[int, tuple[Recorder, asyncio.Task, ChatReader | None, asyncio.Task | None]] = {}
        self.tasks: list[asyncio.Task] = []
        self.last_error: dict[int, str] = {}
        self.paused = False

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        from .bootstrap import run_checks

        self.tasks.append(asyncio.create_task(run_checks(), name="bootstrap"))
        worker.recover_interrupted()
        with session() as db:  # streams left "live" by a crash: their recorder is gone
            for st in db.exec(select(Stream).where(Stream.status == "live")).all():
                st.status = "ended"
                st.ended_at = st.ended_at or utcnow()
                db.add(st)
            db.commit()
        for fn in (self._monitor_loop, self._segment_loop, self._detect_loop, self._live_detect_loop,
                   self._render_loop,
                   self._publish_loop, self._retention_loop, self._analytics_loop):
            self.tasks.append(asyncio.create_task(self._guard(fn), name=fn.__name__))
        log.info("supervisor started")

    async def stop(self) -> None:
        for rec, _t, chat, _ct in self.active.values():
            rec.stop()
            if chat:
                chat.stop()
        for t in self.tasks:
            t.cancel()
        await asyncio.gather(*[t for _, t, _, _ in self.active.values()], return_exceptions=True)
        from .detect import llama_server

        llama_server.stop()

    async def _guard(self, fn) -> None:
        """Restart a loop if it crashes, so one bug never stops the whole tool."""
        while True:
            try:
                await fn()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("%s crashed; restarting in 15s", fn.__name__)
                await asyncio.sleep(15)

    # ------------------------------------------------------------------ capture
    def start_recording(self, ch: Channel, title: str = "", backend: str = "") -> Stream:
        cur = self.active.get(ch.id)
        if cur and not cur[1].done():
            # the monitor loop and a pasted link can both pass their "not active" check while
            # awaiting check_live; never run two recorders for one channel
            return cur[0].stream
        s = get_settings()
        with session() as db:
            st = Stream(channel_id=ch.id, title=title[:200])
            db.add(st)
            db.commit()
            stamp = st.started_at.strftime("%Y%m%d_%H%M%S")
            safe = re.sub(r"[^A-Za-z0-9_-]", "_", ch.name)[:40]
            st.dir = str(s.streams_dir / f"{safe}_{stamp}_{st.id}")
            db.add(st)
            db.commit()
        rec = Recorder(ch, st, backend)
        task = asyncio.create_task(rec.run(), name=f"rec-{ch.id}")
        chat = ctask = None
        if ch.platform in CHAT_PLATFORMS:
            chat = ChatReader(st, ch.platform, ch.url)
            ctask = asyncio.create_task(chat.run(), name=f"chat-{ch.id}")
        self.active[ch.id] = (rec, task, chat, ctask)
        return st

    def stop_recording(self, channel_id: int) -> None:
        if channel_id in self.active:
            rec, _t, chat, _ct = self.active[channel_id]
            rec.stop()
            if chat:
                chat.stop()

    def _reap(self) -> None:
        for cid, (_rec, task, chat, _ct) in list(self.active.items()):
            if task.done():
                if chat:
                    chat.stop()
                del self.active[cid]

    async def _monitor_loop(self) -> None:
        s = get_settings()
        disk_warned = False
        while True:
            self._reap()
            free = disk_free_gb()
            low = free < s.capture.min_free_gb
            if free < s.capture.min_free_gb / 2 and self.active:
                log.error("disk almost full (%.1f GB free): stopping all recordings", free)
                for cid in list(self.active):
                    self.stop_recording(cid)
                await notify(f"🛑 Disk almost full ({free:.1f} GB free): recordings stopped")
            if low and not disk_warned:
                log.warning("low disk space (%.1f GB free): not starting new recordings", free)
                await notify(f"⚠️ Low disk space ({free:.1f} GB free): new recordings paused")
            disk_warned = low
            if not self.paused and not low:
                with session() as db:
                    chans = db.exec(select(Channel).where(Channel.enabled == True,  # noqa: E712
                                                          Channel.platform != "file")).all()
                for ch in chans:
                    if ch.id in self.active:
                        continue
                    if ch.permission not in ("own", "allowed"):
                        self.last_error[ch.id] = "skipped: no clipping permission set"
                        continue
                    st = await check_live(ch.platform, ch.url)
                    self.last_error[ch.id] = st.error
                    if st.live:
                        log.info("%s is LIVE (%s): %s", ch.name, st.backend, st.title)
                        self.start_recording(ch, st.title, st.backend)
                        await notify(f"🔴 {ch.name} is live, recording: {st.title}")
            await asyncio.sleep(s.monitor.poll_seconds)

    async def add_link(self, url: str, permission: str, mode: str, outputs: str, name: str = "",
                       watch: bool = True) -> tuple[Channel, str]:
        """Paste-a-link entry point: any live URL (or local file). Starts recording now if it is live."""
        url = normalize_url(url)
        platform = detect_platform(url)
        if permission not in ("own", "allowed"):
            raise ValueError("Choose 'my own stream' or 'creator allowed it' before clipping.")
        if mode not in ("best_moments", "continuous", "hybrid") or outputs not in ("shorts", "long", "both"):
            raise ValueError("Unknown mode or output.")
        name = name[:80]
        with session() as db:
            ch = db.exec(select(Channel).where(Channel.url == url)).first()
            if not ch:
                ch = Channel(name=name or guess_name(url, platform), url=url)
            ch.platform, ch.permission, ch.mode, ch.outputs = platform, permission, mode, outputs
            ch.enabled = watch and platform != "file"
            if name:
                ch.name = name
            db.add(ch)
            db.commit()
        if ch.id in self.active:
            return ch, f"{ch.name} is already being recorded."
        st = await check_live(platform, url)
        self.last_error[ch.id] = st.error
        if st.live:
            self.start_recording(ch, st.title or ch.name, st.backend)
            how = {"file": "processing file", "direct": "recording stream URL"}.get(st.backend, f"recording via {st.backend}")
            return ch, f"✅ {ch.name} is live — {how}. Clips will appear here as moments are found."
        if platform == "file":
            return ch, f"File not found: {url}"
        if st.vod:
            # a recorded video / past broadcast: process it once through the same pipeline
            with session() as db:
                c = db.get(Channel, ch.id)
                c.enabled = False
                db.add(c)
                db.commit()
            self.start_recording(ch, st.title or ch.name, "vod")
            return ch, f"🎞 {ch.name}: not live, but it's a recorded video — downloading and clipping it now."
        return ch, (f"⏳ {ch.name} is not live right now ({st.error or 'offline'}). "
                    f"{'It is being watched and recording starts automatically when it goes live.' if watch else ''}")

    def process_file(self, ch: Channel) -> Stream:
        """Offline mode: run a local video through the same pipeline."""
        return self.start_recording(ch, Path(ch.url).stem)

    def clip_now(self, stream_id: int, offset: float | None = None) -> None:
        with session() as db:
            st = db.get(Stream, stream_id)
            t = offset if offset is not None else st.duration
            st.manual_requests = ",".join(filter(None, [st.manual_requests, f"{t:.1f}"]))
            db.add(st)
            db.commit()

    # ------------------------------------------------------------------ processing
    async def _segment_loop(self) -> None:
        while True:
            with session() as db:
                row = db.exec(select(Segment, Channel.language)
                              .join(Stream, Stream.id == Segment.stream_id)
                              .join(Channel, Channel.id == Stream.channel_id)
                              .where(Segment.processed == False, Segment.deleted == False)  # noqa: E712
                              # live streams first: a VOD backlog must never delay live clips
                              .order_by(case((Stream.status == "live", 0), else_=1),
                                        Segment.stream_id, Segment.idx)).first()
            if row:
                await engine.process_segment(row[0], row[1])
            else:
                await asyncio.sleep(3)

    async def _detect_loop(self) -> None:
        await self._detect(["ended"], get_settings().detect.interval_seconds)

    async def _live_detect_loop(self) -> None:
        """Separate, faster loop so live clips appear continuously and a long recorded video being
        judged never delays them."""
        await self._detect(["live"], get_settings().detect.live_interval_seconds)

    async def _detect(self, statuses: list[str], interval: int) -> None:
        s = get_settings()
        while True:
            with session() as db:
                rows = db.exec(select(Stream, Channel).join(Channel, Channel.id == Stream.channel_id)
                               .where(Stream.status.in_(statuses)).order_by(Stream.id)).all()
            for st, ch in rows:
                try:
                    await engine.detect_step(st, ch)
                except Exception:
                    log.exception("detection failed for stream %s", st.id)
            await asyncio.sleep(interval if rows else 10)

    async def _render_loop(self) -> None:
        while True:
            if self.paused or not await worker.render_one():
                await asyncio.sleep(5)

    async def _publish_loop(self) -> None:
        while True:
            uploader.mark_published()
            while not self.paused and await uploader.publish_one():
                pass
            await asyncio.sleep(60)

    async def _analytics_loop(self) -> None:
        s = get_settings()
        await asyncio.sleep(120)
        while True:
            if s.analytics.enabled:
                try:
                    await analytics.refresh()
                except Exception as e:
                    log.warning("analytics: %s", e)
            await asyncio.sleep(s.analytics.interval_hours * 3600)

    # ------------------------------------------------------------------ retention
    async def _retention_loop(self) -> None:
        while True:
            try:
                await asyncio.to_thread(cleanup)
            except Exception:
                log.exception("cleanup failed")
            await asyncio.sleep(600)


def cleanup() -> None:
    s = get_settings()
    r = s.retention
    now = utcnow()
    freed = 0
    with session() as db:
        # Raw segments
        for st in db.exec(select(Stream)).all():
            busy = db.exec(select(Clip).where(Clip.stream_id == st.id,
                                              Clip.status.in_(["pending", "rendering"]))).first()
            if busy:
                continue
            if st.status == "finalized" and st.ended_at and now - st.ended_at > timedelta(hours=r.delete_raw_after_hours):
                cutoff = float("inf")
            elif st.status == "live":
                cutoff = st.duration - s.capture.buffer_hours * 3600
                # never drop anything detection or continuous cutting still needs
                ch = db.get(Channel, st.channel_id)
                limits = [cutoff, st.detected_until - 120]
                if ch and ch.mode in ("continuous", "hybrid"):
                    limits += [st.continuous_until if ch.outputs != "shorts" or ch.mode == "hybrid" else cutoff,
                               st.continuous_short_until if ch.mode == "continuous" and ch.outputs != "long" else cutoff]
                cutoff = min(limits)
            else:
                continue
            for seg in db.exec(select(Segment).where(Segment.stream_id == st.id, Segment.deleted == False,  # noqa: E712
                                                     Segment.processed == True, Segment.end < cutoff)).all():
                p = Path(seg.path)
                if p.exists():
                    freed += p.stat().st_size
                    p.unlink()
                seg.deleted = True
                db.add(seg)
        # Renders of published clips
        old = now - timedelta(days=r.delete_renders_after_days)
        done = select(Clip).where(Clip.path != "")
        if r.delete_renders_after_upload:   # file is on YouTube now; no local copy needed
            done = done.where(Clip.status.in_(["scheduled", "published"]), Clip.youtube_id != "")
        else:
            done = done.where(Clip.status == "published", Clip.publish_at < old)
        for c in db.exec(done).all():
            for f in [c.path, c.thumb_path, c.srt_path]:
                if f and Path(f).exists():
                    freed += Path(f).stat().st_size
                    Path(f).unlink()
            c.path = ""
            db.add(c)
        db.commit()
    if freed:
        log.info("cleanup freed %.1f MB", freed / 1e6)


def disk_free_gb() -> float:
    return shutil.disk_usage(get_settings().data_dir).free / 1e9
