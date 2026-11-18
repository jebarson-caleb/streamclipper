"""Record a live stream (or a local file) into fixed-length .ts segments.

streamlink -> stdout -> ffmpeg segment muxer. ffmpeg appends one CSV line per finished
segment (file,start,end); we tail that list and register Segment rows.
"""
from __future__ import annotations

import asyncio
import csv
import logging
import subprocess
import sys
from pathlib import Path

from sqlmodel import select

from ..config import get_settings
from ..db import session
from ..ffmpeg_utils import FFMPEG
from ..models import Channel, Segment, Stream, utcnow
from ..monitor.live_check import ca_args, check_live
from .twitch_ads import TwitchAdWatcher

log = logging.getLogger(__name__)


def _separate_hls(url: str, fmt: str) -> tuple[str, str] | None:
    """(video m3u8, audio m3u8) when the chosen live format is separate video and audio HLS playlists."""
    try:
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "format": fmt, "noplaylist": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        log.debug("format lookup failed: %s", e)
        return None
    req = info.get("requested_formats") or []
    if len(req) == 2 and all(str(f.get("protocol", "")).startswith("m3u8") for f in req):
        v, a = (req[0], req[1]) if req[0].get("vcodec") not in (None, "none") else (req[1], req[0])
        return v["url"], a["url"]
    return None


class Recorder:
    def __init__(self, channel: Channel, stream: Stream, backend: str = ""):
        self.channel = channel
        self.stream = stream
        if channel.platform == "file":
            backend = "file"
        elif channel.platform == "direct":
            backend = "direct"
        self.backend = backend or "streamlink"
        self.dir = Path(stream.dir)
        self.stop_event = asyncio.Event()
        self.procs: list[subprocess.Popen] = []
        self._vod_file: Path | None = None

    # ------------------------------------------------------------------ commands
    def _ffmpeg_cmd(self, run_no: int, src: str = "pipe:0") -> list[str]:
        seg = get_settings().capture.segment_seconds
        return [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-i", src, "-map", "0:v:0?", "-map", "0:a:0?", "-c", "copy",
            # Matroska segments: hold any codec stream-copied (H.264/HEVC/VP9/AV1/Opus); MPEG-TS can't
            "-f", "segment", "-segment_format", "matroska", "-segment_time", str(seg),
            "-segment_list", str(self.dir / f"list_{run_no}.csv"), "-segment_list_type", "csv",
            "-reset_timestamps", "1",
            str(self.dir / f"seg_{run_no:02d}_%05d.mkv"),
        ]

    def _source_cmd(self) -> list[str]:
        """Command that writes the live stream to stdout (streamlink or yt-dlp)."""
        url = self.channel.url
        if self.backend == "ytdlp":
            pair = _separate_hls(url, self._LIVE_FMT)
            if pair:
                # YouTube: video and audio are separate playlists; pair them segment by segment so they
                # can't start a segment apart (see capture/hls_pair.py)
                return [sys.executable, "-m", "streamclipper.capture.hls_pair", *pair]
            # muxed source (Twitch, most sites): yt-dlp pipes it straight to stdout
            return [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--no-part", "--no-playlist",
                    "-f", self._LIVE_FMT, "--merge-output-format", "mkv",
                    "--hls-use-mpegts", "--ffmpeg-location", FFMPEG, "-o", "-", url]
        return [sys.executable, "-m", "streamlink", "--loglevel", "warning", "--twitch-disable-ads",
                "--hls-live-edge", "3", "--ffmpeg-ffmpeg", FFMPEG, "-O", url, get_settings().capture.quality]

    _FMT = ("bv*[height<=720][vcodec^=avc1]+ba[acodec^=mp4a]/bv*[height<=720][vcodec^=avc1]+ba/"
            "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b")
    # live: 1080p, so a small webcam overlay still has enough pixels for the 9:16 facecam panel
    _LIVE_FMT = ("bv*[height<=1080][vcodec^=avc1]+ba[acodec^=mp4a]/bv*[height<=1080][vcodec^=avc1]+ba/"
                 "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b")

    def _download_vod(self) -> Path:
        """Recorded video: download fully to disk with yt-dlp's normal merge (piping a live merge
        to stdout can corrupt AAC packets, which silently kills audio/transcript for part of a VOD)."""
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        out = self.dir / "source.%(ext)s"
        for attempt in range(3):
            r = subprocess.run(
                [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--no-playlist", "-f", self._FMT,
                 "--merge-output-format", "mp4", "--ffmpeg-location", FFMPEG, "--retries", "10",
                 "-o", str(out), self.channel.url],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, creationflags=flags)
            files = [p for p in self.dir.glob("source.*") if p.suffix not in (".part", ".ytdl")]
            if r.returncode == 0 and files:
                return files[0]
            log.warning("VOD download attempt %d failed: %s", attempt + 1, (r.stderr or "")[-300:])
        raise RuntimeError("could not download the video")

    def _start(self, run_no: int) -> subprocess.Popen:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        if self.backend == "vod":
            self._vod_file = self._download_vod()
        if self.backend in ("file", "direct", "vod"):
            src = str(self._vod_file) if self.backend == "vod" else self.channel.url
            pre = []
            if self.backend == "direct" and src.startswith("http"):
                pre = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10", *ca_args(src)]
            cmd = self._ffmpeg_cmd(run_no, src)
            i = cmd.index("-i")
            cmd[i:i] = pre
            p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, creationflags=flags)
            self.procs = [p]
            return p
        src = subprocess.Popen(self._source_cmd(), stdout=subprocess.PIPE, stdin=subprocess.DEVNULL,
                               creationflags=flags)
        ff = subprocess.Popen(self._ffmpeg_cmd(run_no), stdin=src.stdout, creationflags=flags)
        src.stdout.close()  # ffmpeg owns the pipe now
        self.procs = [src, ff]
        return ff

    def _kill(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    p.kill()

    # ------------------------------------------------------------------ segment tracking
    def _ingest(self, run_no: int, base: float, seen: set[str], first: list[float | None]) -> float:
        """Register newly finished segments. Returns the new end offset."""
        lst = self.dir / f"list_{run_no}.csv"
        end_offset = base
        if not lst.exists():
            return end_offset
        with lst.open(newline="", encoding="utf-8") as f:
            rows = [r for r in csv.reader(f) if len(r) >= 3]
        new = []
        for name, s, e in rows:
            s, e = float(s), float(e)
            if first[0] is None:
                first[0] = s
            start, end = base + s - first[0], base + e - first[0]
            end_offset = max(end_offset, end)
            if name in seen:
                continue
            seen.add(name)
            new.append((name, start, end))
        if new:
            with session() as db:
                n = db.exec(select(Segment).where(Segment.stream_id == self.stream.id)).all()
                idx = len(n)
                for name, start, end in new:
                    db.add(Segment(stream_id=self.stream.id, idx=idx, path=str(self.dir / name),
                                   start=start, end=end))
                    idx += 1
                st = db.get(Stream, self.stream.id)
                st.duration = end_offset
                db.add(st)
                db.commit()
        return end_offset

    # ------------------------------------------------------------------ main loop
    async def run(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        base, run_no, end = 0.0, 0, 0.0
        log.info("recording %s via %s -> %s", self.channel.name, self.backend, self.dir)
        ads = None
        if self.backend == "ytdlp" and self.channel.platform == "twitch":
            ads = asyncio.create_task(TwitchAdWatcher(self.channel.url, self.dir, self.stream.started_at,
                                                      utcnow).run(self.stop_event))
        try:
            empty_runs = 0
            while not self.stop_event.is_set():
                proc = await asyncio.to_thread(self._start, run_no)
                seen: set[str] = set()
                first: list[float | None] = [None]
                end = base
                while proc.poll() is None and not self.stop_event.is_set():
                    await asyncio.sleep(3)
                    end = await asyncio.to_thread(self._ingest, run_no, base, seen, first)
                await asyncio.to_thread(self._kill)
                end = await asyncio.to_thread(self._ingest, run_no, base, seen, first)
                if self.stop_event.is_set() or self.backend in ("file", "vod"):
                    break
                # The pipe died: stream ended, or a network drop. Re-check before giving up.
                empty_runs = empty_runs + 1 if not seen else 0
                if empty_runs >= 3:
                    log.warning("%s: 3 restarts in a row produced no video; treating the stream as ended",
                                self.channel.name)
                    break
                st = await check_live(self.channel.platform, self.channel.url)
                if not st.live:
                    break
                self.backend = st.backend or self.backend
                wait = 3 * 2 ** empty_runs              # 3 s, then 6, 12, 24 s: no hammering a dead source
                log.warning("%s: recorder restarting in %ds (stream still live)", self.channel.name, wait)
                await asyncio.sleep(wait)
                base, run_no = end, run_no + 1
        except Exception as e:
            log.error("%s: capture failed: %s", self.channel.name, e)
        finally:
            if ads:
                ads.cancel()
            await asyncio.to_thread(self._kill)
            vod = getattr(self, "_vod_file", None)
            if vod:   # segments are copies of it; free the disk space
                Path(vod).unlink(missing_ok=True)
            with session() as db:
                st = db.get(Stream, self.stream.id)
                st.status = "ended"
                st.ended_at = utcnow()
                db.add(st)
                db.commit()
            log.info("recording ended: %s (%.0fs)", self.channel.name, end)

    def stop(self) -> None:
        self.stop_event.set()
