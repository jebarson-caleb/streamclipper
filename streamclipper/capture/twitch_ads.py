"""Twitch ad breaks when capturing through yt-dlp.

streamlink filters Twitch's stitched ads out of the stream; yt-dlp records them like content. When yt-dlp
is the capture backend, poll the same HLS media playlist and note when ad segments are playing, in seconds
since stream start (the clock chat uses). Detection skips windows that overlap an ad. Spans go to
<stream dir>/ads.json as [[start, end], ...].
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

ADS_FILE = "ads.json"
PAD = 6.0   # playlist edge vs. recorded position drift, either side


def playlist_has_ad(text: str) -> bool:
    """True if the newest segments of a Twitch media playlist are ads (same markers streamlink uses)."""
    ad_ranges = re.findall(r'#EXT-X-DATERANGE:[^\n]*(?:CLASS="twitch-stitched-ad"|ID="stitched-ad-)', text)
    titles = re.findall(r"#EXTINF:[\d.]+,([^\n]*)", text)
    if titles and "Amazon" in titles[-1]:
        return True
    return bool(ad_ranges) and not any(t.strip() == "live" for t in titles[-2:])


def load_ad_spans(stream_dir: str | Path) -> list[tuple[float, float]]:
    try:
        return [(a, b) for a, b in json.loads((Path(stream_dir) / ADS_FILE).read_text())]
    except (OSError, ValueError):
        return []


def overlaps_ad(spans: list[tuple[float, float]], a: float, b: float) -> bool:
    return any(s - PAD < b and a < e + PAD for s, e in spans)


class TwitchAdWatcher:
    def __init__(self, url: str, stream_dir: Path, started: datetime, now):
        self.url, self.dir, self.started, self.now = url, stream_dir, started, now
        self.spans: list[list[float]] = [list(x) for x in load_ad_spans(stream_dir)]
        self._in_ad = False

    def _playlist_url(self) -> str:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        r = subprocess.run([sys.executable, "-m", "yt_dlp", "--no-warnings", "-g", "-f", "b[height<=720]/b",
                            self.url], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                           creationflags=flags, timeout=60)
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else ""

    def _mark(self, ad: bool) -> None:
        t = (self.now() - self.started).total_seconds()
        if ad and not self._in_ad:
            self.spans.append([t, t])
            log.info("Twitch ad break started at %.0fs: moments there will be skipped", t)
        elif ad:
            self.spans[-1][1] = t
        elif self._in_ad:
            self.spans[-1][1] = t
            log.info("Twitch ad break ended at %.0fs", t)
        if ad or self._in_ad:
            (self.dir / ADS_FILE).write_text(json.dumps(self.spans))
        self._in_ad = ad

    async def run(self, stop: asyncio.Event) -> None:
        import httpx

        media = ""
        async with httpx.AsyncClient(timeout=10) as client:
            while not stop.is_set():
                try:
                    if not media:
                        media = await asyncio.to_thread(self._playlist_url)
                    if media:
                        r = await client.get(media)
                        if r.status_code == 200:
                            self._mark(playlist_has_ad(r.text))
                        else:
                            media = ""   # token expired or stream restarted: fetch a new playlist URL
                except Exception as e:
                    log.debug("ad watch: %s", e)
                    media = ""
                try:
                    await asyncio.wait_for(stop.wait(), 2)
                except asyncio.TimeoutError:
                    pass
