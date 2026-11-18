"""Record a live stream served as separate video and audio HLS playlists (YouTube), in sync.

yt-dlp joins each playlist at its own live edge, so audio and video can start a whole segment apart
(measured: audio 5.0 s ahead on a YouTube live) and nothing in the timestamps shows it. Both playlists
number their segments the same way, so take segment N from each, mux the pair, and write MPEG-TS to
stdout for the recorder's segmenter.

    python -m streamclipper.capture.hls_pair VIDEO_M3U8 AUDIO_M3U8
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

LIVE_EDGE = 2      # start this many segments behind the newest one
STALL_S = 120      # no new segment for this long: treat the stream as ended


@dataclass
class Playlist:
    seq: int
    urls: list[str]
    ended: bool


def parse_playlist(text: str) -> Playlist:
    seq, urls = 0, []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
            seq = int(line.split(":", 1)[1])
        elif line and not line.startswith("#"):
            urls.append(line)
    return Playlist(seq, urls, "#EXT-X-ENDLIST" in text)


def common_range(v: Playlist, a: Playlist) -> tuple[int, int]:
    """Sequence numbers present in both playlists, inclusive (lo > hi when there are none)."""
    return max(v.seq, a.seq), min(v.seq + len(v.urls), a.seq + len(a.urls)) - 1


def _video_start(ffprobe: str, path: Path, flags: int) -> float:
    r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=start_time", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL, creationflags=flags)
    try:
        return float(r.stdout.strip().splitlines()[0].split(",")[0])
    except (ValueError, IndexError):
        return 0.0


def main(video_url: str, audio_url: str) -> int:
    import httpx

    from ..ffmpeg_utils import FFMPEG, FFPROBE

    out = sys.stdout.buffer
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    nxt: int | None = None
    last_new = time.monotonic()
    with httpx.Client(timeout=20, follow_redirects=True) as client, tempfile.TemporaryDirectory() as tmp:
        vf, af = Path(tmp) / "v.mp4", Path(tmp) / "a.mp4"
        while True:
            rv, ra = client.get(video_url), client.get(audio_url)
            if rv.status_code != 200 or ra.status_code != 200:
                return 1          # URL expired or stream gone: the recorder re-resolves and restarts
            v, a = parse_playlist(rv.text), parse_playlist(ra.text)
            lo, hi = common_range(v, a)
            if nxt is None:
                nxt = max(lo, hi - LIVE_EDGE)
            nxt = max(nxt, lo)    # fell out of the window: skip ahead rather than stall
            while nxt <= hi:
                vb = client.get(v.urls[nxt - v.seq])
                ab = client.get(a.urls[nxt - a.seq])
                if vb.status_code != 200 or ab.status_code != 200:
                    return 1
                vf.write_bytes(vb.content)
                af.write_bytes(ab.content)
                vstart = _video_start(FFPROBE, vf, flags)
                # Video fragments carry their real media time; YouTube's audio fragments restart at 0 and
                # run 4.96-5.17 s. Put each audio fragment at its video's start, cut where the video ends
                # (-shortest; an input -t would drop the AAC sample rate): the pair stays within one AAC
                # frame and errors can't add up across segments.
                r = subprocess.run([FFMPEG, "-v", "error", "-copyts", "-i", str(vf),
                                    "-itsoffset", f"{vstart:.3f}", "-i", str(af),
                                    "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-shortest",
                                    "-f", "mpegts", "pipe:1"],
                                   stdin=subprocess.DEVNULL, capture_output=True, creationflags=flags)
                if r.returncode == 0:
                    try:
                        out.write(r.stdout)
                        out.flush()
                    except (BrokenPipeError, OSError):
                        return 0  # the recorder stopped reading
                nxt += 1
                last_new = time.monotonic()
            if v.ended and a.ended:
                return 0
            if time.monotonic() - last_new > STALL_S:
                return 1
            time.sleep(1)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
