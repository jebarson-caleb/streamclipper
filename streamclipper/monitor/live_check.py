"""Is a link live right now, and which tool can capture it?

Order: local file -> direct stream URL (ffprobe) -> streamlink plugin -> yt-dlp (1800+ sites, incl. generic pages
with an embedded player). The winning backend is remembered by the recorder.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from .platforms import detect_platform

log = logging.getLogger(__name__)


@dataclass
class LiveStatus:
    live: bool
    title: str = ""
    error: str = ""
    backend: str = ""     # file | direct | streamlink | ytdlp
    vod: bool = False     # not live, but a recorded video/past stream yt-dlp can download and process


class _QuietLogger:
    def debug(self, msg):
        pass

    info = warning = debug

    def error(self, msg):
        log.debug("yt-dlp: %s", msg)


def ca_args(url: str) -> list[str]:
    """ffmpeg on Windows has no CA store: point HTTPS inputs at certifi's bundle."""
    if not url.lower().startswith("https"):
        return []
    try:
        import certifi

        return ["-ca_file", certifi.where()]
    except ImportError:
        return []


def _streamlink(url: str) -> LiveStatus | None:
    """None = streamlink has no plugin for this site (try yt-dlp)."""
    try:
        from streamlink import Streamlink
        from streamlink.exceptions import NoPluginError, PluginError
    except ImportError:  # pragma: no cover
        return None
    session = Streamlink()
    try:
        _name, plugin_cls, resolved = session.resolve_url(url)
    except NoPluginError:
        return None
    except Exception as e:  # plugin failed to load (e.g. a blocked native module): let yt-dlp try
        log.warning("streamlink plugin failed to load for %s: %s", url, str(e)[:200])
        return None
    try:
        plugin = plugin_cls(session, resolved)
        streams = plugin.streams()
        title = ""
        try:
            title = plugin.get_title() or ""
        except Exception:
            pass
        return LiveStatus(bool(streams), title=title, backend="streamlink",
                          error="" if streams else "offline")
    except PluginError as e:
        return LiveStatus(False, error=f"streamlink: {str(e)[:200]}", backend="streamlink")
    except Exception as e:
        return LiveStatus(False, error=f"streamlink: {str(e)[:200]}", backend="streamlink")


def _ytdlp(url: str) -> LiveStatus:
    try:
        import yt_dlp
    except ImportError:  # pragma: no cover
        return LiveStatus(False, error="yt-dlp not installed")
    from ..ffmpeg_utils import FFMPEG

    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True,
            "extract_flat": False, "socket_timeout": 20, "logger": _QuietLogger(),
            "ffmpeg_location": FFMPEG}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        msg = str(e).splitlines()[0][:200]
        return LiveStatus(False, error=f"yt-dlp: {msg}", backend="ytdlp")
    if info.get("_type") == "playlist" and info.get("entries"):
        info = next((e for e in info["entries"] if e and e.get("is_live")), info["entries"][0] or info)
    status = info.get("live_status") or ""
    live = bool(info.get("is_live")) or status == "is_live"
    vod = not live and status in ("", "not_live", "was_live", "post_live") and bool(
        info.get("formats") or info.get("url"))
    return LiveStatus(live, title=info.get("title") or "", backend="ytdlp", vod=vod,
                      error="" if live else f"not live ({status or 'video'})")


def hls_ended(playlist_text: str) -> bool:
    """An HLS media playlist with #EXT-X-ENDLIST is finished (a recording), not live."""
    return "#EXT-X-ENDLIST" in playlist_text


async def _direct(url: str) -> LiveStatus:
    from ..ffmpeg_utils import FFPROBE

    if url.split("?")[0].lower().endswith(".m3u8") and url.lower().startswith("http"):
        try:
            import httpx

            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as c:
                r = await c.get(url)
            if r.status_code == 200 and hls_ended(r.text):
                # finished stream / VOD playlist: process it once as a recording
                return LiveStatus(False, title=Path(url.split("?")[0]).stem, backend="direct", vod=True,
                                  error="stream ended (playlist is complete)")
        except Exception as e:
            log.debug("playlist check failed: %s", e)
    try:
        proc = await asyncio.create_subprocess_exec(
            FFPROBE, "-v", "error", "-rw_timeout", "15000000", *ca_args(url), "-show_entries", "format=format_name",
            "-of", "csv=p=0", "-i", url, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await asyncio.wait_for(proc.communicate(), 30)
        if proc.returncode == 0:
            return LiveStatus(True, title=Path(url.split("?")[0]).stem, backend="direct")
        return LiveStatus(False, error=f"unreachable: {err.decode(errors='replace')[:150]}", backend="direct")
    except (asyncio.TimeoutError, OSError) as e:
        return LiveStatus(False, error=f"unreachable: {e}", backend="direct")


def _streamlink_hls_ok() -> bool:
    """streamlink's HLS needs pycryptodome's native modules, which Windows Application Control can block."""
    try:
        import streamlink.stream.hls  # noqa: F401
        return True
    except Exception as e:
        log.warning("streamlink HLS unavailable (%s); capturing with yt-dlp", str(e)[:120])
        return False


def _check_sync(url: str, platform: str = "") -> LiveStatus:
    if platform == "youtube":
        # streamlink's YouTube plugin also "plays" normal uploads, so trust yt-dlp's live_status here;
        # capture still goes through streamlink when it's live (more robust for long HLS sessions).
        yt = _ytdlp(url)
        if yt.live and _streamlink_hls_ok():
            yt.backend = "streamlink"
        return yt
    st = _streamlink(url)
    if st is not None and (st.live or "streamlink:" not in st.error):
        return st              # streamlink knows this site (live, or cleanly offline)
    yt = _ytdlp(url)           # no plugin, or the plugin errored: let yt-dlp try
    if yt.live or st is None:
        return yt
    return st


async def check_live(platform: str, url: str) -> LiveStatus:
    platform = platform or detect_platform(url)
    if platform == "file":
        ok = Path(url).exists()
        return LiveStatus(ok, title=Path(url).stem, error="" if ok else "file not found", backend="file")
    if platform == "direct":
        return await _direct(url)
    return await asyncio.to_thread(_check_sync, url, platform)
