"""Recognise any live link: platform name, capture backend, and a friendly channel name."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

# host fragment -> platform name (only used for labels/chat; capture works for any site)
_HOSTS = {
    "twitch.tv": "twitch", "youtube.com": "youtube", "youtu.be": "youtube", "kick.com": "kick",
    "tiktok.com": "tiktok", "facebook.com": "facebook", "fb.watch": "facebook", "instagram.com": "instagram",
    "x.com": "x", "twitter.com": "x", "rumble.com": "rumble", "trovo.live": "trovo", "dlive.tv": "dlive",
    "bilibili.com": "bilibili", "live.bilibili.com": "bilibili", "sooplive.co.kr": "soop",
    "afreecatv.com": "soop", "vimeo.com": "vimeo", "dailymotion.com": "dailymotion", "huya.com": "huya",
    "douyu.com": "douyu", "nimo.tv": "nimo", "picarto.tv": "picarto", "steamcommunity.com": "steam",
    "pluto.tv": "pluto", "loco.gg": "loco", "rooter.gg": "rooter",
}
_DIRECT_SCHEMES = ("rtmp", "rtmps", "rtsp", "srt", "udp", "rtp")
_DIRECT_EXT = (".m3u8", ".mpd", ".flv", ".ts", ".mp4", ".mkv")


_ALLOWED_SCHEMES = ("http", "https", *_DIRECT_SCHEMES)


def normalize_url(url: str) -> str:
    """A safe source: an existing local video file (absolute path) or a network URL with a known scheme.

    Raises ValueError otherwise. The result is passed to yt-dlp / streamlink / ffmpeg, so it must never look
    like a command-line option (e.g. "--exec=..." or "--config-locations=...")."""
    url = url.strip().strip('"').strip("'")
    if not url:
        raise ValueError("Paste a live link or a video file path.")
    if any(ord(c) < 32 or c == "\x7f" for c in url) or url.startswith("-"):
        raise ValueError("That doesn't look like a link or a file path.")
    p = Path(url)
    if p.exists():
        if not p.is_file():
            raise ValueError("That's a folder; paste the path of a video file.")
        return str(p.resolve())
    if re.match(r"[A-Za-z]:[\\/]", url) or url.startswith(("\\", "/")):   # a path that isn't an existing file
        raise ValueError(f"File not found: {url}")
    if "://" not in url:
        if re.match(r"[A-Za-z][\w+.-]*:(?!\d)", url):   # "concat:", "javascript:", "file:" ... (not host:port)
            raise ValueError("Not a supported link: use an http(s), RTMP, RTSP or SRT address, or a video file path.")
        url = "https://" + url
    parsed = urlparse(url)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES or not parsed.hostname or any(c.isspace() for c in url):
        raise ValueError("Not a supported link: use an http(s), RTMP, RTSP or SRT address, or a video file path.")
    return url


def detect_platform(url: str) -> str:
    """'file', 'direct' (raw HLS/RTMP/... URL), a known platform name, or 'web' (any other site)."""
    if not url:
        return "web"
    if Path(url).exists():
        return "file"
    p = urlparse(url)
    if p.scheme.lower() in _DIRECT_SCHEMES or p.path.lower().endswith(_DIRECT_EXT):
        return "direct"
    host = (p.hostname or "").lower()
    for frag, name in sorted(_HOSTS.items(), key=lambda kv: -len(kv[0])):
        if host == frag or host.endswith("." + frag):
            return name
    return "web"


def guess_name(url: str, platform: str) -> str:
    if platform == "file":
        return Path(url).stem[:40]
    p = urlparse(url)
    parts = [x for x in p.path.split("/") if x and x.lower() not in ("live", "watch", "channel", "c", "user", "videos")]
    if platform == "youtube" and "v=" in (p.query or ""):
        return "youtube-" + p.query.split("v=")[1][:11]
    if parts:
        name = parts[0].lstrip("@")
        return re.sub(r"[^\w.-]", "_", name)[:40]
    return (p.hostname or "stream").replace("www.", "")[:40]


# chat readers exist for these
CHAT_PLATFORMS = {"twitch", "youtube"}
