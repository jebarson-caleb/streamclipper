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


def normalize_url(url: str) -> str:
    url = url.strip().strip('"').strip("'")
    if not url:
        return url
    if Path(url).exists():
        return str(Path(url).resolve())
    if "://" not in url:
        url = "https://" + url
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
