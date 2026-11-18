"""Live chat readers. Twitch: anonymous IRC. YouTube: chat-downloader.

Messages are stored as ChatMsg rows with an offset relative to stream start.
Mods/broadcaster typing "!clip" request a manual clip.
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
from datetime import datetime

from ..db import session
from ..models import ChatMsg, Stream, utcnow

log = logging.getLogger(__name__)

_TWITCH_LOGIN = re.compile(r"twitch\.tv/([A-Za-z0-9_]+)")


def keyword_hits(text: str, keywords: list[str]) -> int:
    up = text.upper()
    hits = 0
    for kw in keywords:
        k = kw.upper()
        if k.isalnum():
            hits += len(re.findall(rf"(?<![A-Z0-9]){re.escape(k)}(?![A-Z0-9])", up))
        else:
            hits += up.count(k)
    return hits


def parse_ytdlp_chat_line(line: bytes | str) -> list[tuple[str, str, bool, float | None]]:
    """(author, text, is_mod, unix time sent) for each text message in one line of yt-dlp's live_chat JSON."""
    import json

    try:
        d = json.loads(line)
    except ValueError:
        return []
    out = []
    for a in (d.get("replayChatItemAction") or {}).get("actions", []):
        r = ((a.get("addChatItemAction") or {}).get("item") or {}).get("liveChatTextMessageRenderer")
        if not r:
            continue
        text = "".join(run.get("text") or ((run.get("emoji") or {}).get("shortcuts") or [""])[0]
                       for run in (r.get("message") or {}).get("runs", []))
        badges = " ".join(((b.get("liveChatAuthorBadgeRenderer") or {}).get("tooltip") or "")
                          for b in r.get("authorBadges", [])).lower()
        ts = r.get("timestampUsec")
        out.append(((r.get("authorName") or {}).get("simpleText", ""), text,
                    "moderator" in badges or "owner" in badges, int(ts) / 1e6 if ts else None))
    return out


class ChatReader:
    def __init__(self, stream: Stream, platform: str, url: str):
        self.stream = stream
        self.platform = platform
        self.url = url
        self.started: datetime = stream.started_at
        self.buffer: list[ChatMsg] = []
        self._stop = asyncio.Event()

    def _offset(self) -> float:
        return (utcnow() - self.started).total_seconds()

    def _add(self, user: str, text: str, is_mod: bool, sent: float | None = None) -> None:
        # yt-dlp delivers YouTube chat in batches: use each message's own send time when we have it
        t = (datetime.utcfromtimestamp(sent) - self.started).total_seconds() if sent else self._offset()
        self.buffer.append(ChatMsg(stream_id=self.stream.id, t=t, user=user, text=text[:500], is_mod=is_mod))
        if is_mod and text.strip().lower().startswith("!clip"):
            with session() as db:
                st = db.get(Stream, self.stream.id)
                st.manual_requests = ",".join(filter(None, [st.manual_requests, f"{t:.1f}"]))
                db.add(st)
                db.commit()
            log.info("!clip requested by %s at %.0fs", user, t)

    async def _flush_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(5)
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        batch, self.buffer = self.buffer, []
        with session() as db:
            db.add_all(batch)
            db.commit()

    # ------------------------------------------------------------------ twitch
    async def _twitch(self) -> None:
        m = _TWITCH_LOGIN.search(self.url)
        if not m:
            return
        chan = m.group(1).lower()
        while not self._stop.is_set():
            try:
                reader, writer = await asyncio.open_connection("irc.chat.twitch.tv", 6667)
                nick = f"justinfan{random.randint(10000, 99999)}"
                writer.write(b"CAP REQ :twitch.tv/tags\r\n")
                writer.write(f"NICK {nick}\r\nJOIN #{chan}\r\n".encode())
                await writer.drain()
                while not self._stop.is_set():
                    line = await asyncio.wait_for(reader.readline(), timeout=360)
                    if not line:
                        break
                    s = line.decode("utf-8", "replace").rstrip("\r\n")
                    if s.startswith("PING"):
                        writer.write(b"PONG :tmi.twitch.tv\r\n")
                        await writer.drain()
                        continue
                    if " PRIVMSG " not in s:
                        continue
                    tags = {}
                    if s.startswith("@"):
                        raw, s = s[1:].split(" ", 1)
                        tags = dict(kv.split("=", 1) for kv in raw.split(";") if "=" in kv)
                    user = s[1:s.find("!")] if s.startswith(":") else ""
                    text = s.split(" :", 1)[1] if " :" in s else ""
                    badges = tags.get("badges", "")
                    is_mod = tags.get("mod") == "1" or "broadcaster" in badges
                    self._add(tags.get("display-name", user), text, is_mod)
                writer.close()
            except (OSError, asyncio.TimeoutError) as e:
                log.debug("twitch chat reconnect: %s", e)
            await asyncio.sleep(5)

    # ------------------------------------------------------------------ youtube
    def _youtube_sync(self) -> None:
        try:
            from chat_downloader import ChatDownloader
        except ImportError:
            log.warning("chat-downloader not installed; YouTube chat disabled")
            return
        try:
            chat = ChatDownloader().get_chat(self.url)
            for msg in chat:
                if self._stop.is_set():
                    break
                author = msg.get("author", {}) or {}
                badges = " ".join(b.get("title", "") for b in author.get("badges", []) or []).lower()
                is_mod = "moderator" in badges or "owner" in badges
                self._add(author.get("name", ""), msg.get("message", "") or "", is_mod)
        except Exception as e:
            log.warning("youtube chat stopped (%s); following chat with yt-dlp instead", e)
            self._youtube_ytdlp()

    def _youtube_ytdlp(self) -> None:
        """yt-dlp writes a live stream's chat as it arrives, one JSON action per line, to <name>.live_chat.json.part.
        Follow that file."""
        import subprocess
        import sys
        import time
        from pathlib import Path

        out = Path(self.stream.dir) / "chat"
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        proc = subprocess.Popen([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--skip-download",
                                 "--write-subs", "--sub-langs", "live_chat", "-o", str(out), "--", self.url],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                creationflags=flags)
        path = out.with_name(out.name + ".live_chat.json.part")
        pos, buf = 0, b""
        try:
            while not self._stop.is_set():
                time.sleep(1)
                if path.exists():
                    with open(path, "rb") as f:
                        f.seek(pos)
                        buf += f.read()
                        pos = f.tell()
                    *lines, buf = buf.split(b"\n")
                    for line in lines:
                        for user, text, is_mod, sent in parse_ytdlp_chat_line(line):
                            self._add(user, text, is_mod, sent)
                if proc.poll() is not None:
                    log.info("youtube chat (yt-dlp) ended")
                    break
        finally:
            if proc.poll() is None:
                proc.kill()

    async def run(self) -> None:
        flusher = asyncio.create_task(self._flush_loop())
        try:
            if self.platform == "twitch":
                await self._twitch()
            elif self.platform == "youtube":
                await asyncio.to_thread(self._youtube_sync)
        finally:
            flusher.cancel()
            self.flush()

    def stop(self) -> None:
        self._stop.set()
