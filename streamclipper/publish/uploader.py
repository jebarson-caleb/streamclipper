"""Upload ready clips to YouTube as private + publishAt (scheduled release)."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import select

from ..config import get_settings
from ..db import session
from ..models import Channel, Clip
from ..notify import notify
from . import scheduler
from .youtube_auth import youtube_client

log = logging.getLogger(__name__)
MAX_ATTEMPTS = 3
CAPTION_COST, THUMB_COST = 400, 50


def _next_ready() -> tuple[Clip, Channel] | None:
    s = get_settings()
    want = ["approved"] if s.review_mode else ["ready", "approved"]
    with session() as db:
        row = db.exec(select(Clip, Channel).join(Channel, Channel.id == Clip.channel_id)
                      .where(Clip.status.in_(want)).order_by(Clip.score.desc(), Clip.id)).first()
    return row


def _upload_sync(yt, clip: Clip, publish_at: datetime) -> str:
    from googleapiclient.http import MediaFileUpload

    body = {
        "snippet": {
            "title": clip.title[:100],
            "description": clip.description[:4900],
            "tags": [t for t in clip.tags.split(",") if t][:40],
            "categoryId": get_settings().publish.category_id,
        },
        "status": {
            "privacyStatus": "private",
            "publishAt": publish_at.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(clip.path, mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        _status, resp = req.next_chunk(num_retries=5)
    return resp["id"]


def _extras_sync(yt, clip: Clip, video_id: str, captions: bool) -> int:
    """Thumbnail (long videos) + caption tracks. Returns quota units spent."""
    from googleapiclient.http import MediaFileUpload

    spent = 0
    if clip.format == "long" and clip.thumb_path and Path(clip.thumb_path).exists():
        try:
            yt.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(clip.thumb_path)).execute()
            spent += THUMB_COST
        except Exception as e:  # needs a verified channel; not fatal
            log.warning("thumbnail upload failed: %s", e)
    if captions and clip.srt_path:
        base = Path(clip.srt_path)
        for srt in [base, *base.parent.glob(base.stem + ".*.srt")]:
            if not srt.exists():
                continue
            parts = srt.name.split(".")
            lang = parts[-2] if len(parts) > 2 else "en"
            try:
                yt.captions().insert(
                    part="snippet",
                    body={"snippet": {"videoId": video_id, "language": lang, "name": lang, "isDraft": False}},
                    media_body=MediaFileUpload(str(srt), mimetype="application/octet-stream"),
                ).execute()
                spent += CAPTION_COST
            except Exception as e:
                log.warning("caption upload (%s) failed: %s", lang, e)
    return spent


async def publish_one() -> bool:
    s = get_settings()
    if not s.publish.enabled:
        return False
    row = _next_ready()
    if not row:
        return False
    clip, ch = row
    account = ch.target_account or "default"
    if not scheduler.can_upload(account):
        return False
    yt = await asyncio.to_thread(youtube_client, account)
    if yt is None:
        log.debug("YouTube account '%s' not connected; clip %s waits", account, clip.id)
        return False
    publish_at = scheduler.pick_publish_time(clip.format, account)
    log.info("uploading clip %s '%s' -> release %s UTC", clip.id, clip.title, publish_at)
    try:
        vid = await asyncio.to_thread(_upload_sync, yt, clip, publish_at)
    except Exception as e:
        msg = str(e)
        if "quotaExceeded" in msg or "uploadLimitExceeded" in msg:
            log.warning("YouTube quota/upload limit hit; pausing uploads until reset")
            scheduler.spend_quota(account, s.publish.daily_quota)  # block the rest of the day
            return False
        with session() as db:
            c = db.get(Clip, clip.id)
            c.attempts += 1
            c.error = msg[:1000]
            if c.attempts >= MAX_ATTEMPTS:
                c.status = "failed"
            db.add(c)
            db.commit()
        log.error("upload failed for clip %s: %s", clip.id, msg[:300])
        await notify(f"⚠️ Upload failed: {clip.title} ({msg[:150]})")
        return False
    scheduler.spend_quota(account, s.publish.upload_cost)
    extra = await asyncio.to_thread(_extras_sync, yt, clip, vid, s.publish.upload_captions)
    if extra:
        scheduler.spend_quota(account, extra)
    with session() as db:
        c = db.get(Clip, clip.id)
        c.youtube_id = vid
        c.publish_at = publish_at.replace(tzinfo=None)
        c.status = "scheduled"
        c.error = ""
        db.add(c)
        db.commit()
    local = publish_at.astimezone().strftime("%a %d %b %H:%M")
    await notify(f"📤 Scheduled: {clip.title} → https://youtu.be/{vid} (goes public {local})")
    return True


def mark_published() -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with session() as db:
        for c in db.exec(select(Clip).where(Clip.status == "scheduled", Clip.publish_at <= now)).all():
            c.status = "published"
            db.add(c)
        db.commit()
