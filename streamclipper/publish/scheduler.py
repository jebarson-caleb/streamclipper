"""Release slots + YouTube quota bookkeeping. Pure-ish functions so they can be unit tested."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlmodel import select

from ..config import get_settings
from ..db import session
from ..models import Channel, Clip, QuotaLog

PACIFIC = ZoneInfo("America/Los_Angeles")  # YouTube quota resets at midnight Pacific


def quota_day(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.astimezone(PACIFIC).date().isoformat()


def quota_used(account: str, day: str | None = None) -> int:
    day = day or quota_day()
    with session() as db:
        rows = db.exec(select(QuotaLog).where(QuotaLog.day == day, QuotaLog.account == account)).all()
    return sum(r.units for r in rows)


def spend_quota(account: str, units: int) -> None:
    with session() as db:
        db.add(QuotaLog(day=quota_day(), account=account, units=units))
        db.commit()


def can_upload(account: str) -> bool:
    p = get_settings().publish
    return quota_used(account) + p.upload_cost <= p.daily_quota


def next_slot(slots: list[str], taken: list[datetime], tz: ZoneInfo, now: datetime,
              lead: timedelta, max_per_day: int, days_ahead: int = 30) -> datetime:
    """Earliest free slot >= now+lead (all datetimes aware UTC). Respects a per-day cap (local days)."""
    times = sorted(time.fromisoformat(s) for s in slots) or [time(12, 0)]
    taken_set = {t.replace(second=0, microsecond=0) for t in taken}
    per_day: dict[date, int] = {}
    for t in taken:
        d = t.astimezone(tz).date()
        per_day[d] = per_day.get(d, 0) + 1
    start_local = (now + lead).astimezone(tz)
    for d_off in range(days_ahead):
        d = start_local.date() + timedelta(days=d_off)
        if per_day.get(d, 0) >= max_per_day:
            continue
        for tt in times:
            cand = datetime.combine(d, tt, tzinfo=tz)
            if cand < start_local:
                continue
            utc = cand.astimezone(timezone.utc).replace(second=0, microsecond=0)
            if utc not in taken_set:
                return utc
    # Everything full: one minute after the last taken slot.
    return max(taken_set | {now + lead}) + timedelta(minutes=1)


def local_tz():
    """config timezone; 'local' (default) = this PC's timezone."""
    name = get_settings().timezone
    if not name or name.lower() == "local":
        return datetime.now().astimezone().tzinfo
    return ZoneInfo(name)


def pick_publish_time(fmt: str, account: str) -> datetime:
    s = get_settings()
    tz = local_tz()
    now = datetime.now(timezone.utc)
    with session() as db:
        rows = db.exec(select(Clip.publish_at).join(Channel, Channel.id == Clip.channel_id)
                       .where(Clip.publish_at != None,  # noqa: E711
                              Clip.status.in_(["scheduled", "published"]),
                              Channel.target_account == account)).all()
    taken = [r.replace(tzinfo=timezone.utc) for r in rows if r]
    taken = [t for t in taken if t >= now - timedelta(days=1)]
    slots = s.publish.slots.get(fmt) or s.publish.slots.get("short") or ["12:00"]
    return next_slot(slots, taken, tz, now, timedelta(minutes=s.publish.min_lead_minutes),
                     s.publish.max_uploads_per_day)
