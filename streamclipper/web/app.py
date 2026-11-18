"""Local dashboard (FastAPI + Jinja + HTMX)."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import func, select

from ..config import get_settings
from ..db import session
from ..detect.engine import load_hype
from ..detect.llm import is_up as llm_up
from ..gpu import vram_info
from ..logs import RING
from ..models import Channel, Clip, Moment, Stream
from ..monitor.platforms import detect_platform, normalize_url
from ..publish import scheduler
from ..publish.youtube_auth import authorize, connected_accounts
from ..supervisor import Supervisor, disk_free_gb

log = logging.getLogger(__name__)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
sup = Supervisor()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await sup.start()
    yield
    await sup.stop()


app = FastAPI(title="StreamClipper", lifespan=lifespan)


def _fmt_dt(d: datetime | None) -> str:
    if not d:
        return ""
    return d.replace(tzinfo=timezone.utc).astimezone().strftime("%a %d %b %H:%M")


templates.env.filters["dt"] = _fmt_dt
templates.env.filters["hms"] = lambda s: f"{int(s) // 3600}:{int(s) % 3600 // 60:02d}:{int(s) % 60:02d}"


def hype_svg(stream: Stream, width: int = 600, height: int = 70) -> str:
    try:
        h = load_hype(stream.id, stream.duration)
    except Exception:
        return ""
    if len(h) < 2:
        return ""
    step = max(1, len(h) // width)
    pts = [max(h[i:i + step]) for i in range(0, len(h), step)]
    lo, hi = min(min(pts), 0), max(max(pts), 3)
    coords = " ".join(f"{i * width / max(1, len(pts) - 1):.1f},{height - (v - lo) / (hi - lo) * height:.1f}"
                      for i, v in enumerate(pts))
    thr = height - (get_settings().detect.peak_z - lo) / (hi - lo) * height
    return (f'<svg viewBox="0 0 {width} {height}" class="w-full h-16" preserveAspectRatio="none" role="img" '
            f'aria-label="Hype curve for this stream">'
            f'<line x1="0" x2="{width}" y1="{thr:.1f}" y2="{thr:.1f}" stroke="#ff77cd" stroke-dasharray="4 4"/>'
            f'<polyline fill="none" stroke="#f4f4f4" stroke-width="1.5" points="{coords}"/></svg>')


# --------------------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, msg: str = ""):
    return templates.TemplateResponse(request, "home.html", {"page": "home", "msg": msg})


@app.get("/partials/status", response_class=HTMLResponse)
async def status_partial(request: Request):
    s = get_settings()
    with session() as db:
        live = db.exec(select(Stream, Channel).join(Channel, Channel.id == Stream.channel_id)
                       .where(Stream.status.in_(["live", "ended"]))).all()
        counts = dict(db.exec(select(Clip.status, func.count()).group_by(Clip.status)).all())
        recent = db.exec(select(Clip).order_by(Clip.id.desc()).limit(8)).all()
    accounts = connected_accounts()
    quota = {a: scheduler.quota_used(a) for a in accounts}
    return templates.TemplateResponse(request, "_status.html", {
        "live": [(st, ch, hype_svg(st)) for st, ch in live], "counts": counts, "recent": recent,
        "gpu": vram_info(), "disk": disk_free_gb(), "llm": await llm_up(), "quota": quota,
        "daily_quota": s.publish.daily_quota, "paused": sup.paused, "review": s.review_mode,
        "recording": len(sup.active),
    })


@app.post("/pause")
async def toggle_pause():
    sup.paused = not sup.paused
    return RedirectResponse("/", 303)


# --------------------------------------------------------------------------- channels
@app.get("/channels", response_class=HTMLResponse)
async def channels(request: Request, edit: int | None = None):
    with session() as db:
        chans = db.exec(select(Channel).order_by(Channel.id)).all()
        editing = db.get(Channel, edit) if edit else None
    return templates.TemplateResponse(request, "channels.html", {
        "page": "channels", "chans": chans, "editing": editing, "errors": sup.last_error,
        "active": set(sup.active), "accounts": connected_accounts() or ["default"]})


CHANNEL_FIELDS = ["name", "platform", "url", "permission", "permission_note", "mode", "outputs", "threshold",
                  "max_clips_per_stream", "continuous_len_s", "continuous_short_len_s", "language",
                  "translate_to", "credit_text", "target_account", "title_template", "desc_template",
                  "extra_tags"]


@app.post("/channels/save")
async def save_channel(request: Request):
    form = await request.form()
    cid = form.get("id")
    with session() as db:
        ch = db.get(Channel, int(cid)) if cid else Channel(name="", url="")
        for f in CHANNEL_FIELDS:
            if f in form:
                cur = getattr(ch, f)
                val = form[f]
                setattr(ch, f, type(cur)(val) if isinstance(cur, (int, float)) and val != "" else val)
        ch.url = normalize_url(ch.url)
        if ch.platform in ("", "auto"):
            ch.platform = detect_platform(ch.url)
        ch.enabled = form.get("enabled") == "on"
        if ch.permission not in ("own", "allowed"):
            ch.enabled = False
        db.add(ch)
        db.commit()
    return RedirectResponse("/channels", 303)


@app.post("/channels/{cid}/delete")
async def delete_channel(cid: int):
    sup.stop_recording(cid)
    with session() as db:
        ch = db.get(Channel, cid)
        if ch:
            ch.enabled = False   # keep history; just disable
            db.add(ch)
            db.commit()
    return RedirectResponse("/channels", 303)


@app.post("/channels/{cid}/start")
async def start_channel(cid: int):
    with session() as db:
        ch = db.get(Channel, cid)
    if not ch or ch.permission not in ("own", "allowed"):
        raise HTTPException(400, "Set clipping permission (own/allowed) first")
    from urllib.parse import quote

    msg = ""
    if cid not in sup.active:
        _ch, msg = await sup.add_link(ch.url, ch.permission, ch.mode, ch.outputs, ch.name, ch.enabled)
    return RedirectResponse(f"/?msg={quote(msg)}", 303)


@app.post("/channels/{cid}/stop")
async def stop_channel(cid: int):
    sup.stop_recording(cid)
    return RedirectResponse("/", 303)


@app.post("/quick")
async def quick_add(url: str = Form(...), permission: str = Form(""), mode: str = Form("best_moments"),
                    outputs: str = Form("shorts"), name: str = Form(""), watch: str = Form("")):
    from urllib.parse import quote

    try:
        _ch, msg = await sup.add_link(url, permission, mode, outputs, name.strip(), watch == "on")
    except ValueError as e:
        msg = f"⚠️ {e}"
    return RedirectResponse(f"/?msg={quote(msg)}", 303)


@app.post("/streams/{sid}/clip_now")
async def clip_now(sid: int):
    sup.clip_now(sid)
    return RedirectResponse("/", 303)


# --------------------------------------------------------------------------- moments / clips
@app.get("/moments", response_class=HTMLResponse)
async def moments(request: Request, stream: int | None = None):
    with session() as db:
        q = select(Moment).order_by(Moment.id.desc()).limit(200)
        if stream:
            q = select(Moment).where(Moment.stream_id == stream).order_by(Moment.start)
        ms = db.exec(q).all()
    return templates.TemplateResponse(request, "moments.html", {"page": "moments", "moments": ms})


@app.post("/moments/{mid}/force")
async def force_moment(mid: int, fmt: str = Form("short")):
    with session() as db:
        m = db.get(Moment, mid)
        st = db.get(Stream, m.stream_id)
        m.status = "accepted"
        db.add(m)
        db.add(Clip(stream_id=st.id, channel_id=st.channel_id, moment_ids=str(m.id), format=fmt,
                    title=m.title, category=m.category, score=m.score))
        db.commit()
    return RedirectResponse("/clips", 303)


@app.get("/clips", response_class=HTMLResponse)
async def clips(request: Request, status: str = ""):
    with session() as db:
        q = select(Clip).order_by(Clip.id.desc()).limit(200)
        if status:
            q = select(Clip).where(Clip.status == status).order_by(Clip.id.desc())
        cs = db.exec(q).all()
    return templates.TemplateResponse(request, "clips.html", {"page": "clips", "clips": cs, "status": status,
                                                              "review": get_settings().review_mode})


@app.post("/clips/{cid}/{action}")
async def clip_action(cid: int, action: str, title: str = Form(""), description: str = Form("")):
    with session() as db:
        c = db.get(Clip, cid)
        if not c:
            raise HTTPException(404)
        if action == "approve" and c.status == "ready":
            c.status = "approved"
        elif action == "reject" and c.status not in ("scheduled", "published"):
            c.status = "rejected"
        elif action == "retry" and c.status in ("failed", "rejected"):
            c.status, c.attempts, c.error = ("ready" if c.path and Path(c.path).exists() else "pending"), 0, ""
        elif action == "rerender" and c.status not in ("scheduled", "published"):
            c.status, c.attempts = "pending", 0
        elif action == "edit" and c.status not in ("scheduled", "published"):
            c.title = title[:100] or c.title
            c.description = description[:4900] or c.description
        db.add(c)
        db.commit()
    return RedirectResponse(f"/clips#clip{cid}", 303)


@app.get("/media/clip/{cid}")
async def media(cid: int, thumb: bool = False):
    with session() as db:
        c = db.get(Clip, cid)
    p = c and (c.thumb_path if thumb else c.path)
    if not p or not Path(p).exists():
        raise HTTPException(404)
    return FileResponse(p)


# --------------------------------------------------------------------------- settings / logs
@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, msg: str = ""):
    s = get_settings()
    return templates.TemplateResponse(request, "settings.html", {
        "page": "settings", "s": s, "accounts": connected_accounts(), "msg": msg,
        "secret_ok": Path(s.publish.client_secret).exists(),
        "quota": {a: scheduler.quota_used(a) for a in connected_accounts()}})


@app.post("/youtube/connect")
async def youtube_connect(account: str = Form("default")):
    try:
        await asyncio.to_thread(authorize, account.strip() or "default")
        msg = f"Connected account '{account}'"
    except Exception as e:
        msg = f"Connect failed: {e}"
    return RedirectResponse(f"/settings?msg={msg}", 303)


@app.post("/settings/review")
async def toggle_review():
    s = get_settings()
    s.review_mode = not s.review_mode
    return RedirectResponse("/settings", 303)


@app.get("/logs", response_class=HTMLResponse)
async def logs(request: Request):
    return templates.TemplateResponse(request, "logs.html", {"page": "logs"})


@app.get("/partials/logs", response_class=HTMLResponse)
async def logs_partial():
    from html import escape

    return "<pre class='text-xs whitespace-pre-wrap'>" + escape("\n".join(list(RING)[-300:][::-1])) + "</pre>"
