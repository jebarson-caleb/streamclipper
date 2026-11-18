"""Local dashboard (FastAPI + Jinja + HTMX)."""
from __future__ import annotations

import asyncio
import logging
import math
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import func, select

from .. import firstrun
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


app = FastAPI(title="StreamClipper", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

_LOOPBACK = ("127.0.0.1", "localhost", "[::1]")
_SAFE_METHODS = ("GET", "HEAD", "OPTIONS")


def _allowed_hosts() -> set[str] | None:
    """Host headers the dashboard answers to; None = any (dashboard deliberately exposed on the network)."""
    d = get_settings().dashboard
    if d.host not in ("127.0.0.1", "localhost", "::1"):
        return None
    return {f"{h}:{d.port}" for h in _LOOPBACK}


@app.middleware("http")
async def local_guard(request: Request, call_next):
    """Block other websites from driving the dashboard: DNS rebinding (wrong Host) and cross-site form posts
    (CSRF). Browsers always send Origin or Sec-Fetch-Site on POST; the CLI `add` command sends neither."""
    from fastapi.responses import PlainTextResponse

    hosts = _allowed_hosts()
    host = (request.headers.get("host") or "").lower()
    if hosts is not None and host not in hosts:
        return PlainTextResponse("StreamClipper only answers on localhost.", 403)
    if request.method not in _SAFE_METHODS:
        origin = request.headers.get("origin")
        site = request.headers.get("sec-fetch-site")
        if origin is not None:
            ok = origin.lower() == f"http://{host}"
        else:
            ok = site in (None, "same-origin", "none")
        if not ok:
            log.warning("blocked cross-site %s %s (origin=%s)", request.method, request.url.path, origin)
            return PlainTextResponse("Cross-site request blocked.", 403)
    resp = await call_next(request)
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    # "same-origin", not "no-referrer": with no-referrer Chrome sends Origin: null on our own form posts
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    resp.headers.setdefault("Content-Security-Policy", "frame-ancestors 'none'; form-action 'self'; base-uri 'self'")
    return resp


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
    if not firstrun.is_complete():
        return RedirectResponse("/welcome", 303)
    return templates.TemplateResponse(request, "home.html", {"page": "home", "msg": msg})


# --------------------------------------------------------------------------- first-run guide
def _setup_ctx() -> dict:
    st = firstrun.status()
    return {"st": st, "busy": firstrun.busy(st)}


@app.get("/welcome", response_class=HTMLResponse)
async def welcome(request: Request, msg: str = ""):
    return templates.TemplateResponse(request, "welcome.html", {"page": "welcome", "msg": msg, **_setup_ctx()})


@app.get("/partials/setup", response_class=HTMLResponse)
async def setup_partial(request: Request):
    return templates.TemplateResponse(request, "_setup_steps.html", _setup_ctx())


@app.post("/setup/retry", response_class=HTMLResponse)
async def setup_retry(request: Request):
    firstrun.retry_ai()
    await asyncio.sleep(0.3)
    return templates.TemplateResponse(request, "_setup_steps.html", _setup_ctx())


@app.post("/setup/client-secret")
async def setup_client_secret(request: Request):
    import json
    from urllib.parse import quote

    form = await request.form()
    up = form.get("file")
    data = await up.read(65537) if up is not None and hasattr(up, "read") else b""
    try:
        if len(data) > 65536:
            raise ValueError("file is too large to be a Google client file")
        info = json.loads(data.decode("utf-8"))
        inst = info.get("installed") if isinstance(info, dict) else None
        if not (isinstance(inst, dict) and isinstance(inst.get("client_id"), str)
                and isinstance(inst.get("client_secret"), str)):
            raise ValueError("this is not a Desktop app OAuth client file")
    except (ValueError, UnicodeDecodeError) as e:
        return RedirectResponse(f"/welcome?msg={quote('Could not use that file: ' + str(e))}#youtube", 303)
    Path(get_settings().publish.client_secret).write_bytes(data)
    return RedirectResponse(f"/welcome?msg={quote('Google client file saved. Now click Connect.')}#youtube", 303)


@app.post("/welcome/done")
async def welcome_done():
    firstrun.mark_complete()
    return RedirectResponse("/", 303)


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
async def channels(request: Request, edit: int | None = None, msg: str = ""):
    with session() as db:
        chans = db.exec(select(Channel).order_by(Channel.id)).all()
        editing = db.get(Channel, edit) if edit else None
    return templates.TemplateResponse(request, "channels.html", {
        "page": "channels", "chans": chans, "editing": editing, "errors": sup.last_error, "msg": msg,
        "active": set(sup.active), "accounts": connected_accounts() or ["default"]})


CHANNEL_FIELDS = ["name", "platform", "url", "permission", "permission_note", "mode", "outputs", "threshold",
                  "max_clips_per_stream", "continuous_len_s", "continuous_short_len_s", "language",
                  "translate_to", "credit_text", "target_account", "title_template", "desc_template",
                  "extra_tags"]


MODES = ("best_moments", "continuous", "hybrid")
OUTPUTS = ("shorts", "long", "both")
PERMISSIONS = ("own", "allowed", "none")


@app.post("/channels/save")
async def save_channel(request: Request):
    from urllib.parse import quote

    form = await request.form()
    cid = str(form.get("id") or "")
    with session() as db:
        ch = db.get(Channel, int(cid)) if cid.isdigit() else Channel(name="", url="")
        if ch is None:
            raise HTTPException(404)
        try:
            for f in CHANNEL_FIELDS:
                if f in form:
                    cur = getattr(ch, f)
                    val = str(form[f])
                    if isinstance(cur, (int, float)) and not isinstance(cur, bool):
                        if val == "":
                            continue
                        raw = float(val)
                        if not math.isfinite(raw) or raw < 0:
                            raise ValueError(f"{f.replace('_', ' ')} must be a number of 0 or more")
                        num = int(raw) if isinstance(cur, int) else raw
                        setattr(ch, f, num)
                    else:
                        setattr(ch, f, val[:5000])
            ch.threshold = min(ch.threshold, 10.0)
            if ch.mode not in MODES or ch.outputs not in OUTPUTS or ch.permission not in PERMISSIONS:
                raise ValueError("unknown mode, output or permission")
            ch.url = normalize_url(ch.url)
        except ValueError as e:
            return RedirectResponse(f"/channels?msg={quote('Not saved: ' + str(e))}", 303)
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
        try:
            _ch, msg = await sup.add_link(ch.url, ch.permission, ch.mode, ch.outputs, ch.name, ch.enabled)
        except ValueError as e:
            msg = f"⚠️ {e}"
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
    with session() as db:
        if db.get(Stream, sid) is None:
            raise HTTPException(404)
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
    if fmt not in ("short", "long"):
        raise HTTPException(400, "format must be short or long")
    with session() as db:
        m = db.get(Moment, mid)
        st = db.get(Stream, m.stream_id) if m else None
        if not m or not st:
            raise HTTPException(404)
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
        "page": "settings", "s": s, "accounts": connected_accounts(), "msg": msg, "config_json": redacted_config(s),
        "secret_ok": Path(s.publish.client_secret).exists(),
        "quota": {a: scheduler.quota_used(a) for a in connected_accounts()}})


@app.post("/youtube/connect")
async def youtube_connect(account: str = Form("default"), back: str = Form("/settings")):
    try:
        await asyncio.to_thread(authorize, account.strip() or "default")
        msg = f"Connected account '{account}'"
    except Exception as e:
        msg = f"Connect failed: {e}"
    from urllib.parse import quote

    back = back if back in ("/settings", "/welcome") else "/settings"
    return RedirectResponse(f"{back}?msg={quote(msg)}" + ("#youtube" if back == "/welcome" else ""), 303)


_SECRET_KEYS = ("token", "webhook", "secret", "password", "api_key", "chat_id")


def redacted_config(s) -> str:
    """The settings page shows the active config; never show notification tokens or webhook URLs."""
    import json

    def scrub(v, key=""):
        if isinstance(v, dict):
            return {k: scrub(x, k) for k, x in v.items()}
        if isinstance(v, list):
            return [scrub(x, key) for x in v]
        if v and any(w in key.lower() for w in _SECRET_KEYS) and key != "client_secret":
            return "•••• (set)"
        return v

    return json.dumps(scrub(json.loads(s.model_dump_json())), indent=2, ensure_ascii=False)


@app.post("/settings/review")
async def toggle_review():
    s = get_settings()
    s.review_mode = not s.review_mode
    return RedirectResponse("/settings", 303)


# --------------------------------------------------------------------------- caption style
@app.get("/captions", response_class=HTMLResponse)
async def captions_page(request: Request, msg: str = ""):
    from dataclasses import asdict

    from ..edit import caption_style as cs

    return templates.TemplateResponse(request, "captions.html", {
        "page": "captions", "msg": msg, "st": cs.load(), "fonts": list(cs.FONTS), "positions": cs.POSITIONS,
        "presets": {k: asdict(cs.preset(k)) for k in cs.PRESETS}, "labels": cs.PRESET_LABELS,
        "font_files": {k: v for k, v in cs.FONTS.items() if v}})


@app.post("/captions/save")
async def captions_save(request: Request):
    from urllib.parse import quote

    from ..edit import caption_style as cs

    cs.save(cs.validate(dict(await request.form())))
    return RedirectResponse(f"/captions?msg={quote('Caption style saved. New clips use it.')}", 303)


_preview_lock = asyncio.Lock()


@app.post("/captions/preview", response_class=HTMLResponse)
async def captions_preview(request: Request):
    """Render one real 9:16 frame with libass, exactly as clips will look. One render at a time."""
    import base64
    import tempfile

    from ..edit import caption_style as cs
    from ..edit import subtitles as subs
    from ..ffmpeg_utils import FFmpegError, ass_filter, run

    style = cs.validate(dict(await request.form()))
    words = [subs.Word(i * 0.5, i * 0.5 + 0.45, t) for i, t in enumerate("no way that just happened".split())]
    if _preview_lock.locked():
        return HTMLResponse("<p class='hint'>A preview is already rendering…</p>")
    async with _preview_lock:
        with tempfile.TemporaryDirectory(dir=get_settings().data_dir) as tmp:
            ass, png = Path(tmp) / "preview.ass", Path(tmp) / "preview.png"
            ass.write_text(subs.build_ass(words, 2.5, True, hook="", style=style), encoding="utf-8")
            try:
                await run(["-hide_banner", "-y", "-f", "lavfi", "-i", "color=c=0x1d1530:s=1080x1920:d=2.5",
                           "-vf", ass_filter(ass), "-ss", "0.7", "-frames:v", "1", str(png)], timeout=60)
                data = base64.b64encode(png.read_bytes()).decode()
            except (FFmpegError, OSError) as e:
                log.warning("caption preview failed: %s", e)
                return HTMLResponse("<p class='bad text-sm'>Preview failed; check the logs.</p>")
    return HTMLResponse(f"<img src='data:image/png;base64,{data}' alt='Rendered caption preview' "
                        "class='w-full border-2' style='border-color:var(--ink)'>")


@app.get("/fonts/{name}")
async def font_file(name: str):
    """Bundled caption fonts for the live preview. Only the listed files, never a path."""
    from ..edit import caption_style as cs

    if name not in {f for f in cs.FONTS.values() if f}:
        raise HTTPException(404)
    p = cs.fonts_dir() / name
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, media_type="font/ttf", headers={"Cache-Control": "max-age=86400"})


@app.get("/logs", response_class=HTMLResponse)
async def logs(request: Request):
    return templates.TemplateResponse(request, "logs.html", {"page": "logs"})


@app.get("/partials/logs", response_class=HTMLResponse)
async def logs_partial():
    from html import escape

    return "<pre class='text-xs whitespace-pre-wrap'>" + escape("\n".join(list(RING)[-300:][::-1])) + "</pre>"
