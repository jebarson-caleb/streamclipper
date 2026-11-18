"""CLI: streamclipper run | auth | process-file."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from sqlmodel import select

from .config import load_settings
from .logs import setup_logging

log = logging.getLogger("streamclipper")


def _port_in_use(host: str, port: int) -> bool:
    import socket

    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def _run_dashboard(open_browser: bool = False) -> None:
    import threading
    import webbrowser

    import uvicorn

    from .config import get_settings

    s = get_settings()
    url = f"http://{s.dashboard.host}:{s.dashboard.port}"
    if _port_in_use(s.dashboard.host, s.dashboard.port):
        print(f"StreamClipper is already running: {url}")
        if open_browser:
            webbrowser.open(url)
        return
    from .web.app import app

    print(f"\n  StreamClipper dashboard: {url}\n")
    if open_browser:
        threading.Timer(2.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=s.dashboard.host, port=s.dashboard.port, log_level="warning")


def _run_desktop() -> None:
    """Open the dashboard in its own window. Starts the pipeline too, unless it is already running."""
    import threading
    import time
    import webbrowser

    from .config import get_settings

    s = get_settings()
    url = f"http://{s.dashboard.host}:{s.dashboard.port}"
    server = None
    if not _port_in_use(s.dashboard.host, s.dashboard.port):
        import uvicorn

        from .web.app import app

        server = uvicorn.Server(uvicorn.Config(app, host=s.dashboard.host, port=s.dashboard.port,
                                               log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        for _ in range(120):
            if server.started:
                break
            time.sleep(0.25)
    try:
        import webview
    except ImportError:
        log.warning("pywebview is not installed; opening the dashboard in the browser instead")
        webbrowser.open(url)
        if server:
            threading.Event().wait()   # keep the pipeline running until the process is killed
        return
    webview.create_window("StreamClipper", url, width=1280, height=860, min_size=(900, 600),
                          background_color="#0b0f17")
    webview.start()
    if server:   # window closed: shut the pipeline down cleanly
        server.should_exit = True
        time.sleep(3)


def _frozen_setup() -> None:
    """Packaged .exe: keep config.yaml and data/ next to the exe, create config.yaml on first run."""
    import os
    import shutil
    from pathlib import Path

    exe_dir = Path(sys.executable).parent
    os.chdir(exe_dir)
    cfg = exe_dir / "config.yaml"
    example = Path(getattr(sys, "_MEIPASS", exe_dir)) / "config.example.yaml"
    if not cfg.exists() and example.exists():
        shutil.copy(example, cfg)


async def _process_file(path: str, name: str, mode: str, outputs: str, upload: bool,
                        threshold: float = 7.0) -> None:
    from .config import get_settings
    from .db import session
    from .models import Channel, Clip, Stream
    from .supervisor import Supervisor

    s = get_settings()
    s.publish.enabled = upload
    with session() as db:
        ch = Channel(name=name, platform="file", url=path, permission="own", mode=mode, outputs=outputs,
                     enabled=False, threshold=threshold)
        db.add(ch)
        db.commit()
    sup = Supervisor()
    await sup.start()
    st = sup.process_file(ch)
    log.info("processing %s as stream %s", path, st.id)
    while True:
        await asyncio.sleep(5)
        with session() as db:
            cur = db.get(Stream, st.id)
            busy = db.exec(select(Clip).where(Clip.stream_id == st.id,
                                              Clip.status.in_(["pending", "rendering"]))).first()
            clips = db.exec(select(Clip).where(Clip.stream_id == st.id)).all()
        if cur.status == "finalized" and not busy:
            break
    await sup.stop()
    print("\nDone. Clips:")
    for c in clips:
        print(f"  [{c.status}] {c.format} score={c.score:.1f} {c.title}\n      {c.path}")


def _add_via_dashboard(args) -> None:
    """Send the link to the running app (so it records in the background service)."""
    from urllib.parse import parse_qs, urlparse

    import httpx

    from .config import get_settings

    s = get_settings()
    base = f"http://{s.dashboard.host}:{s.dashboard.port}"
    if not _port_in_use(s.dashboard.host, s.dashboard.port):
        print(f"StreamClipper is not running. Start it first:  python -m streamclipper run")
        sys.exit(1)
    r = httpx.post(f"{base}/quick", data={"url": args.url, "permission": args.permission, "mode": args.mode,
                                          "outputs": args.outputs, "name": args.name, "watch": "on"},
                   follow_redirects=False, timeout=120)
    loc = r.headers.get("location", "")
    print(parse_qs(urlparse(loc).query).get("msg", [f"HTTP {r.status_code}"])[0])


def cli(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="streamclipper", description="Local-AI stream clipper → YouTube")
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="start background pipeline + dashboard (default)")
    r.add_argument("--open", action="store_true", help="open the dashboard in the browser")
    sub.add_parser("desktop", help="start the pipeline and open the dashboard in its own window")
    lk =sub.add_parser("add", help="add any live link (or file) and start clipping it")
    lk.add_argument("url")
    lk.add_argument("--permission", required=True, choices=["own", "allowed"])
    lk.add_argument("--mode", default="best_moments", choices=["best_moments", "continuous", "hybrid"])
    lk.add_argument("--outputs", default="shorts", choices=["shorts", "long", "both"])
    lk.add_argument("--name", default="")
    a = sub.add_parser("auth", help="connect a YouTube account (opens browser)")
    a.add_argument("--account", default="default")
    f = sub.add_parser("process-file", help="run a local video through the pipeline (offline test)")
    f.add_argument("path")
    f.add_argument("--name", default="test")
    f.add_argument("--mode", default="best_moments", choices=["best_moments", "continuous", "hybrid"])
    f.add_argument("--outputs", default="shorts", choices=["shorts", "long", "both"])
    f.add_argument("--upload", action="store_true", help="also upload (default: render only)")
    f.add_argument("--threshold", type=float, default=7.0, help="final score needed to keep a moment (0-10)")
    args = p.parse_args(argv)
    import os

    frozen = getattr(sys, "frozen", False)
    if frozen:
        _frozen_setup()

    if sys.stdout is None:   # pythonw (no console): give libraries somewhere harmless to write
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    for stream in (sys.stdout, sys.stderr):   # Windows consoles (cp1252) choke on emoji
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    s = load_settings(args.config)
    setup_logging(s.log_path)
    cmd = args.cmd or ("desktop" if frozen else "run")
    if cmd == "run":
        _run_dashboard(getattr(args, "open", False))
    elif cmd == "desktop":
        _run_desktop()
    elif cmd == "add":
        _add_via_dashboard(args)
    elif cmd == "auth":
        from .publish.youtube_auth import authorize

        authorize(args.account)
        print(f"Connected YouTube account '{args.account}'.")
    elif cmd == "process-file":
        asyncio.run(_process_file(args.path, args.name, args.mode, args.outputs, args.upload, args.threshold))
    else:  # pragma: no cover
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    cli()
