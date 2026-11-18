"""First-run guide: what StreamClipper needs and whether it's ready. The packaged exe bundles ffmpeg and the
AI engine; the AI and speech models download on first launch."""
from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from .config import get_settings

log = logging.getLogger(__name__)
_tasks: set[asyncio.Task] = set()


def _flag() -> Path:
    return get_settings().data_dir / ".setup-complete"


def is_complete() -> bool:
    if _flag().exists():
        return True
    from sqlmodel import select

    from .db import session
    from .models import Channel

    with session() as db:   # existing installs that already have channels skip the guide
        return db.exec(select(Channel)).first() is not None


def mark_complete() -> None:
    _flag().write_text("ok", encoding="utf-8")


def ffmpeg_path() -> str | None:
    from .ffmpeg_utils import _find

    p = _find("ffmpeg")
    return p if Path(p).exists() else shutil.which(p)


def status() -> dict:
    from .bootstrap import STATUS, _ollama_exe
    from .detect import llama_server
    from .detect.llm import bundled
    from .publish.youtube_auth import connected_accounts

    s = get_settings()
    dl = dict(llama_server.DL)
    if bundled():
        ai = "ready" if llama_server.model_ready() else dl["state"]
    else:
        ai = "ready" if STATUS.get("llm") == "ready" else ("missing" if not _ollama_exe() else "starting")
    pct = int(dl["done"] * 100 / dl["total"]) if dl.get("total") else 0
    return {"ffmpeg": ffmpeg_path(), "bundled": bundled(), "ai": ai, "ai_pct": pct, "ai_error": dl.get("error", ""),
            "ai_gb": round(dl.get("total", 0) / 1e9, 1) or 2.1, "whisper": STATUS.get("whisper", "loading"),
            "secret": Path(s.publish.client_secret).exists(), "accounts": connected_accounts()}


def busy(st: dict) -> bool:
    return st["ai"] in ("downloading", "idle", "starting") or st["whisper"] == "loading"


def retry_ai() -> None:
    """Restart a failed model download (bundled engine) or Ollama setup (source install)."""
    from .bootstrap import ensure_ollama
    from .detect import llama_server
    from .detect.llm import bundled

    t = asyncio.create_task(llama_server.download_model() if bundled() else ensure_ollama())
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)
