"""Startup self-checks so the tool works without manual steps: ffmpeg, Ollama server, LLM model."""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx

from .config import get_settings
from .ffmpeg_utils import FFMPEG

log = logging.getLogger(__name__)
STATUS: dict[str, str] = {}


def _ollama_exe() -> str | None:
    hit = shutil.which("ollama")
    if hit:
        return hit
    for p in (Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
              Path(r"C:\Program Files\Ollama\ollama.exe")):
        if p.exists():
            return str(p)
    return None


async def _tags() -> list[str] | None:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{get_settings().llm.base_url}/api/tags")
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


async def ensure_ollama() -> None:
    cfg = get_settings().llm
    models = await _tags()
    exe = _ollama_exe()
    if models is None:
        if not exe:
            STATUS["llm"] = "Ollama not installed (clips will use signal scores only)"
            log.warning(STATUS["llm"])
            return
        log.info("starting Ollama server")
        flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS if sys.platform == "win32" else 0
        subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL, creationflags=flags)
        for _ in range(30):
            await asyncio.sleep(1)
            models = await _tags()
            if models is not None:
                break
        if models is None:
            STATUS["llm"] = "Ollama did not start"
            log.warning(STATUS["llm"])
            return
    if not any(m == cfg.model or m.startswith(cfg.model + ":") for m in models):
        if not exe:
            return
        STATUS["llm"] = f"downloading {cfg.model}…"
        log.info("pulling LLM %s (first run only)", cfg.model)
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        await asyncio.to_thread(subprocess.run, [exe, "pull", cfg.model], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, creationflags=flags)
    STATUS["llm"] = "ready"


async def run_checks() -> None:
    STATUS["ffmpeg"] = FFMPEG if Path(FFMPEG).exists() or shutil.which(FFMPEG) else "MISSING"
    if STATUS["ffmpeg"] == "MISSING":
        log.error("ffmpeg not found. Install it: winget install Gyan.FFmpeg")
    from .detect.llm import bundled

    if bundled():   # packaged exe: download the AI model in the background, whisper loads meanwhile
        from .detect import llama_server

        dl = asyncio.create_task(llama_server.download_model())
        await preload_whisper()
        await dl
        STATUS["llm"] = "ready" if llama_server.model_ready() else f"failed: {llama_server.DL['error']}"
    else:
        await ensure_ollama()
        await preload_whisper()


async def preload_whisper() -> None:
    """Load the speech model at startup so the first minutes of a live stream aren't spent loading it."""
    from .gpu import gpu_lock
    from .transcribe import whisper

    STATUS["whisper"] = "loading"
    try:
        async with gpu_lock():
            await asyncio.to_thread(whisper.warm_up)
        STATUS["whisper"] = "ready"
    except Exception as e:
        STATUS["whisper"] = f"failed: {e}"
        log.warning("could not preload whisper: %s", e)
