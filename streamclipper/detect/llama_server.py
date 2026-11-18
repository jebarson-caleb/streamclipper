"""Bundled local AI: llama.cpp's llama-server plus a GGUF model the app downloads once.

The server starts on demand and stops after `llm.keep_alive` idle, so it shares a small GPU with Whisper.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

from ..config import get_settings
from ..vendor import vendor_dir

log = logging.getLogger(__name__)
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
DL: dict = {"state": "idle", "done": 0, "total": 0, "error": ""}   # shown on the setup guide
_proc: subprocess.Popen | None = None
_cpu_only = False
_last_used = 0.0
_reaper: asyncio.Task | None = None
_start_lock = asyncio.Lock()


def server_exe() -> Path:
    return vendor_dir("llama") / ("llama-server.exe" if sys.platform == "win32" else "llama-server")


def available() -> bool:
    return server_exe().exists()


def model_path() -> Path:
    s = get_settings()
    return s.data_dir / "models" / s.llm.gguf_url.rsplit("/", 1)[-1]


def model_ready() -> bool:
    return model_path().exists() and (model_path().with_suffix(".ok")).exists()


async def download_model() -> None:
    """Download the GGUF once (resumable), verify SHA-256. Progress goes to DL."""
    if model_ready():
        DL.update(state="ready")
        return
    if DL["state"] == "downloading":
        return
    cfg = get_settings().llm
    dest = model_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    DL.update(state="downloading", error="", done=part.stat().st_size if part.exists() else 0, total=0)
    try:
        h = hashlib.sha256()
        if part.exists():   # hash what we already have, then resume
            with open(part, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        headers = {"Range": f"bytes={DL['done']}-"} if DL["done"] else {}
        async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(60, read=120)) as c:
            async with c.stream("GET", cfg.gguf_url, headers=headers) as r:
                if r.status_code == 200 and DL["done"]:   # server ignored the range: start over
                    DL["done"] = 0
                    h = hashlib.sha256()
                    part.unlink(missing_ok=True)
                r.raise_for_status()
                DL["total"] = DL["done"] + int(r.headers.get("content-length", 0))
                with open(part, "ab") as f:
                    async for chunk in r.aiter_bytes(1 << 20):
                        f.write(chunk)
                        h.update(chunk)
                        DL["done"] += len(chunk)
        if cfg.gguf_sha256 and h.hexdigest() != cfg.gguf_sha256:
            part.unlink(missing_ok=True)
            raise RuntimeError("downloaded model failed its checksum; it will be fetched again")
        part.replace(dest)
        dest.with_suffix(".ok").write_text(h.hexdigest(), encoding="utf-8")
        DL.update(state="ready")
        log.info("AI model ready: %s", dest)
    except Exception as e:
        DL.update(state="failed", error=str(e)[:300])
        log.error("AI model download failed: %s", e)


def _gpu_device() -> str | None:
    """Prefer a dedicated GPU (NVIDIA/AMD/Intel Arc) over an integrated one."""
    try:
        out = subprocess.run([str(server_exe()), "--list-devices"], capture_output=True, text=True, timeout=30,
                             creationflags=_NO_WINDOW, cwd=server_exe().parent).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    devs = re.findall(r"^\s*(\w+\d+):\s*(.+)$", out, re.M)
    for dev, name in devs:
        if re.search(r"nvidia|radeon|\barc\b", name, re.I):
            return dev
    return None


def _url() -> str:
    return f"http://127.0.0.1:{get_settings().llm.server_port}"


async def _healthy() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            return (await c.get(f"{_url()}/health")).status_code == 200
    except httpx.HTTPError:
        return False


async def _start(cpu: bool) -> bool:
    global _proc
    cfg = get_settings().llm
    cmd = [str(server_exe()), "-m", str(model_path()), "--host", "127.0.0.1", "--port", str(cfg.server_port),
           "-c", "4096", "-ngl", "0" if cpu else "99"]
    dev = None if cpu else await asyncio.to_thread(_gpu_device)
    if dev:
        cmd += ["--device", dev]
    log.info("starting local AI (%s)", "CPU" if cpu else (dev or "default device"))
    _proc = subprocess.Popen(cmd, cwd=server_exe().parent, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW)
    for _ in range(240):
        if _proc.poll() is not None:
            return False
        if await _healthy():
            return True
        await asyncio.sleep(0.5)
    stop()
    return False


async def ensure_running() -> None:
    global _cpu_only, _reaper, _last_used
    _last_used = time.monotonic()
    async with _start_lock:
        if _proc and _proc.poll() is None and await _healthy():
            return
        if not model_ready():
            raise RuntimeError("AI model not downloaded yet")
        if not await _start(_cpu_only):
            if _cpu_only:
                raise RuntimeError("local AI failed to start")
            log.warning("local AI did not start on the GPU; using the CPU")
            _cpu_only = True
            if not await _start(True):
                raise RuntimeError("local AI failed to start")
        if _reaper is None or _reaper.done():
            _reaper = asyncio.create_task(_reap())


def _idle_seconds() -> float:
    m = re.match(r"(\d+)\s*([smh]?)", str(get_settings().llm.keep_alive).strip())
    return int(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)] if m else 30


async def _reap() -> None:
    while _proc and _proc.poll() is None:
        await asyncio.sleep(5)
        if time.monotonic() - _last_used > _idle_seconds():
            log.info("stopping idle local AI")
            stop()


def stop() -> None:
    global _proc
    if _proc and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(10)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None


async def chat(client: httpx.AsyncClient, system: str, user: str) -> str:
    global _last_used
    await ensure_running()
    r = await client.post(f"{_url()}/v1/chat/completions", json={
        "temperature": 0.4, "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    _last_used = time.monotonic()
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]
