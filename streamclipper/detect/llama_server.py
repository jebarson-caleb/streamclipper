"""Bundled local AI: llama.cpp's llama-server plus a GGUF model the app downloads once.

The server starts on demand and stops after `llm.keep_alive` idle, so it shares a small GPU with Whisper.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import secrets
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
_API_KEY = secrets.token_urlsafe(24)   # per launch: other programs/websites can't use the local AI


def server_exe() -> Path:
    return vendor_dir("llama") / ("llama-server.exe" if sys.platform == "win32" else "llama-server")


def available() -> bool:
    return server_exe().exists()


def model_path() -> Path:
    s = get_settings()
    return s.data_dir / "models" / s.llm.gguf_url.rsplit("/", 1)[-1]


def model_ready() -> bool:
    return model_path().exists() and (model_path().with_suffix(".ok")).exists()


FIT_MARGIN_MIB = 768   # VRAM to leave free when fitting the model on the GPU
CONTEXT_MIB = 450      # KV cache + compute buffers at a 4096 context
STALL_SECONDS = 60     # no data for this long: drop the connection and resume
STALL_RETRIES = 8
RETRY_DELAY = 2.0      # seconds, times the attempt number


async def _fetch(url: str, part: Path):
    """One attempt: resume `part` from its current size. Returns the running SHA-256 of the whole file."""
    h = hashlib.sha256()
    done = part.stat().st_size if part.exists() else 0
    if done:                       # hash what we already have
        with open(part, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    DL["done"] = done
    headers = {"Range": f"bytes={done}-"} if done else {}
    async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(60, read=STALL_SECONDS)) as c:
        async with c.stream("GET", url, headers=headers) as r:
            if r.status_code == 200 and done:   # server ignored the range: start over
                done = DL["done"] = 0
                h = hashlib.sha256()
                part.unlink(missing_ok=True)
            r.raise_for_status()
            DL["total"] = done + int(r.headers.get("content-length", 0))
            chunks = r.aiter_bytes().__aiter__()   # as it arrives (a chunk size would buffer up to it first)
            with open(part, "ab") as f:
                while True:
                    try:   # a server that stops sending without closing must not hang us forever
                        chunk = await asyncio.wait_for(chunks.__anext__(), STALL_SECONDS)
                    except StopAsyncIteration:
                        break
                    f.write(chunk)
                    h.update(chunk)
                    DL["done"] += len(chunk)
    if DL["total"] and DL["done"] < DL["total"]:
        raise httpx.ReadError("connection closed early")
    return h


def remove_old_models() -> None:
    """Delete models this app downloaded before (marked by a .ok file) once the current one is ready."""
    keep = model_path().with_suffix(".ok")
    for ok in model_path().parent.glob("*.ok"):
        if ok != keep:
            gguf = ok.with_suffix(".gguf")
            log.info("removing the previous AI model %s", gguf.name)
            gguf.unlink(missing_ok=True)
            ok.unlink(missing_ok=True)


async def download_model() -> None:
    """Download the GGUF once (resumable), verify SHA-256. Progress goes to DL."""
    if model_ready():
        remove_old_models()
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
        for attempt in range(1, STALL_RETRIES + 1):
            try:
                h = await _fetch(cfg.gguf_url, part)
                break
            except (httpx.HTTPError, asyncio.TimeoutError, OSError) as e:
                if attempt == STALL_RETRIES:
                    raise RuntimeError(f"download kept failing ({e or 'no data'}); check your connection") from e
                log.warning("AI model download interrupted (%s); resuming (attempt %d)", e or "stalled", attempt + 1)
                await asyncio.sleep(min(30, RETRY_DELAY * attempt))
        if cfg.gguf_sha256 and h.hexdigest() != cfg.gguf_sha256:
            part.unlink(missing_ok=True)
            raise RuntimeError("downloaded model failed its checksum; it will be fetched again")
        part.replace(dest)
        dest.with_suffix(".ok").write_text(h.hexdigest(), encoding="utf-8")
        remove_old_models()
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


def _nvidia_free_mib() -> int | None:
    """Free video memory from nvidia-smi. The Vulkan driver over-reports it on NVIDIA laptops (it counted
    3.2 GB free when 0.9 GB was), which made llama.cpp's own fitting crash mid-answer."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, creationflags=_NO_WINDOW).stdout
        return int(out.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def gpu_layers(free_mib: int | None) -> int:
    """Layers to put on the GPU: -1 = let llama.cpp decide (no nvidia-smi), 0 = CPU only, else a count."""
    if free_mib is None:
        return -1
    cfg = get_settings().llm
    model_mib = (model_path().stat().st_size / 2 ** 20) if model_path().exists() else 2400
    usable = free_mib - FIT_MARGIN_MIB - CONTEXT_MIB
    if usable < 300:
        return 0
    return max(1, min(cfg.gguf_layers, int(cfg.gguf_layers * usable / model_mib)))


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
           "-c", "4096", "--api-key", _API_KEY, "--jinja"]
    # GPU: let llama.cpp fit as many layers as the free video memory allows (games, OBS and Whisper use it
    # too) and run the rest on the CPU. A fixed "-ngl 99" crashes mid-answer when VRAM is short.
    if cpu:
        cmd += ["-ngl", "0"]
    else:
        layers = gpu_layers(_nvidia_free_mib())
        if layers == 0:
            log.info("not enough free video memory for the local AI; using the CPU")
            cpu = True
            cmd += ["-ngl", "0"]
        else:   # NVIDIA: a layer count from nvidia-smi; other GPUs: llama.cpp's own fitting
            cmd += ["-ngl", str(layers)] if layers > 0 else ["--fit", "on", "--fit-target", str(FIT_MARGIN_MIB)]
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
    global _last_used, _cpu_only
    body = {"temperature": 0.4, "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},   # Qwen3: answer directly, no reasoning text
            "max_tokens": 600,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    for attempt in (1, 2):
        await ensure_running()
        try:
            r = await client.post(f"{_url()}/v1/chat/completions", json=body,
                                  headers={"Authorization": f"Bearer {_API_KEY}"})
            break
        except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectError):
            # the engine died mid-answer (usually out of video memory): retry once on the CPU
            if attempt == 2 or (_proc and _proc.poll() is None):
                raise
            log.warning("local AI stopped during a request; restarting it on the CPU")
            stop()
            _cpu_only = True
    _last_used = time.monotonic()
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    text = r.json()["choices"][0]["message"]["content"] or ""
    return re.sub(r"(?s)<think>.*?</think>", "", text).strip()
