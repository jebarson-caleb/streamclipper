"""One heavy GPU job at a time (4 GB VRAM: Whisper and the LLM take turns)."""
from __future__ import annotations

import asyncio
import shutil
import subprocess

_lock: asyncio.Lock | None = None


def gpu_lock() -> asyncio.Lock:
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


def vram_info() -> dict:
    """Used/total MiB from nvidia-smi, or {} when unavailable."""
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[0]
        name, used, total, util = [x.strip() for x in out.split(",")]
        return {"name": name, "used": int(used), "total": int(total), "util": int(util)}
    except Exception:
        return {}


def is_oom(err: str) -> bool:
    e = err.lower()
    return "out of memory" in e or ("cuda" in e and "memory" in e)
