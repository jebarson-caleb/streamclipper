"""Thin helpers around the ffmpeg / ffprobe binaries."""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from functools import lru_cache

from .config import get_settings

log = logging.getLogger(__name__)

def _find(name: str) -> str:
    """PATH first, then the usual Windows install spots (winget, choco, scoop, C:\\ffmpeg)."""
    import glob
    import os

    from .vendor import packaged, vendor_dir

    bundled = vendor_dir("ffmpeg") / (name + (".exe" if os.name == "nt" else ""))
    if packaged() and bundled.exists():   # the installed app ships its own ffmpeg
        os.environ["PATH"] = str(bundled.parent) + os.pathsep + os.environ.get("PATH", "")
        return str(bundled)
    hit = shutil.which(name)
    if hit:
        return hit
    exe = name + (".exe" if os.name == "nt" else "")
    local = os.environ.get("LOCALAPPDATA", "")
    patterns = [
        os.path.join(local, "Microsoft", "WinGet", "Packages", "*FFmpeg*", "**", "bin", exe),
        os.path.join(local, "Microsoft", "WinGet", "Links", exe),
        r"C:\ProgramData\chocolatey\bin\\" + exe,
        os.path.join(os.path.expanduser("~"), "scoop", "shims", exe),
        r"C:\ffmpeg\bin\\" + exe,
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "ffmpeg", "bin", exe),
    ]
    for pat in patterns:
        found = sorted(glob.glob(pat, recursive=True), reverse=True)  # newest version first
        if found:
            d = os.path.dirname(found[0])
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")  # for child processes
            return found[0]
    return name


FFMPEG = _find("ffmpeg")
FFPROBE = _find("ffprobe")


class FFmpegError(RuntimeError):
    pass


async def run(args: list[str], timeout: float | None = None, binary: str = FFMPEG) -> str:
    """Run ffmpeg/ffprobe, return stderr+stdout text; raise FFmpegError on failure."""
    proc = await asyncio.create_subprocess_exec(
        binary, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise FFmpegError(f"timeout: {' '.join(args[:6])}")
    text = (out or b"").decode("utf-8", "replace") + (err or b"").decode("utf-8", "replace")
    if proc.returncode != 0:
        raise FFmpegError(text[-2000:])
    return text


async def run_bytes(args: list[str], timeout: float | None = None) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        FFMPEG, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await asyncio.wait_for(proc.communicate(), timeout)
    if proc.returncode != 0:
        raise FFmpegError((err or b"").decode("utf-8", "replace")[-2000:])
    return out


async def probe(path: str) -> dict:
    text = await run(
        ["-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        timeout=60, binary=FFPROBE,
    )
    return json.loads(text[text.index("{"):])


async def duration(path: str) -> float:
    info = await probe(path)
    return float(info.get("format", {}).get("duration", 0) or 0)


async def video_size(path: str) -> tuple[int, int]:
    info = await probe(path)
    for st in info.get("streams", []):
        if st.get("codec_type") == "video":
            return int(st["width"]), int(st["height"])
    return 1920, 1080


async def has_audio(path: str) -> bool:
    info = await probe(path)
    return any(st.get("codec_type") == "audio" for st in info.get("streams", []))


@lru_cache(maxsize=1)
def video_encoder() -> str:
    """h264_nvenc when the GPU encoder actually works, else libx264."""
    pref = get_settings().render.encoder
    if pref != "auto":
        return pref
    try:
        r = subprocess.run(
            [FFMPEG, "-hide_banner", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.5",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, timeout=30,
        )
        if r.returncode == 0:
            return "h264_nvenc"
    except Exception:
        pass
    log.warning("h264_nvenc unavailable, falling back to libx264 (CPU)")
    return "libx264"


def encode_args(crf: int = 21) -> list[str]:
    enc = video_encoder()
    if enc == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(crf + 2),
                "-b:v", "0", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "superfast", "-crf", str(crf), "-pix_fmt", "yuv420p"]


AUDIO_ARGS = ["-c:a", "aac", "-b:a", "160k", "-ar", "48000"]


def ass_path_escape(p: str) -> str:
    """Escape a Windows path for use inside an ffmpeg filter argument."""
    return p.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
