"""Quality check for rendered clips. Usage:  python scripts/qa.py [config.yaml]

Grades every ready clip on: format, start/finish (no black/frozen edges), crop smoothness,
subtitles, audio loudness/sync, metadata, and long-form structure. Exit code 1 if anything fails.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlmodel import select  # noqa: E402

from streamclipper.config import load_settings  # noqa: E402
from streamclipper.db import session  # noqa: E402
from streamclipper.ffmpeg_utils import FFMPEG, FFPROBE  # noqa: E402
from streamclipper.models import Channel, Clip  # noqa: E402

AW, AH = 96, 170


def probe(path: str) -> dict:
    out = subprocess.run([FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
                         capture_output=True, text=True).stdout
    return json.loads(out)


def frames(path: str, start: float, dur: float, w: int, h: int, fps: int = 10) -> np.ndarray:
    raw = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{max(0, start):.3f}", "-t", f"{dur:.3f}", "-i", path,
                          "-vf", f"fps={fps},scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    n = len(raw) // (w * h)
    return np.frombuffer(raw[: n * w * h], dtype=np.uint8).reshape(n, h, w).astype(np.float32)


def loudness(path: str) -> tuple[float, float]:
    err = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", path, "-af", "ebur128=peak=true",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    i = re.findall(r"I:\s+(-?[\d.]+) LUFS", err)
    p = re.findall(r"Peak:\s+(-?[\d.]+) dBFS", err)
    return (float(i[-1]) if i else -99.0, float(p[-1]) if p else 0.0)


def srt_cues(path: Path) -> list[tuple[float, float, str]]:
    if not path.exists():
        return []
    ts = lambda s: sum(float(x) * m for x, m in zip(s.replace(",", ".").split(":"), (3600, 60, 1)))
    cues = []
    for block in path.read_text(encoding="utf-8").strip().split("\n\n"):
        lines = block.strip().splitlines()
        if len(lines) >= 3 and "-->" in lines[1]:
            a, b = lines[1].split(" --> ")
            cues.append((ts(a), ts(b), " ".join(lines[2:])))
    return cues


def camera_stats(log_path: Path) -> dict[str, dict]:
    """camera path lines precede the 'rendered <file>' line of the same render."""
    out, pending = {}, None
    if not log_path.exists():
        return out
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"camera path: max ([\d.]+) px/frame, p95 ([\d.]+), max accel ([\d.]+)", line)
        if m:
            pending = {"max": float(m.group(1)), "p95": float(m.group(2)), "acc": float(m.group(3))}
            continue
        m = re.search(r"rendered (\S+\.mp4) \((\w+),", line)
        if m:
            out[m.group(1)] = dict(pending or {}, layout=m.group(2))
            pending = None
    return out


def check(clip: Clip, ch: Channel, cam: dict, titles: list[str]) -> list[str]:
    fails: list[str] = []
    p = Path(clip.path)
    if not p.exists():
        return ["file missing"]
    info = probe(str(p))
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    dur = float(info["format"]["duration"])
    short = clip.format == "short"
    W, H = (1080, 1920) if short else (1920, 1080)
    # 1. format
    if not v or (int(v["width"]), int(v["height"])) != (W, H):
        fails.append(f"resolution {v and v['width']}x{v and v['height']} != {W}x{H}")
    if v and v.get("r_frame_rate") not in ("30/1",):
        fails.append(f"fps {v.get('r_frame_rate')}")
    if short and not (6.0 <= dur <= 60.0):
        fails.append(f"short duration {dur:.1f}s outside 6-60s")
    if not a:
        fails.append("no audio")
    elif abs(float(a.get("duration", dur)) - float(v.get("duration", dur) if v else dur)) > 0.15:
        fails.append(f"A/V length mismatch {a.get('duration')} vs {v.get('duration')}")
    # 2. start / finish: no black and no frozen frames just inside the fades
    w, h = (AW, AH) if short else (AH, AW)
    head = frames(str(p), 0.25, 1.0, w, h)
    tail = frames(str(p), dur - 1.4, 1.0, w, h)
    for name, fr in (("start", head), ("end", tail)):
        if len(fr) < 2:
            fails.append(f"{name}: could not read frames")
            continue
        if fr.mean() < 14:
            fails.append(f"{name} is black (luma {fr.mean():.0f})")
        if np.abs(np.diff(fr, axis=0)).mean() < 0.25:
            fails.append(f"{name} is frozen (end card/logo/still)")
    # 3. crop smoothness (tracking layout only)
    c = cam.get(p.name, {})
    if c.get("layout") == "track":
        if "max" not in c:
            fails.append("camera path stats missing (rendered before smoothing)")
        elif c["max"] > 14.0 or c["acc"] > 1.5:
            fails.append(f"camera jerky (max {c['max']} px/frame, accel {c['acc']})")
    # 4. audio
    lufs, peak = loudness(str(p))
    if not (-17.5 <= lufs <= -11.0):
        fails.append(f"loudness {lufs} LUFS (target -14)")
    if peak > -0.5:
        fails.append(f"clipping risk: peak {peak} dBFS")
    # 5. subtitles
    cues = srt_cues(Path(clip.srt_path)) if clip.srt_path else []
    prev_end = -1.0
    for s0, e0, txt in cues:
        if e0 <= s0 or s0 < -0.05 or e0 > dur + 0.3:
            fails.append(f"subtitle timing invalid: {s0:.2f}-{e0:.2f} '{txt[:20]}'")
        if s0 < prev_end - 0.05:
            fails.append(f"subtitles overlap at {s0:.2f}")
        if e0 - s0 > 6.0:
            fails.append(f"subtitle on screen {e0 - s0:.1f}s: '{txt[:25]}'")
        if len(txt.split()) == 1 and e0 - s0 > 1.6:
            fails.append(f"single word held {e0 - s0:.1f}s: '{txt}'")
        prev_end = e0
    # 6. metadata
    t = clip.title or ""
    if not t.strip() or len(t) > 100:
        fails.append(f"bad title length {len(t)}")
    if titles.count(t) > 1:
        fails.append("duplicate title")
    if short and "#shorts" not in t.lower() and "#shorts" not in clip.description.lower():
        fails.append("no #shorts tag")
    if ch.permission == "allowed" and ch.url not in clip.description and not ch.credit_text:
        fails.append("missing creator credit in description")
    if len(clip.description) > 5000 or sum(len(x) + 1 for x in clip.tags.split(",")) > 500:
        fails.append("description/tags too long")
    # 7. long form
    if not short:
        if not clip.thumb_path or not Path(clip.thumb_path).exists():
            fails.append("no thumbnail")
        if "," in clip.moment_ids:
            chap = [ln for ln in clip.description.splitlines() if re.match(r"^\d+:\d\d ", ln)]
            if len(chap) < 3 or not chap[0].startswith("0:00"):
                fails.append("compilation without valid chapters")
    return fails


def main() -> int:
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    s = load_settings(cfg)
    cam = camera_stats(s.log_path)
    bad = 0
    with session() as db:
        rows = db.exec(select(Clip, Channel).join(Channel, Channel.id == Clip.channel_id)
                       .where(Clip.status.in_(["ready", "approved", "scheduled", "published"]))
                       .order_by(Clip.stream_id, Clip.id)).all()
    by_stream: dict[int, list[str]] = {}
    for c, _ in rows:
        by_stream.setdefault(c.stream_id, []).append(c.title)
    for clip, ch in rows:
        fails = check(clip, ch, cam, by_stream[clip.stream_id])
        lay = cam.get(Path(clip.path).name, {}).get("layout", "-")
        status = "PASS" if not fails else "FAIL"
        bad += bool(fails)
        print(f"[{status}] {ch.name:10} clip {clip.id:3} {clip.format:5} {lay:9} {clip.duration:5.1f}s  {clip.title[:50]}")
        for f in fails:
            print(f"         - {f}")
    print(f"\n{len(rows) - bad}/{len(rows)} clips pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
