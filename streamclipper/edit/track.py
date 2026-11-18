"""Action tracking for 9:16 crops: follow where the motion is (players, ball, car) with a smooth
virtual camera, instead of a fixed centre crop or a face lock that grabs bystanders."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..ffmpeg_utils import FFmpegError, run_bytes

log = logging.getLogger(__name__)

AW, AH, FPS = 192, 108, 6          # analysis resolution / rate


@dataclass
class Track:
    times: list[float]             # seconds from clip start
    centers: list[float]           # crop centre, normalised 0..1 of source width
    coverage: float                # share of motion energy the 9:16 window captured (0..1)


async def load_frames(concat: Path, start: float, dur: float) -> np.ndarray:
    raw = await run_bytes(
        ["-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat), "-an", "-vf",
         f"trim=start={start:.3f}:duration={dur:.3f},setpts=PTS-STARTPTS,fps={FPS},scale={AW}:{AH},format=gray",
         "-f", "rawvideo", "pipe:1"],
        timeout=300,
    )
    n = len(raw) // (AW * AH)
    return np.frombuffer(raw[: n * AW * AH], dtype=np.uint8).reshape(n, AH, AW).astype(np.float32)


OUT_FPS = 30                       # camera path is computed per output frame (no 6 Hz stepping)
MAX_SPEED = 0.30                   # max pan speed, source widths per second
MAX_ACCEL = 0.60                   # max change of pan speed per second (eases in and out)


def _gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    """Zero-phase Gaussian smoothing (edge padded) - no lag, natural ease in/out."""
    if len(x) < 3 or sigma <= 0:
        return x
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, (r, r), mode="edge"), k, mode="valid")


def _limit(x: np.ndarray, fps: float) -> np.ndarray:
    """Cap speed and acceleration (forward then backward pass so the result stays centred)."""
    vmax, amax = MAX_SPEED / fps, MAX_ACCEL / fps / fps
    for seq in (x, x[::-1]):
        v = 0.0
        out = seq.copy()
        for i in range(1, len(seq)):
            want = seq[i] - out[i - 1]
            dv = np.clip(want - v, -amax, amax)
            v = float(np.clip(v + dv, -vmax, vmax))
            out[i] = out[i - 1] + v
        seq[:] = out
    return x


def plan_shot(targets: np.ndarray, crop_frac: float) -> np.ndarray:
    """One shot's camera path at OUT_FPS from per-frame action centres (NaN = no information)."""
    n = len(targets)
    if n == 0:
        return targets
    ok = ~np.isnan(targets)
    if not ok.any():
        return np.full(n, 0.5)
    idx = np.arange(n)
    t = np.interp(idx, idx[ok], targets[ok])                       # fill gaps, no freezing/snapping
    lo, hi = np.percentile(t, 10), np.percentile(t, 90)
    if hi - lo <= crop_frac * 0.8:
        # the action stays inside the frame: a locked camera looks best (no motion at all)
        return np.full(n, float((lo + hi) / 2))
    path = _gauss(t, sigma=0.8 * OUT_FPS)                          # ~0.8 s smoothing
    # dead-zone follower: hold still until the action drifts > 8 % of the crop width away,
    # so small back-and-forth never moves the camera (no hunting)
    dz = crop_frac * 0.08
    cam = np.empty_like(path)
    c = path[0]
    for i, v in enumerate(path):
        if v - c > dz:
            c = v - dz
        elif c - v > dz:
            c = v + dz
        cam[i] = c
    path = _limit(cam, OUT_FPS)                                    # cap speed + acceleration
    return _gauss(path, sigma=0.25 * OUT_FPS)                      # ease the starts and stops


def compute_track(frames: np.ndarray, crop_frac: float, cuts: list[float] | None = None) -> Track:
    """crop_frac: crop width / source width (9:16 of a 16:9 source ~ 0.316). Returns a path at OUT_FPS."""
    n = len(frames)
    if n < 3:
        return Track([0.0], [0.5], 0.0)
    diff = np.abs(np.diff(frames, axis=0))                      # n-1 motion maps
    # ignore the outer 6 % of rows (score bugs / tickers / letterbox) and pixel noise
    r0 = int(AH * 0.06)
    diff = diff[:, r0:AH - r0, :]
    diff[diff < 12] = 0
    smooth = lambda m: np.apply_along_axis(lambda c: np.convolve(c, np.ones(9) / 9, mode="same"), 1, m)
    motion = smooth(diff.sum(axis=1))                           # (n-1, AW) motion energy per column
    m_total = motion.sum(axis=1) + 1e-6
    # visual detail (edges/contrast): where a still subject, a title card or a face is
    img = frames[1:, r0:AH - r0, :]
    gx = np.abs(np.diff(img, axis=2, prepend=img[:, :, :1]))
    gy = np.abs(np.diff(img, axis=1, prepend=img[:, :1, :]))
    edge = gx + gy
    edge[edge < 18] = 0
    detail = smooth(edge.sum(axis=1))
    d_total = detail.sum(axis=1) + 1e-6
    # frames with almost no motion carry no information from motion: use detail alone there
    quiet = m_total < np.percentile(m_total, 20) * 0.5 + 1
    w_motion = np.where(quiet, 0.0, 0.7)[:, None]
    col = w_motion * motion / m_total[:, None] + (1 - w_motion) * detail / d_total[:, None]
    win = max(4, int(round(crop_frac * AW)))
    cs = np.cumsum(np.pad(col, ((0, 0), (1, 0))), axis=1)
    win_energy = cs[:, win:] - cs[:, :-win]                     # energy in each window position
    best_left = win_energy.argmax(axis=1)
    total = col.sum(axis=1) + 1e-6
    centers = (best_left + win / 2) / AW
    cover = win_energy.max(axis=1) / total
    # frames with neither motion nor detail (black, flat colour) carry no information
    blank = quiet & (d_total < 50)
    centers[blank] = np.nan
    # resample analysis samples (FPS) onto output frames (OUT_FPS)
    dur = n / FPS
    t_an = (np.arange(len(centers)) + 1) / FPS
    t_out = np.arange(int(dur * OUT_FPS) + 1) / OUT_FPS
    near = np.clip(np.searchsorted(t_an, t_out), 0, len(t_an) - 1)
    targets = centers[near]
    # plan each shot separately: the camera may only jump at a hard cut (where it's invisible)
    cut_idx = sorted({int(round(c * OUT_FPS)) for c in (cuts or []) if 0 < c * OUT_FPS < len(t_out)})
    bounds = [0] + cut_idx + [len(t_out)]
    path = np.concatenate([plan_shot(targets[s:e], crop_frac) for s, e in zip(bounds, bounds[1:]) if e > s])
    half = crop_frac / 2
    path = np.clip(path, half, 1 - half)
    weights = m_total / m_total.sum()
    return Track([float(x) for x in t_out], [float(v) for v in path], float((cover * weights).sum()))


def path_stats(track: Track, src_w: int, cuts: list[float] | None = None) -> dict:
    """Smoothness metrics in source pixels per output frame, ignoring the frame at each cut."""
    x = np.array(track.centers) * src_w
    if len(x) < 3:
        return {"max_step": 0.0, "p95_step": 0.0, "max_accel": 0.0}
    step = np.abs(np.diff(x))
    acc = np.abs(np.diff(np.diff(x)))
    skip = {int(round(c * OUT_FPS)) + d for c in (cuts or []) for d in (-1, 0, 1)}
    keep_s = [v for i, v in enumerate(step) if i + 1 not in skip and i not in skip]
    keep_a = [v for i, v in enumerate(acc) if not ({i, i + 1, i + 2} & skip)]
    return {"max_step": float(max(keep_s, default=0)), "p95_step": float(np.percentile(keep_s, 95)) if keep_s else 0.0,
            "max_accel": float(max(keep_a, default=0))}


def sendcmd_file(track: Track, src_w: int, crop_w: int, path: Path, target: str = "crop@trk") -> int:
    """Write an ffmpeg sendcmd script that moves the crop; returns the initial x."""
    lines, last = [], None
    for t, c in zip(track.times, track.centers):
        x = int(round(c * src_w - crop_w / 2))
        x = max(0, min(src_w - crop_w, x))       # 1 px precision (crop runs with exact=1)
        if x != last:
            lines.append(f"{t:.3f} {target} x {x};")
            last = x
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return int(lines[0].split()[-1].rstrip(";")) if lines else (src_w - crop_w) // 2
