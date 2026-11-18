"""Per-second scene-change score using ffmpeg's scene detector on a downscaled 4 fps copy."""
from __future__ import annotations

import re

from ..ffmpeg_utils import FFmpegError, run

_PTS = re.compile(r"pts_time:([\d.]+)")
_SCORE = re.compile(r"lavfi\.scene_score=([\d.]+)")


async def scene_per_second(path: str, seconds: int) -> list[float]:
    try:
        text = await run(
            ["-hide_banner", "-i", path, "-an", "-vf",
             "fps=4,scale=256:-2,select='gte(scene,0)',metadata=print", "-f", "null", "-"],
            timeout=180,
        )
    except FFmpegError:
        return [0.0] * seconds
    out = [0.0] * seconds
    t = None
    for line in text.splitlines():
        m = _PTS.search(line)
        if m:
            t = float(m.group(1))
            continue
        m = _SCORE.search(line)
        if m and t is not None:
            i = int(t)
            if 0 <= i < seconds:
                out[i] = max(out[i], float(m.group(1)))
    return out
