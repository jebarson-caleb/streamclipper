"""Long 16:9 highlight compilation from several moments, with chapters."""
from __future__ import annotations

import logging
from pathlib import Path

from ..ffmpeg_utils import duration, run
from ..models import Moment
from .render import render_span

log = logging.getLogger(__name__)


async def compile_moments(stream_id: int, moments: list[Moment], out: Path, credit: str = "") \
        -> tuple[str, list[tuple[float, str]], list]:
    """Returns (path, chapters [(t, title)], all words with compilation-relative times)."""
    parts, chapters, words = [], [], []
    t = 0.0
    tmp_dir = out.parent / f"{out.stem}_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(moments):
        p = tmp_dir / f"{i:02d}.mp4"
        try:
            res = await render_span(stream_id, m.start, m.end, p, vertical=False, credit=credit,
                                    peak=m.peak or None)
        except Exception as e:
            log.warning("compilation: skip moment %s: %s", m.id, e)
            continue
        d = await duration(res.path)
        parts.append(res.path)
        chapters.append((t, (m.title or f"Moment {i + 1}")[:60]))
        words += [type(w)(w.start + t, w.end + t, w.text) for w in res.words]
        t += d
    if not parts:
        raise RuntimeError("no renderable moments for compilation")
    lst = tmp_dir / "list.txt"
    lst.write_text("".join(f"file '{Path(p).as_posix()}'\n" for p in parts), encoding="utf-8")
    await run(["-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
               "-c", "copy", "-movflags", "+faststart", str(out)], timeout=1800)
    for p in parts:
        Path(p).unlink(missing_ok=True)
    lst.unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    if chapters:
        chapters[0] = (0.0, chapters[0][1])
    return str(out), chapters, words
