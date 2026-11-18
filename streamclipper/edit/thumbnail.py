"""1280x720 thumbnail: a frame from the best moment + big outlined text."""
from __future__ import annotations

import logging
import textwrap
from pathlib import Path

from ..ffmpeg_utils import run

log = logging.getLogger(__name__)


def _font(size: int):
    from PIL import ImageFont

    for name in ("arialbd.ttf", "Arial Black.ttf", "ariblk.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


async def make_thumbnail(video: str, at: float, text: str, out: Path) -> str:
    frame = out.with_suffix(".frame.png")
    await run(["-hide_banner", "-loglevel", "error", "-y", "-ss", f"{max(0, at):.2f}", "-i", video,
               "-frames:v", "1", "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720",
               str(frame)], timeout=60)
    from PIL import Image, ImageDraw, ImageEnhance

    img = Image.open(frame).convert("RGB")
    img = ImageEnhance.Contrast(ImageEnhance.Color(img).enhance(1.3)).enhance(1.15)
    draw = ImageDraw.Draw(img)
    text = (text or "").upper()[:40]
    if text:
        font = _font(110)
        lines = textwrap.wrap(text, 14)[:2]
        y = 720 - 60 - 120 * len(lines)
        for line in lines:
            draw.text((50, y), line, font=font, fill=(255, 230, 0), stroke_width=10, stroke_fill=(0, 0, 0))
            y += 120
    img.save(out, "JPEG", quality=90)
    frame.unlink(missing_ok=True)
    return str(out)
