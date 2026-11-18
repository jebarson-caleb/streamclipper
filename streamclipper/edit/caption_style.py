"""How captions look: presets plus your own choices. Every value is validated, because it ends up inside the
.ass subtitle file (a font name or color with a comma or newline could otherwise inject subtitle lines)."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, fields

# family name -> bundled font file (None = Windows system font)
FONTS: dict[str, str | None] = {
    "Arial Black": None,
    "Archivo Black": "ArchivoBlack-Regular.ttf",
    "Anton": "Anton-Regular.ttf",
    "Bebas Neue": "BebasNeue-Regular.ttf",
    "Bangers": "Bangers-Regular.ttf",
    "Luckiest Guy": "LuckiestGuy-Regular.ttf",
    "Permanent Marker": "PermanentMarker-Regular.ttf",
    "Instrument Serif": "InstrumentSerif-Regular.ttf",
}
POSITIONS = ("low", "middle", "top")
_HEX = re.compile(r"#[0-9A-Fa-f]{6}")


@dataclass
class CaptionStyle:
    preset: str = "pop"
    font: str = "Arial Black"
    size: int = 78                 # at 1080x1920; long 16:9 videos use 70 % of it
    text_color: str = "#FFFFFF"
    highlight_color: str = "#FFF000"
    outline_color: str = "#000000"
    outline: int = 6               # outline width, or box padding when box is on
    box: bool = False              # solid box behind the words (uses outline_color)
    uppercase: bool = True
    position: str = "low"          # low | middle | top
    angle: int = 0                 # tilt in degrees, -15..15
    pop: bool = True               # active word scales up
    glow: bool = False             # soft blurred outline
    words_per_line: int = 3


PRESETS: dict[str, dict] = {
    "pop": {},
    "clean": {"font": "Archivo Black", "size": 70, "highlight_color": "#FFFFFF", "outline": 3, "uppercase": False,
              "pop": False},
    "boxed": {"font": "Archivo Black", "size": 68, "outline_color": "#101010", "highlight_color": "#FF77CD",
              "box": True, "outline": 14},
    "neon": {"font": "Bebas Neue", "size": 104, "outline_color": "#00E5FF", "highlight_color": "#FF77CD",
             "outline": 4, "glow": True},
    "editorial": {"font": "Archivo Black", "size": 72, "text_color": "#101010", "outline_color": "#FFFFFF",
                  "highlight_color": "#0038FF", "box": True, "outline": 16, "angle": -3},
    "marker": {"font": "Permanent Marker", "size": 84, "highlight_color": "#FF77CD", "outline": 5, "angle": -4,
               "uppercase": False},
    "comic": {"font": "Luckiest Guy", "size": 86, "highlight_color": "#FFD000", "outline": 8, "angle": 2},
}
PRESET_LABELS = {"pop": "Bold pop", "clean": "Clean", "boxed": "Boxed", "neon": "Neon", "editorial": "Editorial",
                 "marker": "Marker", "comic": "Comic"}


def preset(name: str) -> CaptionStyle:
    return CaptionStyle(preset=name if name in PRESETS else "pop", **PRESETS.get(name, {}))


def _bool(v) -> bool:
    return v is True or str(v).lower() in ("1", "true", "on", "yes")


def _int(v, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(float(v))))
    except (TypeError, ValueError, OverflowError):
        return default


def validate(data: dict) -> CaptionStyle:
    """Build a style from untrusted input (form or stored JSON). Unknown or bad values fall back to defaults."""
    base = preset(str(data.get("preset", "pop")))
    d = CaptionStyle()
    out = CaptionStyle(
        preset=base.preset if str(data.get("preset")) in PRESETS else "custom",
        font=data.get("font") if data.get("font") in FONTS else base.font,
        size=_int(data.get("size", base.size), 36, 140, base.size),
        outline=_int(data.get("outline", base.outline), 0, 24, base.outline),
        angle=_int(data.get("angle", base.angle), -15, 15, 0),
        words_per_line=_int(data.get("words_per_line", base.words_per_line), 1, 6, d.words_per_line),
        position=data.get("position") if data.get("position") in POSITIONS else base.position,
        **{k: (str(data[k]).upper() if _HEX.fullmatch(str(data.get(k, ""))) else getattr(base, k))
           for k in ("text_color", "highlight_color", "outline_color")},
        **{k: _bool(data[k]) if k in data else getattr(base, k) for k in ("box", "uppercase", "pop", "glow")},
    )
    return out


def to_json(style: CaptionStyle) -> str:
    return json.dumps(asdict(style))


def from_json(text: str) -> CaptionStyle:
    try:
        raw = json.loads(text)
    except (TypeError, ValueError):
        return CaptionStyle()
    return validate({k: v for k, v in raw.items() if k in {f.name for f in fields(CaptionStyle)}}) \
        if isinstance(raw, dict) else CaptionStyle()


def ass_color(hex_rgb: str, alpha: int = 0) -> str:
    """#RRGGBB -> &HAABBGGRR (ASS colors are blue-green-red)."""
    h = hex_rgb.lstrip("#")
    return f"&H{alpha:02X}{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def load() -> CaptionStyle:
    from ..db import session
    from ..models import CaptionSetting

    with session() as db:
        row = db.get(CaptionSetting, 1)
    return from_json(row.data) if row else CaptionStyle()


def save(style: CaptionStyle) -> None:
    from ..db import session
    from ..models import CaptionSetting

    with session() as db:
        row = db.get(CaptionSetting, 1) or CaptionSetting(id=1)
        row.data = to_json(style)
        db.add(row)
        db.commit()


def fonts_dir():
    from ..vendor import vendor_dir

    return vendor_dir("fonts")
