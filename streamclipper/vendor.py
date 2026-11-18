"""Tools bundled with the installed app.

Installed layout (scripts/build.ps1):  <app>\\runtime\\pythonw.exe (official signed Python),
<app>\\runtime\\Lib\\site-packages\\..., <app>\\vendor\\{ffmpeg,llama}\\, <app>\\vendor\\.packaged
In a source checkout the same tools live in <repo>\\vendor\\ (scripts/fetch_vendor.ps1).
"""
from __future__ import annotations

import sys
from pathlib import Path


def app_root() -> Path:
    return Path(sys.executable).resolve().parent.parent


def packaged() -> bool:
    """True when running as the installed app, not from a source checkout."""
    return (app_root() / "vendor" / ".packaged").exists()


def vendor_dir(name: str) -> Path:
    base = app_root() / "vendor" if packaged() else Path(__file__).resolve().parent.parent / "vendor"
    return base / name
