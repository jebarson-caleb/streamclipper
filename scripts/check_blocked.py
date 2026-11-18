"""Load every native file (.pyd/.dll) under a folder and fail if Windows Application Control blocks any.

On PCs with Smart App Control enforcing, Windows refuses to load unsigned files it has no reputation for
(error 4551). Run with the bundled runtime: build\app\runtime\python.exe scripts\check_blocked.py build\app
"""
import ctypes
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = [p for p in root.rglob("*") if p.suffix.lower() in (".pyd", ".dll")]
for d in {p.parent for p in files}:
    try:
        os.add_dll_directory(str(d))
    except OSError:
        pass
blocked = []
for p in sorted(files):
    try:
        ctypes.WinDLL(str(p))
    except OSError as e:
        if getattr(e, "winerror", None) == 4551:   # "An Application Control policy has blocked this file"
            blocked.append(p)
print(f"checked {len(files)} native files, {len(blocked)} blocked by Application Control")
for p in blocked:
    print("  BLOCKED", p.relative_to(root))
sys.exit(1 if blocked else 0)
