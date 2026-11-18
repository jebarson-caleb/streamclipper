# PyInstaller spec for StreamClipper.exe. Build with:  powershell -ExecutionPolicy Bypass -File scripts\build.ps1
# Set STREAMCLIPPER_CUDA=1 to bundle the NVIDIA cuBLAS/cuDNN DLLs (GPU Whisper, adds ~2 GB).
import os
import shutil
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).parent
datas, binaries, hiddenimports = [], [], []

for pkg in ["streamclipper", "streamlink", "streamlink_cli", "yt_dlp", "faster_whisper", "ctranslate2", "av",
            "chat_downloader", "googleapiclient", "google_auth_oauthlib", "webview", "uvicorn", "curl_cffi",
            "certifi", "tzdata", "onnxruntime", "cv2"]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += collect_submodules("sqlmodel") + collect_submodules("sqlalchemy.dialects.sqlite")

datas += [(str(ROOT / "config.example.yaml"), ".")]

deno = shutil.which("deno")
if deno:
    binaries += [(deno, ".")]

excludes = ["tkinter", "pytest", "IPython", "matplotlib"]
if os.environ.get("STREAMCLIPPER_CUDA") == "1":
    d, b, h = collect_all("nvidia")
    datas += d
    binaries += b
else:
    excludes += ["nvidia"]

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StreamClipper",
    console=False,
    icon=str(ROOT / "packaging" / "icon.ico") if (ROOT / "packaging" / "icon.ico").exists() else None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="StreamClipper")
