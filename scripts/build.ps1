# Builds the Windows installer: dist\StreamClipper-Setup-<version>.exe
# Run from the project folder:
#   powershell -ExecutionPolicy Bypass -File scripts\build.ps1          # needs Inno Setup: winget install JRSoftware.InnoSetup
#   powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Cuda    # also bundle NVIDIA cuBLAS/cuDNN for GPU Whisper (~2 GB more)
#
# The app runs on the official, signed Python runtime from python.org (no PyInstaller bootloader), so Windows
# Smart App Control trusts the program that loads everything. Layout of build\app (= the install folder):
#   runtime\pythonw.exe + Lib\site-packages\   official Python 3.11 embeddable + StreamClipper and its libraries
#   vendor\ffmpeg  vendor\llama  vendor\deno   bundled tools (scripts\fetch_vendor.ps1)
#   config.example.yaml  icon.ico  LICENSE
param([switch]$Cuda)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = ".\.venv\Scripts\python.exe"   # build-time Python 3.11 (runs pip for the bundled runtime)
if (-not (Test-Path $py)) {
    Write-Host "Creating virtual environment..."
    py -3.11 -m venv .venv
}
& $py -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"
if ($LASTEXITCODE -ne 0) { throw "the build venv must be Python 3.11 (same as the bundled runtime)" }

& powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "fetch_vendor.ps1")
if ($LASTEXITCODE -ne 0) { throw "fetch_vendor failed" }

$app = "build\app"
if (Test-Path $app) { Remove-Item -Recurse -Force $app }
New-Item -ItemType Directory -Force "$app\runtime", "$app\vendor" | Out-Null
Copy-Item "vendor\python\*" "$app\runtime" -Recurse
# let the embeddable runtime see site-packages and run site (it ignores PYTHONPATH by design)
Set-Content -Encoding ascii "$app\runtime\python311._pth" "python311.zip`r`n.`r`nLib\site-packages`r`nimport site"

$site = "$app\runtime\Lib\site-packages"
& $py -m pip install --upgrade pip
& $py -m pip install --target $site --no-warn-script-location .
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
if ($Cuda) {
    & $py -m pip install --target $site --no-warn-script-location nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
    if ($LASTEXITCODE -ne 0) { throw "pip install (CUDA) failed" }
}
# SQLAlchemy's compiled speedups are optional (each has a pure-Python .py next to it) and some of them are
# blocked by Windows Smart App Control, so ship the pure-Python versions.
Get-ChildItem "$site\sqlalchemy" -Recurse -Filter "*_cy.*.pyd" | ForEach-Object {
    if (-not (Test-Path (Join-Path $_.DirectoryName (($_.Name -split '\.')[0] + ".py")))) { throw "no pure-Python fallback for $($_.Name)" }
    Remove-Item $_.FullName
}
# console launchers pip writes to bin\ are unused (the app runs modules with -m); keep only deno for yt-dlp
New-Item -ItemType Directory -Force "$app\vendor\deno" | Out-Null
Move-Item "$site\bin\deno.exe" "$app\vendor\deno\deno.exe"
Remove-Item -Recurse -Force "$site\bin"

Copy-Item "vendor\ffmpeg" "$app\vendor\ffmpeg" -Recurse
New-Item -ItemType Directory -Force "$app\vendor\llama" | Out-Null
# ggml-rpc.dll (multi-machine inference) isn't used and Smart App Control blocks it
Get-ChildItem "vendor\llama" -File | Where-Object { ($_.Extension -eq ".dll" -and $_.Name -ne "ggml-rpc.dll") -or $_.Name -eq "llama-server.exe" -or $_.Name -like "LICENSE*" } |
    Copy-Item -Destination "$app\vendor\llama"
Remove-Item "$app\vendor\ffmpeg\.ok" -ErrorAction SilentlyContinue
Set-Content "$app\vendor\.packaged" "installed layout"
Copy-Item config.example.yaml, LICENSE, packaging\icon.ico $app

# precompile so the first launch is quick
& "$app\runtime\python.exe" -m compileall -q -j 0 $site | Out-Null

# checks: nothing Windows Application Control would block, the app imports, the bundled tools run
& "$app\runtime\python.exe" scripts\check_blocked.py $app
if ($LASTEXITCODE -ne 0) { throw "some bundled files are blocked by Windows Application Control (see above)" }
& "$app\runtime\python.exe" -c "import streamclipper.web.app, streamclipper.detect.llama_server, streamlink.stream.hls, faster_whisper, cv2, av, ctranslate2, Crypto.Cipher.AES; print('imports ok')"
if ($LASTEXITCODE -ne 0) { throw "bundled runtime failed to import the app" }
$ErrorActionPreference = "Continue"   # llama-server prints its version on stderr
foreach ($tool in @(@("ffmpeg\ffmpeg.exe", "-version"), @("llama\llama-server.exe", "--version"), @("deno\deno.exe", "--version"))) {
    $out = & "$app\vendor\$($tool[0])" $tool[1] 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "bundled $($tool[0]) failed to run: $out" }
    Write-Host "$($tool[0]): $(($out | Select-Object -First 1))"
}
$ErrorActionPreference = "Stop"

$version = (Select-String -Path pyproject.toml -Pattern '^version\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
$iscc = (Get-Command iscc -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
    $iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
              "$env:ProgramFiles\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $iscc) { throw "Inno Setup not found. Install it: winget install JRSoftware.InnoSetup" }
New-Item -ItemType Directory -Force dist | Out-Null
& $iscc "/DAppVersion=$version" packaging\installer.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
Write-Host ""
Write-Host "Installer: dist\StreamClipper-Setup-$version.exe"
