# Builds the desktop app: dist\StreamClipper\StreamClipper.exe (+ a zip next to it).
# Run from the project folder:
#   powershell -ExecutionPolicy Bypass -File scripts\build.ps1          # CPU Whisper, ~0.5 GB
#   powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Cuda    # bundle NVIDIA DLLs for GPU Whisper, ~2.5 GB
param([switch]$Cuda)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Creating virtual environment..."
    py -3.11 -m venv .venv
}
& $py -m pip install --upgrade pip
& $py -m pip install -e ".[desktop,build]"
if ($Cuda) {
    & $py -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
    $env:STREAMCLIPPER_CUDA = "1"
} else {
    $env:STREAMCLIPPER_CUDA = "0"
}

# deno.exe (pip package) lives in the venv Scripts folder; let the spec find it
$env:Path = (Resolve-Path ".\.venv\Scripts").Path + ";" + $env:Path

& $py -m PyInstaller packaging\streamclipper.spec --noconfirm --clean --distpath dist --workpath build
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$zip = "dist\StreamClipper-windows-x64" + $(if ($Cuda) { "-cuda" } else { "" }) + ".zip"
if (Test-Path $zip) { Remove-Item $zip }
tar.exe -a -c -f $zip -C dist StreamClipper   # Compress-Archive fails on files over 2 GB
if ($LASTEXITCODE -ne 0) { throw "zip failed" }
Write-Host ""
Write-Host "Built: dist\StreamClipper\StreamClipper.exe"
Write-Host "Zip:   $zip"
