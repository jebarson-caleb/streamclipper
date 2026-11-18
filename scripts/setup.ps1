# One-time setup: installs Python 3.11, ffmpeg and Ollama (via winget), creates .venv, installs deps, pulls the LLM.
# Run from the project folder:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Need($cmd) { -not (Get-Command $cmd -ErrorAction SilentlyContinue) }

if (Need "py") {
    Write-Host "Installing Python 3.11..."
    winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
}
if (Need "ffmpeg") {
    Write-Host "Installing ffmpeg (full build, has NVENC + libass)..."
    winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
}
if (Need "ollama") {
    Write-Host "Installing Ollama..."
    winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements
}

# Refresh PATH for this session so the new tools are visible
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    py -3.11 -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# CUDA libraries for faster-whisper on GPU (cuBLAS + cuDNN 9)
& .\.venv\Scripts\python.exe -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"

if (-not (Test-Path "config.yaml")) { Copy-Item config.example.yaml config.yaml }

Write-Host "Pulling local LLM (about 2 GB)..."
ollama pull qwen3:4b

& powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "create_shortcut.ps1")

Write-Host ""
Write-Host "Setup complete."
Write-Host "  1. Put your Google OAuth desktop client JSON here as client_secret.json"
Write-Host "  2. Start:  .\.venv\Scripts\python.exe -m streamclipper run"
Write-Host "  3. Open http://127.0.0.1:8765 , add channels, connect YouTube in Settings"
Write-Host "  4. Optional: scripts\install_service.ps1 to start automatically at logon"
