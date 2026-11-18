@echo off
rem Double-click to start StreamClipper in the background and open the dashboard.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo First run: installing everything. This takes a few minutes...
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
)
start "" ".venv\Scripts\pythonw.exe" -m streamclipper run --open
