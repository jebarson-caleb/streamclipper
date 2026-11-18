# Registers a Windows scheduled task that starts StreamClipper hidden at logon and restarts it if it crashes.
# Run:    powershell -ExecutionPolicy Bypass -File scripts\install_service.ps1
# Remove: Unregister-ScheduledTask -TaskName StreamClipper -Confirm:$false
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$pyw = Join-Path $root ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pyw)) { throw "Run scripts\setup.ps1 first (missing $pyw)" }

$action   = New-ScheduledTaskAction -Execute $pyw -Argument "-m streamclipper run" -WorkingDirectory $root
$trigger  = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) `
            -StartWhenAvailable
Register-ScheduledTask -TaskName "StreamClipper" -Action $action -Trigger $trigger -Settings $settings `
    -Description "Local AI stream clipper" -Force | Out-Null
Start-ScheduledTask -TaskName "StreamClipper"
Write-Host "StreamClipper installed and started. Dashboard: http://127.0.0.1:8765"
