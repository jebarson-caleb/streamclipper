# Puts a "StreamClipper" shortcut on the Desktop and in the Start menu.
$root = Split-Path -Parent $PSScriptRoot
$pyw  = Join-Path $root ".venv\Scripts\pythonw.exe"
$shell = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath("Desktop")
$startMenu = Join-Path ([Environment]::GetFolderPath("Programs")) "StreamClipper.lnk"
foreach ($path in @((Join-Path $desktop "StreamClipper.lnk"), $startMenu)) {
    $lnk = $shell.CreateShortcut($path)
    $lnk.TargetPath = $pyw
    $lnk.Arguments = "-m streamclipper run --open"
    $lnk.WorkingDirectory = $root
    $lnk.IconLocation = "$env:SystemRoot\System32\imageres.dll,18"
    $lnk.Description = "StreamClipper: auto-clip live streams to YouTube"
    $lnk.Save()
    Write-Host "Created $path"
}
