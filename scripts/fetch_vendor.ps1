# Downloads the tools bundled with StreamClipper into vendor\ (pinned versions, SHA-256 checked):
#   vendor\python\  official Python 3.11 embeddable runtime (signed by the Python Software Foundation)  - PSF
#   vendor\llama\   llama.cpp llama-server (Vulkan build: NVIDIA/AMD/Intel GPU, falls back to CPU)  - MIT
#   vendor\ffmpeg\  ffmpeg.exe + ffprobe.exe (GyanD essentials build)                                - GPLv3
# Run from the project folder:  powershell -ExecutionPolicy Bypass -File scripts\fetch_vendor.ps1
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$items = @(
    @{ Name = "python"; Url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip";
       Sha = "009d6bf7e3b2ddca3d784fa09f90fe54336d5b60f0e0f305c37f400bf83cfd3b" },
    @{ Name = "llama"; Url = "https://github.com/ggml-org/llama.cpp/releases/download/b12040/llama-b12040-bin-win-vulkan-x64.zip";
       Sha = "1198da0b60ab53e9a92940e00d12b9d1baebdcf9b42e31ef3298d6dc590b6bc8" },
    @{ Name = "ffmpeg"; Url = "https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-essentials_build.zip";
       Sha = "60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba" }
)
New-Item -ItemType Directory -Force vendor\downloads | Out-Null
foreach ($it in $items) {
    $dest = Join-Path "vendor" $it.Name
    if (Test-Path (Join-Path $dest ".ok")) { Write-Host "$($it.Name): already there"; continue }
    $zip = Join-Path "vendor\downloads" ([IO.Path]::GetFileName($it.Url))
    if (-not (Test-Path $zip)) {
        Write-Host "Downloading $($it.Url)"
        Invoke-WebRequest -Uri $it.Url -OutFile $zip -UseBasicParsing
    }
    $hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
    if ($hash -ne $it.Sha) { Remove-Item $zip; throw "$($it.Name): SHA-256 mismatch ($hash)" }
    $tmp = Join-Path "vendor\downloads" ($it.Name + "-x")
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    Expand-Archive $zip -DestinationPath $tmp
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    New-Item -ItemType Directory -Force $dest | Out-Null
    if ($it.Name -eq "ffmpeg") {
        Get-ChildItem $tmp -Recurse -Include ffmpeg.exe, ffprobe.exe, LICENSE | Copy-Item -Destination $dest
    } elseif ($it.Name -eq "python") {
        Copy-Item "$tmp\*" $dest -Recurse
        $sig = Get-AuthenticodeSignature (Join-Path $dest "python.exe")
        if ($sig.Status -ne "Valid" -or $sig.SignerCertificate.Subject -notlike "*Python Software Foundation*") {
            Remove-Item -Recurse -Force $dest; throw "python: python.exe is not signed by the Python Software Foundation"
        }
    } else {
        Get-ChildItem $tmp -Recurse -File | Copy-Item -Destination $dest
    }
    Remove-Item -Recurse -Force $tmp
    Set-Content (Join-Path $dest ".ok") $it.Sha
    Write-Host "$($it.Name): ok"
}
