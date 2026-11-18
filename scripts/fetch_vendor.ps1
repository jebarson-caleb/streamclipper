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

# Caption fonts (OFL / Apache-2.0) from Google's fonts repository at a fixed commit, plus their licenses.
$fontsBase = "https://raw.githubusercontent.com/google/fonts/5bd87dfc525c8a0d1d8529d3684716e00c3fc59a"
$fonts = @(
    @("ofl/archivoblack/ArchivoBlack-Regular.ttf", "dd9a89a019b4849f66ab75455fe7bdf931311042cbb0f0f97acc061539703180"),
    @("ofl/anton/Anton-Regular.ttf", "a4ba3a92350ebb031da0cb47630ac49eb265082ca1bc0450442f4a83ab947cab"),
    @("ofl/bebasneue/BebasNeue-Regular.ttf", "08e4623805102d819f58601e46e345648846075e363b2ceb23313c2d1c83ec73"),
    @("ofl/bangers/Bangers-Regular.ttf", "4160a7311de9342674cce9160cde9fcbb30f48190397d86ff1b70b455af65824"),
    @("apache/luckiestguy/LuckiestGuy-Regular.ttf", "cfbdd68a039f92df51cf3721506af6242e64594c6325fe0bedbeff3fe385d980"),
    @("apache/permanentmarker/PermanentMarker-Regular.ttf", "28f82c8a7943cb8e9d599f8554da1d4fc75dbcf69b9885ad6c0611d20c6946c5"),
    @("ofl/instrumentserif/InstrumentSerif-Regular.ttf", "498efd461f6ddfcb7a111bf9a565709d2085d48201d501ead960d93e84ffbb88"),
    @("ofl/instrumentserif/InstrumentSerif-Italic.ttf", "08939b8bdf534afec24ae0ef5e03f948940cd9a8fe08e7fecbad040e62327385")
)
New-Item -ItemType Directory -Force vendor\fonts | Out-Null
foreach ($f in $fonts) {
    $dest = Join-Path "vendor\fonts" ([IO.Path]::GetFileName($f[0]))
    if (-not (Test-Path $dest) -or (Get-FileHash $dest -Algorithm SHA256).Hash.ToLower() -ne $f[1]) {
        Invoke-WebRequest -Uri "$fontsBase/$($f[0])" -OutFile $dest -UseBasicParsing
        if ((Get-FileHash $dest -Algorithm SHA256).Hash.ToLower() -ne $f[1]) { Remove-Item $dest; throw "font SHA-256 mismatch: $($f[0])" }
    }
    # plain string ops: Split-Path would turn "apache/x" into "apache\x" (wrong URL, wrong license type)
    $dir = $f[0].Substring(0, $f[0].LastIndexOf("/")); $licName = if ($dir.StartsWith("apache/")) { "LICENSE.txt" } else { "OFL.txt" }
    $lic = Join-Path "vendor\fonts" ($dir.Substring($dir.LastIndexOf("/") + 1) + "-" + $licName)
    if (-not (Test-Path $lic)) { Invoke-WebRequest -Uri "$fontsBase/$dir/$licName" -OutFile $lic -UseBasicParsing }
}
Write-Host "fonts: ok"

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
