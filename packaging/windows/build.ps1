# Build "Subtitle Checker" for 64-bit Windows. Run from any folder in PowerShell:
#   powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1
#
# Same recipe as the Mac build: a clean torch-free virtualenv (Sarvam stack,
# ONNX VAD, no forced alignment) plus static ffmpeg/ffprobe, frozen with
# PyInstaller. Output: dist\Subtitle Checker\ (run "Subtitle Checker.exe") and
# dist\Subtitle-Checker-windows.zip.
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
$Work = Join-Path $Root "build\windows"
$Py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$FfUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

New-Item -ItemType Directory -Force -Path "$Work\bin" | Out-Null
if (-not (Test-Path "$Work\venv\Scripts\pyinstaller.exe")) {
    & $Py -m venv "$Work\venv"
    & "$Work\venv\Scripts\python.exe" -m pip install -q --upgrade pip
}
& "$Work\venv\Scripts\python.exe" -m pip install -q "$Root[asr,vision,audio]" pyinstaller

if (-not (Test-Path "$Work\bin\ffmpeg.exe")) {
    Invoke-WebRequest -Uri $FfUrl -OutFile "$Work\ffmpeg.zip"
    Expand-Archive -Force "$Work\ffmpeg.zip" "$Work\ffmpeg"
    Get-ChildItem "$Work\ffmpeg" -Recurse -Include ffmpeg.exe, ffprobe.exe |
        Copy-Item -Destination "$Work\bin"
}

& "$Work\venv\Scripts\pyinstaller.exe" --noconfirm --clean --noconsole `
    --name "Subtitle Checker" `
    --workpath "$Work\pyi" --distpath "$Root\dist" --specpath $Work `
    --add-binary "$Work\bin\ffmpeg.exe;bin" `
    --add-binary "$Work\bin\ffprobe.exe;bin" `
    --collect-data subtitle_checker `
    --collect-submodules subtitle_checker `
    "$Root\packaging\launcher.py"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$Zip = Join-Path $Root "dist\Subtitle-Checker-windows.zip"
if (Test-Path $Zip) { Remove-Item $Zip }
Compress-Archive -Path "$Root\dist\Subtitle Checker" -DestinationPath $Zip
Get-Item $Zip | Select-Object Name, @{n = "MB"; e = { [math]::Round($_.Length / 1MB) } }
