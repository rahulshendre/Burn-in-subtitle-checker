#!/usr/bin/env bash
# Build "Subtitle Checker.app" for Apple Silicon Macs.
#
# Uses a clean torch-free virtualenv (the app runs the Sarvam stack, ONNX VAD
# and no forced alignment) and bundles static ffmpeg/ffprobe builds, so the
# tester's Mac needs nothing installed. Output: dist/Subtitle Checker.app and
# dist/Subtitle-Checker-mac.zip.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="$ROOT/build/macos"
PY="${PYTHON:-python3.13}"
FF_URL="https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/release"

mkdir -p "$WORK/bin"
if [ ! -x "$WORK/venv/bin/pyinstaller" ]; then
  "$PY" -m venv "$WORK/venv"
  "$WORK/venv/bin/pip" install -q --upgrade pip
fi
"$WORK/venv/bin/pip" install -q "$ROOT[asr,vision,audio]" pyinstaller

for tool in ffmpeg ffprobe; do
  if [ ! -x "$WORK/bin/$tool" ]; then
    curl -sSL -o "$WORK/$tool.zip" "$FF_URL/$tool.zip"
    unzip -o -q "$WORK/$tool.zip" -d "$WORK/bin"
  fi
done

"$WORK/venv/bin/pyinstaller" --noconfirm --clean --windowed \
  --name "Subtitle Checker" \
  --osx-bundle-identifier org.planetread.subtitlechecker \
  --workpath "$WORK/pyi" --distpath "$ROOT/dist" --specpath "$WORK" \
  --add-binary "$WORK/bin/ffmpeg:bin" \
  --add-binary "$WORK/bin/ffprobe:bin" \
  --collect-data subtitle_checker \
  --collect-submodules subtitle_checker \
  "$ROOT/packaging/macos/launcher.py"

cd "$ROOT/dist"
rm -f Subtitle-Checker-mac.zip
ditto -c -k --keepParent "Subtitle Checker.app" Subtitle-Checker-mac.zip
du -sh "Subtitle Checker.app" Subtitle-Checker-mac.zip
