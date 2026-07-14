"""ffmpeg-backed evidence for the HTML report.

The renderer (report.html) is pure: it asks an Evidence provider for the media
each card embeds. This module implements that provider against the source video
with two short ffmpeg calls - a scaled PNG frame at a timestamp, and a small MP3
clip of a span. Both stream to stdout, so nothing touches disk. A failed or
empty ffmpeg run returns None and the renderer draws a placeholder: a report
must still generate even if one snippet cannot be cut.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from subtitle_checker.artifacts import CheckResult, SubtitleEvent
from subtitle_checker.report.html import render_report

FRAME_WIDTH = 320
AUDIO_RATE = 22_050
AUDIO_BITRATE = "64k"


class FfmpegEvidence:
    """Cuts frames and audio snippets from one video on demand via ffmpeg."""

    def __init__(self, video: Path, *, frame_width: int = FRAME_WIDTH) -> None:
        self._video = str(video)
        self._width = frame_width

    def frame_png(self, t: float) -> bytes | None:
        return _run(
            [
                "ffmpeg", "-v", "error",
                "-ss", f"{max(0.0, t):.3f}",
                "-i", self._video,
                "-frames:v", "1",
                "-vf", f"scale={self._width}:-2",
                "-f", "image2pipe", "-c:v", "png", "-",
            ]
        )

    def audio_clip(self, start: float, end: float) -> tuple[bytes, str] | None:
        data = _run(
            [
                "ffmpeg", "-v", "error",
                "-ss", f"{max(0.0, start):.3f}",
                "-i", self._video,
                "-t", f"{max(0.1, end - start):.3f}",
                "-vn", "-ac", "1", "-ar", str(AUDIO_RATE),
                "-c:a", "libmp3lame", "-b:a", AUDIO_BITRATE,
                "-f", "mp3", "-",
            ]
        )
        return (data, "audio/mpeg") if data else None


def _run(cmd: list[str]) -> bytes | None:
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return proc.stdout or None


def write_report(
    video: Path,
    results: list[CheckResult],
    out_path: Path,
    *,
    title: str | None = None,
    generated: str | None = None,
    skipped: list[tuple[SubtitleEvent, str]] | None = None,
) -> Path:
    """Render ``results`` against ``video`` and write a self-contained HTML file."""
    evidence = FfmpegEvidence(video)
    document = render_report(
        results, evidence, title=title or video.stem, generated=generated,
        skipped=skipped,
    )
    out_path.write_text(document, encoding="utf-8")
    return out_path
