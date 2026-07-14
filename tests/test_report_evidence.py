"""ffmpeg evidence extractor - cuts real frames and audio from a synthetic clip."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from subtitle_checker.artifacts import CheckResult, Verdict
from subtitle_checker.report.evidence import FfmpegEvidence, write_report

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def _make_clip(path: Path, duration: float = 2.0) -> None:
    subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=15:duration={duration}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-shortest", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


def test_frame_png_returns_a_png(tmp_path: Path) -> None:
    clip = tmp_path / "v.mp4"
    _make_clip(clip)
    png = FfmpegEvidence(clip).frame_png(1.0)
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_audio_clip_returns_mp3_bytes(tmp_path: Path) -> None:
    clip = tmp_path / "v.mp4"
    _make_clip(clip)
    got = FfmpegEvidence(clip).audio_clip(0.2, 1.5)
    assert got is not None
    data, mime = got
    assert mime == "audio/mpeg"
    assert len(data) > 100


def test_missing_video_returns_none(tmp_path: Path) -> None:
    ev = FfmpegEvidence(tmp_path / "nope.mp4")
    assert ev.frame_png(0.5) is None
    assert ev.audio_clip(0.0, 1.0) is None


def test_write_report_creates_self_contained_html(tmp_path: Path) -> None:
    clip = tmp_path / "v.mp4"
    _make_clip(clip)
    results = [CheckResult(0.5, 1.5, Verdict.TEXT_MISMATCH, "differs", "अ", "आ", 0.2)]
    out = write_report(clip, results, tmp_path / "report.html", generated="2026-07-10")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "data:image/png;base64," in text
    assert "data:audio/mpeg;base64," in text
