"""Smoke tests for the CLI entrypoint."""

import shutil
import subprocess
from pathlib import Path

import pytest

from subtitle_checker import __version__
from subtitle_checker.artifacts import CheckResult, Verdict, save_artifact
from subtitle_checker.cli import _merge_results, main

_needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def _make_clip(path: Path, duration: float = 1.5) -> None:
    subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=15:duration={duration}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-shortest", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


def test_version_flag_prints_version(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help_and_succeeds(capsys: pytest.CaptureFixture) -> None:
    assert main([]) == 0
    assert "check" in capsys.readouterr().out


def test_check_rejects_missing_video(capsys: pytest.CaptureFixture) -> None:
    assert main(["check", "--video", "no_such_clip.mp4"]) == 2
    assert "not found" in capsys.readouterr().err


def test_merge_prefers_asr_evidence_over_bare_alignment_flag() -> None:
    align = CheckResult(1.0, 3.0, Verdict.TEXT_MISMATCH, "alignment", subtitle_text="X")
    asr = CheckResult(1.0, 3.0, Verdict.TEXT_MISMATCH, "asr", "X", heard_text="Y", score=0.3)
    merged = _merge_results([align], [asr])
    assert len(merged) == 1
    assert merged[0].heard_text == "Y"  # ASR row (with heard text) wins the span


def test_merge_keeps_structural_flag_and_adds_ok_ledger() -> None:
    miss = CheckResult(5.0, 7.0, Verdict.MISSING_SUBTITLE, "missing")
    ok = CheckResult(1.0, 3.0, Verdict.OK, "ok", "A", "A")
    merged = _merge_results([miss], [ok])
    spans = {(r.start, r.end) for r in merged}
    assert spans == {(5.0, 7.0), (1.0, 3.0)}


def test_merge_alignment_flag_survives_when_asr_says_ok() -> None:
    align = CheckResult(1.0, 3.0, Verdict.TEXT_MISMATCH, "alignment", subtitle_text="X")
    ok = CheckResult(1.0, 3.0, Verdict.OK, "ok", "X", "X")
    merged = _merge_results([align], [ok])
    assert len(merged) == 1
    assert merged[0].verdict is Verdict.TEXT_MISMATCH  # flag wins, OK row deduped out


def test_report_rejects_missing_video(capsys: pytest.CaptureFixture) -> None:
    assert main(["report", "--results", "out", "--video", "no_such_clip.mp4"]) == 2
    assert "not found" in capsys.readouterr().err


@_needs_ffmpeg
def test_report_rejects_missing_results(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    clip = tmp_path / "v.mp4"
    _make_clip(clip)
    assert main(["report", "--results", str(tmp_path / "missing"), "--video", str(clip)]) == 2
    assert "no check_results" in capsys.readouterr().err


@_needs_ffmpeg
def test_report_generates_self_contained_html(tmp_path: Path) -> None:
    clip = tmp_path / "v.mp4"
    _make_clip(clip)
    results = [
        CheckResult(0.4, 1.2, Verdict.TEXT_MISMATCH, "differs", "अ", "आ", 0.2),
        CheckResult(0.4, 1.2, Verdict.OK, "matches", "ठीक है", "ठीक है"),
    ]
    save_artifact(tmp_path / "v_check_results.json", "check_results", results)
    assert main(["report", "--results", str(tmp_path), "--video", str(clip)]) == 0
    out = tmp_path / "v_report.html"
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "data:image/png;base64," in text
