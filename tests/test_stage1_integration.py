"""End-to-end Stage 1 check: burn known subtitles, detect them back.

Uses the evaluation harness's own burner to build the fixture — the harness
grades the detector. ASCII text keeps the test independent of which fonts
the CI runner has; Devanagari rendering is verified separately on real
clips.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from subtitle_checker.artifacts import SubtitleEvent
from subtitle_checker.evaluation.burn import burn_subtitles
from subtitle_checker.subtitles.reconstruct import detect_raw_events

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)

TRUTH = [
    SubtitleEvent(start=1.0, end=3.0, text="HELLO WORLD ONE"),
    SubtitleEvent(start=4.5, end=6.5, text="SECOND LINE HERE"),
    SubtitleEvent(start=8.0, end=10.0, text="THIRD AND LAST"),
]

# sampling at 4 fps quantises boundaries to 0.25s; allow one sample of slack
TOLERANCE_S = 0.3


@pytest.fixture(scope="module")
def burned_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("stage1")
    base = tmp / "base.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=black:s=640x360:d=12",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(base),
        ],
        check=True,
    )
    return burn_subtitles(base, TRUTH, tmp / "burned.mp4", font="DejaVu Sans")


def test_detects_each_burned_line_once(burned_video: Path) -> None:
    events = detect_raw_events(burned_video)
    assert len(events) == len(TRUTH)


def test_event_timing_matches_truth(burned_video: Path) -> None:
    events = detect_raw_events(burned_video)
    for detected, truth in zip(sorted(events, key=lambda e: e.start), TRUTH):
        assert detected.start == pytest.approx(truth.start, abs=TOLERANCE_S)
        assert detected.end == pytest.approx(truth.end, abs=TOLERANCE_S)
