"""Tests for whole-video subtitle legibility scoring."""

import numpy as np

from subtitle_checker.artifacts import SubtitleEvent
from subtitle_checker.subtitles.legibility import (
    CONTRAST_CEIL,
    CONTRAST_FLOOR,
    contrast,
    line_score,
    video_legibility,
)


def _crop(background: int, text: int, text_fraction: float = 0.3) -> np.ndarray:
    """A synthetic grayscale crop: a bright text band over a flat background."""
    crop = np.full((100, 100), background, dtype=np.uint8)
    rows = int(100 * text_fraction)
    crop[:rows, :] = text
    return crop


def test_contrast_high_for_white_on_dark() -> None:
    # Bright text over a dark scene: a full brightness range, easy to read.
    assert contrast(_crop(background=20, text=250)) > 0.7


def test_contrast_low_for_white_on_bright() -> None:
    # White text over a bright background - the washed-out case a channel cares
    # about: text and background share one brightness, so contrast collapses.
    assert contrast(_crop(background=235, text=250)) < 0.1


def test_contrast_empty_crop_is_zero() -> None:
    assert contrast(np.empty((0, 0), dtype=np.uint8)) == 0.0


def test_line_score_clamps_to_the_calibrated_band() -> None:
    assert line_score(CONTRAST_FLOOR) == 0.0
    assert line_score(CONTRAST_FLOOR - 0.1) == 0.0
    assert line_score(CONTRAST_CEIL) == 100.0
    assert line_score(CONTRAST_CEIL + 0.5) == 100.0
    midpoint = (CONTRAST_FLOOR + CONTRAST_CEIL) / 2
    assert line_score(midpoint) == 50.0


def _event(start: float, end: float, legibility: float | None) -> SubtitleEvent:
    return SubtitleEvent(start=start, end=end, text="x", legibility=legibility)


def test_video_legibility_none_when_nothing_measured() -> None:
    events = [_event(0, 1, None), _event(1, 2, None)]
    assert video_legibility(events) is None


def test_video_legibility_grades_and_surfaces_worst_lines() -> None:
    events = [
        _event(0, 1, CONTRAST_CEIL),  # score 100
        _event(1, 2, CONTRAST_FLOOR),  # score 0
        _event(2, 3, None),  # skipped
    ]
    result = video_legibility(events, worst_n=1)
    assert result is not None
    assert result.line_count == 2
    assert result.score == 50.0
    assert len(result.worst) == 1
    assert result.worst[0].start == 1  # the least legible line comes first


def test_worst_excludes_clear_lines() -> None:
    # A line that already reads clearly is not surfaced as "least legible", even
    # when the video has few lines. The list is captions worth inspecting, not a
    # fixed bottom-N, so a clear line never fills a slot next to a real problem.
    events = [
        _event(0, 1, CONTRAST_CEIL),  # score 100 -> Clear
        _event(1, 2, CONTRAST_FLOOR),  # score 0 -> Poor
    ]
    result = video_legibility(events, worst_n=5)
    assert result is not None
    assert [round(line.score) for line in result.worst] == [0]


def test_worst_empty_when_every_line_is_clear() -> None:
    events = [_event(0, 1, CONTRAST_CEIL), _event(1, 2, CONTRAST_CEIL)]
    result = video_legibility(events, worst_n=5)
    assert result is not None
    assert result.worst == []


def test_unreadable_line_is_not_scored() -> None:
    # A band with bright, high-contrast pixels but no caption OCR could read must
    # not surface a legibility score - contrast would grade the scene, not a line.
    events = [
        _event(0, 1, CONTRAST_CEIL),  # a real, readable line
        SubtitleEvent(start=1, end=2, text="", legibility=CONTRAST_CEIL),  # unreadable
    ]
    result = video_legibility(events, worst_n=5)
    assert result is not None
    assert result.line_count == 1
    assert all(line.text for line in result.worst)


def test_none_when_only_unreadable_lines() -> None:
    events = [SubtitleEvent(start=0, end=1, text="  ", legibility=CONTRAST_CEIL)]
    assert video_legibility(events) is None


def test_grade_is_duration_weighted() -> None:
    # A long illegible caption drags the grade further than a brief flash would.
    long_bad = video_legibility([_event(0, 10, CONTRAST_FLOOR), _event(10, 11, CONTRAST_CEIL)])
    brief_bad = video_legibility([_event(0, 1, CONTRAST_FLOOR), _event(1, 11, CONTRAST_CEIL)])
    assert long_bad is not None and brief_bad is not None
    assert long_bad.score < brief_bad.score


def test_bands_bucket_subtitle_time_by_readability() -> None:
    # One clear line, one mixed, one poor - each a known on-screen duration. The
    # bands report how many seconds of the video sit in each readability band.
    span = CONTRAST_CEIL - CONTRAST_FLOOR
    mixed = CONTRAST_FLOOR + 0.60 * span  # maps to score 60 -> Mixed
    result = video_legibility(
        [
            _event(0, 4, CONTRAST_CEIL),  # score 100 -> Clear, 4s
            _event(4, 6, mixed),  # score 60 -> Mixed, 2s
            _event(6, 7, CONTRAST_FLOOR),  # score 0 -> Poor, 1s
        ]
    )
    assert result is not None
    assert [b.label for b in result.bands] == ["Clear", "Mixed", "Poor"]
    assert {b.label: b.seconds for b in result.bands} == {
        "Clear": 4.0,
        "Mixed": 2.0,
        "Poor": 1.0,
    }
    assert round(sum(b.share for b in result.bands), 5) == 1.0
