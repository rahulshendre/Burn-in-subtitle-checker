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


def test_grade_is_duration_weighted() -> None:
    # A long illegible caption drags the grade further than a brief flash would.
    long_bad = video_legibility([_event(0, 10, CONTRAST_FLOOR), _event(10, 11, CONTRAST_CEIL)])
    brief_bad = video_legibility([_event(0, 1, CONTRAST_FLOOR), _event(1, 11, CONTRAST_CEIL)])
    assert long_bad is not None and brief_bad is not None
    assert long_bad.score < brief_bad.score
