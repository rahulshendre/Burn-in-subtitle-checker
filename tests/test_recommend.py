"""Tests for additive legibility recommendations."""

from subtitle_checker.artifacts import SubtitleEvent
from subtitle_checker.subtitles.legibility import CONTRAST_CEIL, CONTRAST_FLOOR
from subtitle_checker.subtitles.recommend import advice_summary, legibility_advice


def _ev(legibility, ratio=None, lh=None, text="line") -> SubtitleEvent:
    return SubtitleEvent(
        start=0.0,
        end=2.0,
        text=text,
        confidence=0.9,
        legibility=legibility,
        line_count=1,
        legibility_ratio=ratio,
        line_height_frac=lh,
    )


def test_clear_line_gets_no_advice() -> None:
    # Above the Clear band -> nothing to fix, even with low factors set.
    assert legibility_advice([_ev(CONTRAST_CEIL, ratio=2.0, lh=0.03)]) == []


def test_low_contrast_line_gets_contrast_tip() -> None:
    advice = legibility_advice([_ev(CONTRAST_FLOOR, ratio=2.7, lh=0.09)])
    assert len(advice) == 1
    assert advice[0].low_contrast and not advice[0].small_text
    assert "contrast" in advice[0].tips[0]


def test_small_text_line_gets_size_tip() -> None:
    advice = legibility_advice([_ev(CONTRAST_FLOOR, ratio=9.0, lh=0.04)])
    assert len(advice) == 1
    assert advice[0].small_text and not advice[0].low_contrast
    assert "small text" in advice[0].tips[0]


def test_both_factors_short_gives_two_tips() -> None:
    advice = legibility_advice([_ev(CONTRAST_FLOOR, ratio=2.7, lh=0.04)])
    assert len(advice[0].tips) == 2


def test_below_clear_but_factors_ok_gives_no_tip() -> None:
    # A low grade with both diagnostic factors above target has no actionable fix,
    # so it is not advised - recommendations need a concrete factor to name.
    assert legibility_advice([_ev(CONTRAST_FLOOR, ratio=9.0, lh=0.09)]) == []


def test_unmeasured_or_empty_lines_skipped() -> None:
    assert legibility_advice([_ev(None, ratio=2.0, lh=0.03)]) == []
    assert legibility_advice([_ev(CONTRAST_FLOOR, ratio=2.0, lh=0.03, text="  ")]) == []


def test_summary_leads_with_the_dominant_issue() -> None:
    advice = legibility_advice(
        [
            _ev(CONTRAST_FLOOR, ratio=2.7, lh=0.09),  # contrast
            _ev(CONTRAST_FLOOR, ratio=2.5, lh=0.09),  # contrast
            _ev(CONTRAST_FLOOR, ratio=9.0, lh=0.04),  # size
        ]
    )
    summary = advice_summary(advice)
    assert summary is not None
    assert "contrast" in summary and "3 line" in summary


def test_summary_none_when_no_advice() -> None:
    assert advice_summary([]) is None
