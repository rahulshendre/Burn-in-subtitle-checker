"""Tests for guideline-compliance grading of subtitle lines."""

from subtitle_checker.artifacts import SubtitleEvent
from subtitle_checker.subtitles.compliance import (
    MAX_CHARS_ON_SCREEN,
    check_compliance,
    line_compliance,
)


def _event(text: str, line_count: int | None) -> SubtitleEvent:
    return SubtitleEvent(start=0, end=1, text=text, line_count=line_count)


def test_short_single_line_is_compliant() -> None:
    line = line_compliance(_event("हम सब से नज़रे", line_count=1))
    assert line.chars_ok is True
    assert line.lines_ok is True
    assert line.compliant
    assert line.failures() == []


def test_over_budget_caption_fails_the_character_cap() -> None:
    line = line_compliance(_event("क" * (MAX_CHARS_ON_SCREEN + 5), line_count=2))
    assert line.chars_ok is False
    assert not line.compliant
    assert "characters" in line.failures()[0]


def test_two_line_caption_within_the_on_screen_budget_passes() -> None:
    # 60 characters over two lines is within the 70-character on-screen budget -
    # and the character check does not depend on the exact line count.
    line = line_compliance(_event("क" * 60, line_count=1))
    assert line.chars_ok is True
    assert line.compliant


def test_three_lines_break_the_line_cap() -> None:
    line = line_compliance(_event("क" * 10, line_count=3))
    assert line.lines_ok is False
    assert not line.compliant
    assert "lines on screen" in line.failures()[0]


def test_unmeasured_line_count_makes_no_claim_on_lines() -> None:
    # No line count measured (bright false event, ticker) - the tool makes no
    # claim on the line cap, but still checks the characters it can read.
    line = line_compliance(_event("हम सब", line_count=None))
    assert line.lines_ok is None
    assert line.chars_ok is True
    assert line.compliant


def test_check_compliance_none_without_readable_lines() -> None:
    assert check_compliance([_event("", 1), _event("   ", None)]) is None


def test_check_compliance_counts_and_lists_violations() -> None:
    events = [
        _event("हम सब", line_count=1),  # compliant
        _event("क" * (MAX_CHARS_ON_SCREEN + 10), line_count=2),  # too many characters
        _event("ख" * 10, line_count=3),  # too many lines
        _event("", line_count=1),  # unreadable - not graded
    ]
    comp = check_compliance(events)
    assert comp is not None
    assert comp.graded == 3
    assert comp.compliant == 1
    assert comp.chars_pass == 2 and comp.chars_measured == 3
    assert comp.lines_pass == 2 and comp.lines_measured == 3
    assert len(comp.violations) == 2
    assert round(comp.share, 3) == round(1 / 3, 3)
