"""Merging a subtitle that a flicker split into two events."""

from subtitle_checker.artifacts import SubtitleEvent
from subtitle_checker.subtitles.reconstruct import _merge_repeated


def ev(start: float, end: float, text: str, conf: float = 0.9) -> SubtitleEvent:
    return SubtitleEvent(start=start, end=end, text=text, confidence=conf)


def test_identical_text_small_gap_merges() -> None:
    # One subtitle flickered off for 0.25s - same text, adjacent - is one line.
    events = [
        ev(28.5, 29.5, "अरे यही तो वायरल होगा"),
        ev(29.75, 30.5, "अरे यही तो वायरल होगा"),
    ]
    merged = _merge_repeated(events)
    assert len(merged) == 1
    assert merged[0].start == 28.5
    assert merged[0].end == 30.5  # end stretched to cover both


def test_merged_span_clears_the_short_floor() -> None:
    # Split, each piece was under 1.5s; rejoined it is 2.0s and now verifiable.
    events = [ev(28.5, 29.5, "same"), ev(29.75, 30.5, "same")]
    merged = _merge_repeated(events)
    assert merged[0].end - merged[0].start == 2.0


def test_different_text_stays_separate() -> None:
    events = [ev(1.0, 2.0, "पहली पंक्ति"), ev(2.1, 3.0, "दूसरी पंक्ति")]
    assert len(_merge_repeated(events)) == 2


def test_same_text_large_gap_stays_separate() -> None:
    # A real repeated line after a pause is two subtitles, not a flicker.
    events = [ev(1.0, 2.0, "दोहराया"), ev(5.0, 6.0, "दोहराया")]
    assert len(_merge_repeated(events)) == 2


def test_empty_text_never_merges() -> None:
    # Unreadable events (no OCR text) must not collapse into each other.
    events = [ev(1.0, 2.0, "", conf=0.0), ev(2.1, 3.0, "", conf=0.0)]
    assert len(_merge_repeated(events)) == 2


def test_empty_list() -> None:
    assert _merge_repeated([]) == []
