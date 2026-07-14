"""Pure alignment scoring over a fake aligner - no torch, no model."""

from __future__ import annotations

import numpy as np

from subtitle_checker.artifacts import SubtitleEvent
from subtitle_checker.match.align import (
    SAMPLE_RATE,
    WordSpan,
    _mean_score,
    score_event,
    score_events,
)

# Ten seconds of silence - long enough that every event window has samples.
AUDIO = np.zeros(SAMPLE_RATE * 10, dtype=np.float32)

# Sentinel standing in for an injected wrong word (see evaluation.defects).
WRONG = "XXXX"


class FakeAligner:
    """Each word gets a one-second slot; a word scores 0.9 unless it is the
    WRONG sentinel, which scores 0.1. Deterministic, torch-free."""

    def align(self, audio: np.ndarray, text: str) -> list[WordSpan]:
        spans = []
        for i, word in enumerate(text.split()):
            score = 0.1 if word == WRONG else 0.9
            spans.append(WordSpan(word, float(i), float(i) + 1.0, score))
        return spans


def test_matching_line_scores_high() -> None:
    event = SubtitleEvent(2.0, 5.0, "यह सच है")
    result = score_event(event, AUDIO, FakeAligner())
    assert result.score == 0.9
    assert result.text == "यह सच है"


def test_one_wrong_word_drops_the_score() -> None:
    clean = score_event(SubtitleEvent(2.0, 5.0, "यह सच है"), AUDIO, FakeAligner())
    swapped = score_event(SubtitleEvent(2.0, 5.0, f"यह {WRONG} है"), AUDIO, FakeAligner())
    assert swapped.score < clean.score
    assert swapped.score < 0.7  # a single swap is separable from a clean line


def test_empty_text_is_unalignable() -> None:
    # no words to align → None (UNCHECKABLE), not a zero-score mismatch
    event = SubtitleEvent(3.0, 4.0, "")
    result = score_event(event, AUDIO, FakeAligner())
    assert result.score is None
    assert (result.aligned_start, result.aligned_end) == (3.0, 4.0)


def test_window_beyond_audio_end_is_unalignable() -> None:
    # event past the 10 s clip → empty window → no spans → None, not zero
    event = SubtitleEvent(50.0, 52.0, "बहुत दूर")
    result = score_event(event, AUDIO, FakeAligner())
    assert result.score is None


def test_aligned_span_is_offset_into_absolute_time() -> None:
    # window starts at event.start - pad (0.2); three 1 s words → 0..3 in-window
    event = SubtitleEvent(5.0, 8.0, "एक दो तीन")
    result = score_event(event, AUDIO, FakeAligner(), pad=0.2)
    assert result.aligned_start == 4.8
    assert result.aligned_end == 4.8 + 3.0


def test_score_events_keeps_one_result_per_event_in_order() -> None:
    events = [SubtitleEvent(1.0, 2.0, "पहला"), SubtitleEvent(3.0, 4.0, "दूसरा")]
    results = score_events(events, AUDIO, FakeAligner())
    assert [r.text for r in results] == ["पहला", "दूसरा"]


def test_mean_score_weights_by_duration() -> None:
    # a long confident word outweighs a short weak one: (0.2*3 + 1.0*1) / 4
    spans = [WordSpan("a", 0.0, 3.0, 0.2), WordSpan("b", 3.0, 4.0, 1.0)]
    assert _mean_score(spans) == 0.4


def test_mean_score_zero_length_falls_back_to_plain_mean() -> None:
    spans = [WordSpan("a", 1.0, 1.0, 0.4), WordSpan("b", 2.0, 2.0, 0.6)]
    assert _mean_score(spans) == 0.5


def test_mean_score_of_nothing_is_zero() -> None:
    assert _mean_score([]) == 0.0
