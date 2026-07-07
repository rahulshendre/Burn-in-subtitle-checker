"""Alignment-eval mechanics over a fake aligner that knows the true lines."""

from __future__ import annotations

import random

import numpy as np

from subtitle_checker.artifacts import SubtitleEvent
from subtitle_checker.evaluation.alignment_eval import (
    _best_threshold,
    _swap_one_word,
    evaluate_alignment,
)
from subtitle_checker.match.align import WordSpan

AUDIO = np.zeros(16_000 * 20, dtype=np.float32)


class MatchAligner:
    """Scores 0.9 for a known-correct line, 0.3 for anything else — stands in
    for an aligner that recognises the true transcript over the audio."""

    def __init__(self, correct_texts: set[str]) -> None:
        self._correct = correct_texts

    def align(self, audio: np.ndarray, text: str) -> list[WordSpan]:
        score = 0.9 if text in self._correct else 0.3
        return [WordSpan(text, 0.0, 1.0, score)]


def test_swap_one_word_changes_exactly_one_word() -> None:
    rng = random.Random(0)
    out = _swap_one_word("एक दो तीन", ["चार", "पाँच"], rng).split()
    original = "एक दो तीन".split()
    assert len(out) == 3
    assert sum(a != b for a, b in zip(original, out)) == 1


def test_best_threshold_separates_clean_distributions() -> None:
    threshold, recall, precision = _best_threshold([0.8, 0.9, 0.85], [0.2, 0.3, 0.25])
    assert 0.3 < threshold <= 0.8
    assert recall == 1.0
    assert precision == 1.0


def test_best_threshold_empty_side_is_neutral() -> None:
    assert _best_threshold([], [0.2]) == (0.5, 0.0, 0.0)


def test_evaluate_scores_correct_above_swapped() -> None:
    events = [
        SubtitleEvent(1.0, 4.0, "यह पहली पंक्ति है"),
        SubtitleEvent(5.0, 8.0, "यह दूसरी पंक्ति है"),
        SubtitleEvent(9.0, 12.0, "यह तीसरी पंक्ति है"),
    ]
    result = evaluate_alignment(events, AUDIO, MatchAligner({e.text for e in events}))
    assert result.pairs == 3
    assert result.correct_mean == 0.9
    assert result.swapped_mean == 0.3
    assert result.recall == 1.0
    assert result.precision == 1.0


def test_evaluate_filters_low_confidence_and_short_lines() -> None:
    events = [
        SubtitleEvent(1.0, 4.0, "यह अच्छी पंक्ति है", 0.9),
        SubtitleEvent(5.0, 8.0, "धुंधला पाठ यहाँ", 0.2),  # low OCR conf → skipped
        SubtitleEvent(9.0, 12.0, "छोटा", 0.9),  # one word → skipped
    ]
    result = evaluate_alignment(events, AUDIO, MatchAligner({e.text for e in events}))
    assert result.pairs == 1
