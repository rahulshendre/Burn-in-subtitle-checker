"""Alignment verdict logic over hand-built scores and region timelines."""

from __future__ import annotations

from subtitle_checker.artifacts import AudioKind, AudioRegion, Verdict
from subtitle_checker.match.align import AlignmentScore
from subtitle_checker.match.verdicts import check_alignment

SPEECH = [AudioRegion(0.0, 5.0, AudioKind.SPEECH)]
MUSIC = [AudioRegion(0.0, 5.0, AudioKind.MUSIC)]


def _score(score, ocr_confidence=0.9, text="यह बात है") -> AlignmentScore:
    return AlignmentScore(1.0, 3.0, text, ocr_confidence, score, 1.0, 3.0)


def test_low_score_over_speech_is_text_mismatch() -> None:
    results = check_alignment([_score(0.1)], SPEECH)
    assert len(results) == 1
    assert results[0].verdict is Verdict.TEXT_MISMATCH
    assert results[0].subtitle_text == "यह बात है"
    assert results[0].score == 0.1


def test_good_score_over_speech_is_unflagged() -> None:
    assert check_alignment([_score(0.7)], SPEECH) == []


def test_event_without_speech_is_left_to_structural() -> None:
    # low score, but no speech beneath it → alignment abstains, structural owns it
    assert check_alignment([_score(0.1)], MUSIC) == []


def test_low_ocr_confidence_is_uncheckable_not_mismatch() -> None:
    results = check_alignment([_score(0.1, ocr_confidence=0.2)], SPEECH)
    assert len(results) == 1
    assert results[0].verdict is Verdict.UNCHECKABLE


def test_unalignable_none_score_over_speech_is_uncheckable() -> None:
    results = check_alignment([_score(None)], SPEECH)
    assert len(results) == 1
    assert results[0].verdict is Verdict.UNCHECKABLE


def test_threshold_is_precision_first() -> None:
    # just above the cut clears; just below trips it
    assert check_alignment([_score(0.31)], SPEECH) == []
    assert check_alignment([_score(0.29)], SPEECH)[0].verdict is Verdict.TEXT_MISMATCH
