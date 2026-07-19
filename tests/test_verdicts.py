"""Alignment verdict logic over hand-built scores and region timelines."""

from __future__ import annotations

from subtitle_checker.artifacts import AudioKind, AudioRegion, Verdict
from subtitle_checker.match.align import AlignmentScore
from subtitle_checker.match.verdicts import check_alignment

SPEECH = [AudioRegion(0.0, 6.0, AudioKind.SPEECH)]
MUSIC = [AudioRegion(0.0, 6.0, AudioKind.MUSIC)]


def _score(score, ocr_confidence=0.9, start=1.0, end=3.0, text="यह बात है") -> AlignmentScore:
    return AlignmentScore(start, end, text, ocr_confidence, score, start, end)


def test_low_score_over_speech_is_text_mismatch() -> None:
    results = check_alignment([_score(0.1)], SPEECH)
    assert len(results) == 1
    assert results[0].verdict is Verdict.TEXT_MISMATCH
    assert results[0].subtitle_text == "यह बात है"
    assert results[0].score == 0.1


def test_good_score_over_speech_is_verified() -> None:
    # a strong alignment verifies the line (OK), whatever the OCR confidence
    results = check_alignment([_score(0.7)], SPEECH)
    assert len(results) == 1
    assert results[0].verdict is Verdict.OK
    assert results[0].score == 0.7


def test_event_without_speech_is_left_to_structural() -> None:
    # low score, but no speech beneath it → alignment abstains, structural owns it
    assert check_alignment([_score(0.1)], MUSIC) == []


def test_low_ocr_confidence_abstains() -> None:
    # garbled OCR scores low too, so a low score here is not evidence - no flag
    assert check_alignment([_score(0.1, ocr_confidence=0.2)], SPEECH) == []


def test_unalignable_none_score_abstains() -> None:
    assert check_alignment([_score(None)], SPEECH) == []


def test_short_line_low_score_abstains() -> None:
    # a 0.6 s line aligns unreliably low even when correct - must not be flagged
    assert check_alignment([_score(0.1, start=1.0, end=1.6)], SPEECH) == []


def test_threshold_is_precision_first() -> None:
    # just above the cut clears; just below trips it (on a long-enough line)
    assert check_alignment([_score(0.31)], SPEECH) == []
    assert check_alignment([_score(0.29)], SPEECH)[0].verdict is Verdict.TEXT_MISMATCH


def test_low_ocr_confidence_high_alignment_is_verified() -> None:
    # unreliable OCR conf, but the words align strongly → verify, don't abstain
    results = check_alignment([_score(0.6, ocr_confidence=0.2)], SPEECH)
    assert len(results) == 1
    assert results[0].verdict is Verdict.OK


def test_low_ocr_confidence_moderate_alignment_still_abstains() -> None:
    # below the verify floor: not proof enough to confirm, so still no verdict
    assert check_alignment([_score(0.5, ocr_confidence=0.2)], SPEECH) == []


def test_rescue_does_not_require_min_span() -> None:
    # a short line the mismatch test would skip is still verifiable on a high score
    results = check_alignment(
        [_score(0.6, ocr_confidence=0.2, start=1.0, end=1.4)], SPEECH
    )
    assert results[0].verdict is Verdict.OK


def test_trusted_line_high_alignment_is_verified() -> None:
    # a good score with trustworthy OCR is now verified locally (the merge dedups
    # it against the ASR ledger's OK, so it is not double-counted downstream)
    results = check_alignment([_score(0.6, ocr_confidence=0.9)], SPEECH)
    assert len(results) == 1
    assert results[0].verdict is Verdict.OK


def test_trusted_mid_score_still_abstains() -> None:
    # between the mismatch cut and the verify floor: not flag, not proof - abstain
    assert check_alignment([_score(0.45, ocr_confidence=0.9)], SPEECH) == []
