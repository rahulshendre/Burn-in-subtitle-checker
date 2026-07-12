"""Coverage metric: verified fraction of the speech-covered lines."""

import pytest

from subtitle_checker.artifacts import (
    AudioKind,
    AudioRegion,
    CheckResult,
    SubtitleEvent,
    Verdict,
)
from subtitle_checker.evaluation.coverage import coverage_score

SPEECH = AudioRegion(0.0, 9.0, AudioKind.SPEECH)
MUSIC = AudioRegion(9.5, 13.0, AudioKind.MUSIC)

# three lines with speech under them, one over music
E_OK = SubtitleEvent(0.0, 2.0, "line one")
E_MISMATCH = SubtitleEvent(3.0, 5.0, "line two")
E_ABSTAIN = SubtitleEvent(6.0, 8.0, "line three")
E_MUSIC = SubtitleEvent(10.0, 12.0, "sung line")
EVENTS = [E_OK, E_MISMATCH, E_ABSTAIN, E_MUSIC]
REGIONS = [SPEECH, MUSIC]


def _claim(event: SubtitleEvent, verdict: Verdict) -> CheckResult:
    return CheckResult(event.start, event.end, verdict, "", subtitle_text=event.text)


def test_coverage_counts_only_judged_speech_lines() -> None:
    results = [_claim(E_OK, Verdict.OK), _claim(E_MISMATCH, Verdict.TEXT_MISMATCH)]
    score = coverage_score(EVENTS, results, REGIONS)
    # the music line is not in the denominator (Stage 2's UNCHECKABLE, not us)
    assert score.speech_events == 3
    assert score.verified == 2
    assert score.abstained == 1
    assert score.coverage == pytest.approx(2 / 3)


def test_uncheckable_row_is_not_coverage() -> None:
    # a row exists for the third line but it is an abstention, not a verdict
    results = [_claim(E_OK, Verdict.OK), _claim(E_ABSTAIN, Verdict.UNCHECKABLE)]
    score = coverage_score(EVENTS, results, REGIONS)
    assert score.verified == 1
    assert score.abstained == 2


def test_no_speech_lines_is_zero_coverage() -> None:
    score = coverage_score([E_MUSIC], [], [MUSIC])
    assert score.speech_events == 0
    assert score.coverage == 0.0
