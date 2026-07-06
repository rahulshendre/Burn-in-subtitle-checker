"""Closed-loop structural eval: plant drop/extra, score the flags."""

from __future__ import annotations

from subtitle_checker.artifacts import AudioKind, SubtitleEvent, Verdict
from subtitle_checker.evaluation.defects import DefectType
from subtitle_checker.evaluation.structural_eval import (
    evaluate_structural,
    regions_from_truth,
)


def make_truth(n: int = 6, line_s: float = 3.0, gap_s: float = 2.5) -> list[SubtitleEvent]:
    events = []
    t = 2.0
    for i in range(n):
        events.append(SubtitleEvent(start=t, end=t + line_s, text=f"पंक्ति {i}"))
        t += line_s + gap_s
    return events


def test_regions_alternate_speech_and_silence_over_truth() -> None:
    regions = regions_from_truth(make_truth(n=2))
    kinds = [r.kind for r in regions]
    assert AudioKind.SPEECH in kinds and AudioKind.SILENCE in kinds
    # every truth line has a speech region exactly on its span
    speech = [(r.start, r.end) for r in regions if r.kind is AudioKind.SPEECH]
    assert (2.0, 5.0) in speech


def test_drop_and_extra_are_both_caught_with_right_verdicts() -> None:
    score, results = evaluate_structural(make_truth(), seed=3)
    assert score.recall == 1.0
    assert score.precision == 1.0
    assert score.by_type[DefectType.DROP_LINE.value].verdict_correct == 1
    assert score.by_type[DefectType.EXTRA_LINE.value].verdict_correct == 1
    verdicts = {r.verdict for r in results}
    assert Verdict.MISSING_SUBTITLE in verdicts
    assert Verdict.ORPHAN_SUBTITLE in verdicts


def test_no_stray_flags_on_the_untouched_lines() -> None:
    _, results = evaluate_structural(make_truth(n=8), seed=1)
    # exactly two flags: one missing (dropped line), one orphan (extra line)
    assert len(results) == 2
