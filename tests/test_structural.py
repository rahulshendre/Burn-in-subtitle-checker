"""Structural flag logic over hand-built subtitle and region timelines."""

from __future__ import annotations

from subtitle_checker.artifacts import AudioKind, AudioRegion, SubtitleEvent, Verdict
from subtitle_checker.match.structural import check_structural


def _verdicts(results, verdict):
    return [r for r in results if r.verdict is verdict]


def test_subtitle_over_speech_is_not_flagged() -> None:
    events = [SubtitleEvent(1.0, 3.0, "नमस्ते")]
    regions = [
        AudioRegion(0.0, 1.0, AudioKind.SILENCE),
        AudioRegion(1.0, 3.0, AudioKind.SPEECH),
        AudioRegion(3.0, 5.0, AudioKind.SILENCE),
    ]
    assert check_structural(events, regions) == []


def test_subtitle_over_silence_is_orphan() -> None:
    events = [SubtitleEvent(1.0, 3.0, "कोई आवाज़ नहीं")]
    regions = [AudioRegion(0.0, 5.0, AudioKind.SILENCE)]
    orphans = _verdicts(check_structural(events, regions), Verdict.ORPHAN_SUBTITLE)
    assert len(orphans) == 1
    assert orphans[0].subtitle_text == "कोई आवाज़ नहीं"


def test_subtitle_over_music_is_uncheckable_not_orphan() -> None:
    events = [SubtitleEvent(1.0, 3.0, "गाना")]
    regions = [AudioRegion(0.0, 5.0, AudioKind.MUSIC)]
    results = check_structural(events, regions)
    assert _verdicts(results, Verdict.ORPHAN_SUBTITLE) == []
    assert len(_verdicts(results, Verdict.UNCHECKABLE)) == 1


def test_speech_with_no_subtitle_is_missing() -> None:
    events: list[SubtitleEvent] = []
    regions = [AudioRegion(2.0, 4.0, AudioKind.SPEECH)]
    missing = _verdicts(check_structural(events, regions), Verdict.MISSING_SUBTITLE)
    assert len(missing) == 1
    assert (missing[0].start, missing[0].end) == (2.0, 4.0)


def test_speech_fully_covered_by_subtitle_is_not_missing() -> None:
    events = [SubtitleEvent(1.5, 4.5, "ढका हुआ")]
    regions = [AudioRegion(2.0, 4.0, AudioKind.SPEECH)]
    assert _verdicts(check_structural(events, regions), Verdict.MISSING_SUBTITLE) == []


def test_short_uncovered_speech_below_floor_is_ignored() -> None:
    # only 0.3 s of speech pokes out past the subtitle — under the floor
    events = [SubtitleEvent(2.0, 3.7, "लगभग ढका")]
    regions = [AudioRegion(2.0, 4.0, AudioKind.SPEECH)]
    assert _verdicts(check_structural(events, regions), Verdict.MISSING_SUBTITLE) == []


def test_partial_speech_overlap_clears_subtitle() -> None:
    # subtitle 1-4, speech only 3-4: 1 s / 3 s = 0.33 >= SPEECH_COVER_MIN
    events = [SubtitleEvent(1.0, 4.0, "थोड़ी बात")]
    regions = [AudioRegion(0.0, 3.0, AudioKind.SILENCE), AudioRegion(3.0, 4.0, AudioKind.SPEECH)]
    assert check_structural(events, regions) == []


def test_missing_flag_spans_only_the_uncovered_gap() -> None:
    # speech 2-8, subtitle covers 2-5 (+0.5s pad), so 5.5-8 is the missing stretch
    events = [SubtitleEvent(2.0, 5.0, "पहला भाग")]
    regions = [AudioRegion(2.0, 8.0, AudioKind.SPEECH)]
    missing = _verdicts(check_structural(events, regions), Verdict.MISSING_SUBTITLE)
    assert len(missing) == 1
    assert (missing[0].start, missing[0].end) == (5.5, 8.0)
