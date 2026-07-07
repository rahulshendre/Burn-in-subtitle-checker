"""Structural checks — the cheapest, most robust signal, zero ASR.

Cross the subtitle timeline against the audio-region timeline and flag the two
anomalies that need no word recognition at all:

* MISSING_SUBTITLE — the audio has speech that no subtitle covers.
* ORPHAN_SUBTITLE — a subtitle sits over silence, with no speech beneath it.

A subtitle over *music* is neither: sung or scored dialogue can slip past the
VAD, so calling it an orphan would be a false accusation. Those become
UNCHECKABLE — an honest "can't tell here" rather than a wrong flag.

Subtitles that do have speech under them are left untouched; verifying that the
*words* match is Stage 3's job, not this one.
"""

from __future__ import annotations

from subtitle_checker.artifacts import AudioKind, AudioRegion, CheckResult, SubtitleEvent, Verdict

# Shortest stretch of uncovered speech worth flagging. A real dropped line is a
# whole utterance; below this it is a breath, a between-lines pause, or VAD
# jitter. Set from real footage: on a clean serial (Mann Atisunder) every
# sub-2s "gap" was a false alarm at a subtitle boundary, not a missing line.
MIN_UNCOVERED_SPEECH_S = 2.0
# Grace added to each subtitle span before measuring coverage. Consecutive
# subtitles blink off for a fraction of a second while the speech runs on;
# without this every line transition reads as missing speech.
COVER_PAD_S = 0.5
# A subtitle counts as having dialogue if at least this fraction of its span
# overlaps speech. Low on purpose: any real speech under it clears it.
SPEECH_COVER_MIN = 0.3
# With no speech, this much music overlap makes a subtitle UNCHECKABLE rather
# than an orphan — the benefit of the doubt goes to "maybe sung".
MUSIC_COVER_MIN = 0.5


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _kind_overlap(event: SubtitleEvent, regions: list[AudioRegion], kind: AudioKind) -> float:
    return sum(_overlap(event.start, event.end, r.start, r.end) for r in regions if r.kind is kind)


def _merge_missing(
    flags: list[CheckResult], events: list[SubtitleEvent]
) -> list[CheckResult]:
    """Collapse consecutive MISSING flags into one span.

    One unsubtitled stretch of speech gets fragmented when the VAD splits it
    across regions (a breath or a bar of music between clauses) — that should
    read as a single missing line, not several. Two flags are merged when they
    overlap or when nothing but the gap sits between them; a *subtitle* in the
    gap means they are genuinely separate drops and they stay apart.
    """
    if not flags:
        return flags
    ordered = sorted(flags, key=lambda r: r.start)
    merged: list[CheckResult] = [ordered[0]]
    for flag in ordered[1:]:
        prev = merged[-1]
        subtitle_between = any(
            _overlap(prev.end, flag.start, e.start, e.end) > 0 for e in events
        )
        if flag.start <= prev.end or not subtitle_between:
            merged[-1] = CheckResult(
                start=prev.start,
                end=max(prev.end, flag.end),
                verdict=Verdict.MISSING_SUBTITLE,
                reason=prev.reason,
            )
        else:
            merged.append(flag)
    return merged


def _uncovered(
    start: float, end: float, covers: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Sub-spans of [start, end] left uncovered by the covering intervals."""
    gaps: list[tuple[float, float]] = []
    cursor = start
    for cs, ce in sorted(covers):
        cs, ce = max(cs, start), min(ce, end)
        if ce <= cursor:
            continue
        if cs > cursor:
            gaps.append((cursor, cs))
        cursor = ce
    if cursor < end:
        gaps.append((cursor, end))
    return gaps


def check_structural(
    events: list[SubtitleEvent], regions: list[AudioRegion]
) -> list[CheckResult]:
    """Flag speech with no subtitle and subtitles with no speech."""
    results: list[CheckResult] = []

    # ORPHAN / UNCHECKABLE — judged per subtitle event.
    for event in events:
        span = event.end - event.start
        if span <= 0:
            continue
        speech = _kind_overlap(event, regions, AudioKind.SPEECH)
        if speech / span >= SPEECH_COVER_MIN:
            continue  # has dialogue — Stage 3 checks the words
        music = _kind_overlap(event, regions, AudioKind.MUSIC)
        if music / span >= MUSIC_COVER_MIN:
            results.append(
                CheckResult(
                    start=event.start,
                    end=event.end,
                    verdict=Verdict.UNCHECKABLE,
                    reason="subtitle over music — speech not verifiable without separation",
                    subtitle_text=event.text,
                )
            )
        else:
            results.append(
                CheckResult(
                    start=event.start,
                    end=event.end,
                    verdict=Verdict.ORPHAN_SUBTITLE,
                    reason="subtitle present with no speech beneath it",
                    subtitle_text=event.text,
                )
            )

    # MISSING — speech the subtitles never cover. Each subtitle is padded so a
    # brief blink between consecutive lines does not read as a gap in coverage.
    covers = [(e.start - COVER_PAD_S, e.end + COVER_PAD_S) for e in events]
    missing: list[CheckResult] = []
    for region in regions:
        if region.kind is not AudioKind.SPEECH:
            continue
        for gap_start, gap_end in _uncovered(region.start, region.end, covers):
            if gap_end - gap_start >= MIN_UNCOVERED_SPEECH_S:
                missing.append(
                    CheckResult(
                        start=gap_start,
                        end=gap_end,
                        verdict=Verdict.MISSING_SUBTITLE,
                        reason="speech present with no subtitle on screen",
                    )
                )
    results.extend(_merge_missing(missing, events))

    results.sort(key=lambda r: r.start)
    return results
