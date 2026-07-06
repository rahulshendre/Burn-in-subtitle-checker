"""Closed-loop evaluation of the Stage 2 structural flags.

Build a ground-truth audio-region timeline straight from verified truth
subtitle lines — each line is speech, the gaps between are silence — then plant
the two structural defects (a dropped line, an extra line) and confirm the
structural checker flags them, scored automatically against the labels.

The regions are synthetic on purpose: this measures the *structural logic*
against exact ground truth, the same way the Stage 1 eval burns known text and
detects it back. How faithfully the real VAD recovers regions from messy
broadcast audio is a separate question, checked by running the VAD on real
clips and reading the flags (see docs/).
"""

from __future__ import annotations

from subtitle_checker.artifacts import AudioKind, AudioRegion, CheckResult, SubtitleEvent
from subtitle_checker.evaluation.defects import DefectType, plan_defects
from subtitle_checker.evaluation.score import EvalScore, score_results
from subtitle_checker.match.structural import check_structural

# The defects Stage 2 is responsible for; word swaps and timing drift are Stage 3.
STRUCTURAL_DEFECTS = [DefectType.DROP_LINE, DefectType.EXTRA_LINE]


def regions_from_truth(events: list[SubtitleEvent]) -> list[AudioRegion]:
    """Speech under each truth line, silence in the gaps — the clean timeline."""
    ordered = sorted(events, key=lambda e: e.start)
    regions: list[AudioRegion] = []
    cursor = 0.0
    for e in ordered:
        if e.start > cursor:
            regions.append(AudioRegion(cursor, e.start, AudioKind.SILENCE))
        regions.append(AudioRegion(e.start, e.end, AudioKind.SPEECH))
        cursor = e.end
    return regions


def evaluate_structural(
    truth: list[SubtitleEvent], seed: int = 0
) -> tuple[EvalScore, list[CheckResult]]:
    """Plant drop/extra defects on truth, run structural checks, and score them.

    Regions come from the *truth* timeline — the audio never changes, so a
    dropped subtitle still has speech under it (→ MISSING) and an extra line
    still sits in silence (→ ORPHAN).
    """
    regions = regions_from_truth(truth)
    mutated, defects = plan_defects(truth, seed=seed, types=STRUCTURAL_DEFECTS)
    results = check_structural(mutated, regions)
    return score_results(results, defects), results
