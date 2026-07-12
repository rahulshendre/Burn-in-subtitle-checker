"""Coverage: how many checkable lines the pipeline actually judges.

The defect scorer (score.py) measures whether planted errors are caught. It
says nothing about the lines the pipeline quietly declines: a correct line the
OCR-confidence gate abstains on costs nothing there, because a correct line is
not a planted defect. But to an editor those abstentions are the difference
between a tool that verifies the episode and one that shrugs at half of it, so
they need a metric of their own.

Coverage is that axis. The denominator is the events with speech under them
(event_has_speech) - the lines Stage 3 is responsible for judging; a subtitle
over music or silence is Stage 2's UNCHECKABLE, not an abstention. The
numerator is those that received a word-level verdict (OK or TEXT_MISMATCH).
The gap between them is over-abstention. Read coverage next to score.py's
precision: precision says "don't be wrong", coverage says "don't shrug".
"""

from __future__ import annotations

from dataclasses import dataclass

from subtitle_checker.artifacts import AudioRegion, CheckResult, SubtitleEvent, Verdict
from subtitle_checker.match.structural import event_has_speech

# A word-level judgement on a specific line, as opposed to UNCHECKABLE or the
# gap-based MISSING/ORPHAN verdicts that do not claim an event's span.
JUDGED_VERDICTS = frozenset({Verdict.OK, Verdict.TEXT_MISMATCH})
# Alignment and ASR results copy their event's span exactly, so a tight claim
# test is enough to tie a verdict back to the event it judged.
CLAIM_TOL_S = 0.05


@dataclass
class CoverageScore:
    """How many of the lines Stage 3 must judge actually got a verdict."""

    speech_events: int  # denominator: events with speech under them
    verified: int  # of those, how many got a word-level verdict

    @property
    def abstained(self) -> int:
        return self.speech_events - self.verified

    @property
    def coverage(self) -> float:
        return self.verified / self.speech_events if self.speech_events else 0.0


def _claims(result: CheckResult, event: SubtitleEvent, tol: float) -> bool:
    return abs(result.start - event.start) < tol and abs(result.end - event.end) < tol


def coverage_score(
    events: list[SubtitleEvent],
    results: list[CheckResult],
    regions: list[AudioRegion],
    tol: float = CLAIM_TOL_S,
) -> CoverageScore:
    """Fraction of speech-covered events that received a word-level verdict."""
    judged = [r for r in results if r.verdict in JUDGED_VERDICTS]
    speech = [e for e in events if event_has_speech(e, regions)]
    verified = sum(1 for e in speech if any(_claims(r, e, tol) for r in judged))
    return CoverageScore(speech_events=len(speech), verified=verified)
