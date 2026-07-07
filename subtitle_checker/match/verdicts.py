"""Stage 3 verdicts from forced-alignment scores.

Alignment judges exactly the events structural leaves alone — the ones with
speech under them (see match.structural.event_has_speech). It is a gross-match
and presence signal, not a word-level one: a single wrong word barely moves the
frame-weighted score (measured in evaluation.alignment_eval), so only text
essentially absent from the audio is flagged TEXT_MISMATCH. Subtle word errors
are the ASR cross-check's job when Sarvam access lands.

Precision-first throughout: the layer would rather stay silent than flag a
correct line that merely sits under a heavy background score, and it abstains
(UNCHECKABLE) when the OCR text is too poor to blame the audio.
"""

from __future__ import annotations

from subtitle_checker.artifacts import AudioRegion, CheckResult, SubtitleEvent, Verdict
from subtitle_checker.match.align import AlignmentScore
from subtitle_checker.match.structural import event_has_speech

# Below this alignment score the words are treated as not spoken here. Set well
# under the correct-speech band (~0.6-0.75 on real clips) with room for
# background-score cases (Mann correct ~0.375): only text all but absent from
# the audio (a wrong or missing line, ~0.29 and below) trips it. Precision over
# recall — see docs and evaluation.alignment_eval for the measurements.
TEXT_MISMATCH_MAX = 0.30
# OCR text below this confidence cannot be blamed on the audio: garbled input
# scores as low as a real mismatch, so the layer abstains instead of accusing.
MIN_OCR_CONF = 0.5


def check_alignment(
    scores: list[AlignmentScore],
    regions: list[AudioRegion],
    text_mismatch_max: float = TEXT_MISMATCH_MAX,
    min_ocr_conf: float = MIN_OCR_CONF,
) -> list[CheckResult]:
    """Flag speech-covered subtitle events whose text does not match the audio."""
    results: list[CheckResult] = []
    for s in scores:
        event = SubtitleEvent(s.start, s.end, s.text, s.ocr_confidence)
        if not event_has_speech(event, regions):
            continue  # no speech beneath it — structural's call (ORPHAN / UNCHECKABLE)
        if s.score is None:
            results.append(_uncheckable(s, "subtitle text too short or empty to align"))
        elif s.ocr_confidence < min_ocr_conf:
            results.append(_uncheckable(s, "OCR text too unreliable to verify against audio"))
        elif s.score < text_mismatch_max:
            results.append(
                CheckResult(
                    start=s.start,
                    end=s.end,
                    verdict=Verdict.TEXT_MISMATCH,
                    reason="subtitle text does not match the speech beneath it",
                    subtitle_text=s.text,
                    score=s.score,
                )
            )
        # else: text matches the speech — left unflagged (precision-first, no OK noise)
    return results


def _uncheckable(s: AlignmentScore, reason: str) -> CheckResult:
    return CheckResult(
        start=s.start,
        end=s.end,
        verdict=Verdict.UNCHECKABLE,
        reason=reason,
        subtitle_text=s.text,
        score=s.score,
    )
