"""Stage 3 verdicts from forced-alignment scores.

Alignment judges exactly the events structural leaves alone — the ones with
speech under them (see match.structural.event_has_speech). It is a gross-match
signal, not a word-level one: a single wrong word barely moves the frame-
weighted score (measured in evaluation.alignment_eval), so only text essentially
absent from the audio is flagged TEXT_MISMATCH. Subtle word errors are the ASR
cross-check's job when Sarvam access lands.

Precision-first: the layer emits a flag only when it is confident, and abstains
silently otherwise — it never accuses a correct line that merely sits under a
heavy background score, carries poor OCR, or is too short to align reliably. A
subtitle over *music* is still Stage 2's UNCHECKABLE; alignment does not add
per-line noise of its own.
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
# A confident low score needs a long enough line behind it. Short subtitles (a
# fragment, a one-beat reaction) have too few audio frames to align reliably and
# score low even when correct — on Mann two ~0.6 s lines with clean OCR dropped
# under the cut. Only lines at least this long can trip TEXT_MISMATCH.
MIN_MISMATCH_SPAN = 1.5


def check_alignment(
    scores: list[AlignmentScore],
    regions: list[AudioRegion],
    text_mismatch_max: float = TEXT_MISMATCH_MAX,
    min_ocr_conf: float = MIN_OCR_CONF,
    min_span: float = MIN_MISMATCH_SPAN,
) -> list[CheckResult]:
    """Flag speech-covered subtitle events whose text does not match the audio."""
    results: list[CheckResult] = []
    for s in scores:
        event = SubtitleEvent(s.start, s.end, s.text, s.ocr_confidence)
        if not event_has_speech(event, regions):
            continue  # no speech beneath it — structural's call (ORPHAN / UNCHECKABLE)
        if s.score is None or s.ocr_confidence < min_ocr_conf or s.end - s.start < min_span:
            continue  # cannot trust a low score here — abstain, never accuse
        if s.score < text_mismatch_max:
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
