"""Stage 3 verdicts from forced-alignment scores.

Alignment judges exactly the events structural leaves alone - the ones with
speech under them (see match.structural.event_has_speech). It is a gross-match
signal, not a word-level one: a single wrong word barely moves the frame-
weighted score (measured in evaluation.alignment_eval), so only text essentially
absent from the audio is flagged TEXT_MISMATCH. Subtle word errors are the ASR
cross-check's job when Sarvam access lands.

Precision-first: the layer flags only when it is confident, and abstains
silently otherwise - it never accuses a correct line that merely sits under a
heavy background score, carries poor OCR, or is too short to align reliably. A
subtitle over *music* is still Stage 2's UNCHECKABLE; alignment does not add
per-line noise of its own.

It does emit one positive verdict. When OCR confidence is too low for the
mismatch test but the words align strongly regardless, the line is VERIFIED (OK)
instead of dropped: a wrong or absent line cannot align that well, so a high
score is standalone proof the text was spoken, and abstaining on it only throws
away coverage the OCR-confidence gate never earned (see evaluation.coverage).
"""

from __future__ import annotations

from subtitle_checker.artifacts import AudioRegion, CheckResult, SubtitleEvent, Verdict
from subtitle_checker.match.align import AlignmentScore
from subtitle_checker.match.structural import event_has_speech

# Below this alignment score the words are treated as not spoken here. Set well
# under the correct-speech band (~0.6-0.75 on real clips) with room for
# background-score cases (Mann correct ~0.375): only text all but absent from
# the audio (a wrong or missing line, ~0.29 and below) trips it. Precision over
# recall - see docs and evaluation.alignment_eval for the measurements.
TEXT_MISMATCH_MAX = 0.30
# OCR text below this confidence cannot be blamed on the audio: garbled input
# scores as low as a real mismatch, so the layer abstains instead of accusing.
MIN_OCR_CONF = 0.5
# A confident low score needs a long enough line behind it. Short subtitles (a
# fragment, a one-beat reaction) have too few audio frames to align reliably and
# score low even when correct - on Mann two ~0.6 s lines with clean OCR dropped
# under the cut. Only lines at least this long can trip TEXT_MISMATCH.
MIN_MISMATCH_SPAN = 1.5
# A line the OCR-confidence gate would drop is VERIFIED instead when it aligns at
# least this well: a score this high is standalone proof the words were spoken
# (a wrong line cannot reach it), so the read is confirmed rather than abstained.
# Set at the floor of the confidently-correct band - real correct lines measure
# ~0.6-0.75; a music-heavy clip's borderline reads sit lower and stay abstained,
# which is honest. Positive verification only: it never turns into a flag.
ALIGN_VERIFY_MIN = 0.55


def check_alignment(
    scores: list[AlignmentScore],
    regions: list[AudioRegion],
    text_mismatch_max: float = TEXT_MISMATCH_MAX,
    min_ocr_conf: float = MIN_OCR_CONF,
    min_span: float = MIN_MISMATCH_SPAN,
    align_verify_min: float = ALIGN_VERIFY_MIN,
) -> list[CheckResult]:
    """Judge speech-covered events by how their text aligns to the audio.

    A confident low score flags TEXT_MISMATCH. A confident high score on a line
    the OCR-confidence gate would otherwise drop VERIFIES it (OK). Everything in
    between abstains.
    """
    results: list[CheckResult] = []
    for s in scores:
        event = SubtitleEvent(s.start, s.end, s.text, s.ocr_confidence)
        if not event_has_speech(event, regions):
            continue  # no speech beneath it - structural's call (ORPHAN / UNCHECKABLE)
        if s.score is None:
            continue  # unalignable - UNCHECKABLE elsewhere, never a mismatch
        if s.ocr_confidence < min_ocr_conf:
            # Garbled OCR scores as low as a real mismatch, so a low score here is
            # not evidence - never accuse. But a high score is independent proof
            # the words were spoken, so verify the line despite weak OCR
            # confidence (recovering coverage the conf gate would discard).
            if s.score >= align_verify_min:
                results.append(
                    CheckResult(
                        start=s.start,
                        end=s.end,
                        verdict=Verdict.OK,
                        reason="the subtitle text aligns to the speech beneath it",
                        subtitle_text=s.text,
                        score=s.score,
                    )
                )
            continue
        if s.end - s.start < min_span:
            continue  # too short to align reliably - abstain, never accuse
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
        # else: text matches the speech - left unflagged (ASR ledger owns OK here)
    return results
