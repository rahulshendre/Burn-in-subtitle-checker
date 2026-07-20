"""Fuse the per-line signals into one confidence score (the score-fusion logic).

An editor sees two confidences per line: how cleanly the subtitle text was read
(OCR) and how well the spoken audio matches it (the ASR cross-check, or forced
alignment when the ASR did not run). Management asked for both folded into a
single 0-100 number. The audio match is the stronger evidence that a line is
correct, so it carries the most weight; OCR confidence tempers it, because a
subtitle we could not read cleanly is judged with less certainty.

A line with no audio signal beneath it - over music or silence - has nothing to
match against, so it gets no combined score (that is Stage 2's UNCHECKABLE, not
a low number an editor would misread as a problem). Whole-video OCR legibility
is a separate measure, not this.
"""

from __future__ import annotations

# The audio match is the primary evidence of correctness; OCR confidence is a
# secondary temper on how sure we are of the text we matched. The weights sum to
# one so the result stays a clean 0-100. They are tunable on purpose - this blend
# is the logic, not a fixed model output.
W_MATCH = 0.7
W_OCR = 0.3


def combined_score(
    ocr_confidence: float | None,
    match_confidence: float | None,
    *,
    w_match: float = W_MATCH,
    w_ocr: float = W_OCR,
) -> float | None:
    """Blend OCR read-reliability and audio agreement into a 0-100 confidence.

    Both inputs are 0..1. ``match_confidence`` is the audio-vs-text agreement
    (the ASR token_set_ratio, or the alignment score when the ASR did not run).
    None means no audio signal beneath the line, so correctness cannot be scored
    - the function returns None rather than a low number that reads as a problem.
    A missing OCR confidence counts as zero: an unread subtitle earns no credit.
    """
    if match_confidence is None:
        return None
    ocr = 0.0 if ocr_confidence is None else ocr_confidence
    return round(100 * (w_match * match_confidence + w_ocr * ocr), 1)
