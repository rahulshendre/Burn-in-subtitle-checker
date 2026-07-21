"""Whole-video subtitle legibility scoring.

Legibility is presentation quality, not correctness: can a viewer actually read
the subtitle off the screen? The proxy is optical contrast between the text
strokes and the background they sit on. White text over a dark scene reads
easily; white text over a bright wall or a washed-out, light background loses
its edges. The same optical fact drives OCR - strokes that barely separate from
the background defeat a reader and an OCR engine alike - so contrast is a
model-agnostic legibility signal we read straight off the frame, with no model
and no network call.

Per line we take the Michelson contrast between the bright text strokes and the
local background in the subtitle crop, over robust percentiles so a stray pixel
does not swing it. A line only scores low when the crop holds no dark anchor at
all - the white-on-bright case - which is exactly the failure a channel cares
about. The whole-video score is the duration-weighted mean of the per-line
scores, with the least legible lines surfaced so a channel sees which captions
fail. This is a standalone, channel-facing measure - "are your subtitles
legible, and which ones are not" - separate from the audio mismatch check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from subtitle_checker.artifacts import SubtitleEvent

# Percentiles standing in for the text strokes (high) and the local background
# (low). Robust ends, not min/max, so one speck of glare or shadow cannot swing
# the contrast.
HI_PCT = 95.0
LO_PCT = 5.0

# The Michelson contrast that maps to a legibility of 0 and of 100. Below the
# floor the text and its background are the same brightness - washed out; at the
# ceiling there is a full dark-to-bright range behind the strokes - clearly
# readable. These are the calibrated ends of the legible band, tunable on
# purpose: this mapping is the logic, not a fixed model output.
CONTRAST_FLOOR = 0.15
CONTRAST_CEIL = 0.65

# How many of the least legible lines to surface for a channel to inspect.
DEFAULT_WORST_N = 5


def contrast(crop: np.ndarray) -> float:
    """Michelson contrast of a grayscale subtitle crop, 0..1.

    ``(hi - lo) / (hi + lo)`` over the bright and dark percentiles. High when the
    text strokes stand well clear of a darker background; near zero when the
    whole crop sits at one brightness (white text on a bright background).
    """
    if crop.size == 0:
        return 0.0
    hi = float(np.percentile(crop, HI_PCT))
    lo = float(np.percentile(crop, LO_PCT))
    total = hi + lo
    if total <= 0:
        return 0.0
    return (hi - lo) / total


def line_score(c: float) -> float:
    """Map a line's contrast to a 0-100 legibility score (clamped, linear)."""
    span = CONTRAST_CEIL - CONTRAST_FLOOR
    frac = (c - CONTRAST_FLOOR) / span
    return round(100 * min(1.0, max(0.0, frac)), 1)


@dataclass
class LineLegibility:
    """One line's legibility: where it is, what it reads, how legible it is."""

    start: float
    end: float
    text: str
    contrast: float
    score: float


@dataclass
class VideoLegibility:
    """A whole-video legibility grade with the least legible lines called out."""

    score: float
    line_count: int
    worst: list[LineLegibility]


def video_legibility(
    events: list[SubtitleEvent], worst_n: int = DEFAULT_WORST_N
) -> VideoLegibility | None:
    """Grade a whole video's subtitle legibility from its measured events.

    Reads the per-line ``legibility`` contrast set at detection time; events with
    no measured contrast (nothing to read) are skipped. The grade is the
    duration-weighted mean of the per-line scores - a long low-contrast caption
    hurts a viewer more than a brief flash - and ``worst`` lists the lowest
    scorers for inspection. Returns None when no line could be measured.
    """
    lines = [
        LineLegibility(
            start=e.start,
            end=e.end,
            text=e.text,
            contrast=e.legibility,
            score=line_score(e.legibility),
        )
        for e in events
        if e.legibility is not None
    ]
    if not lines:
        return None

    weights = [max(line.end - line.start, 1e-6) for line in lines]
    grade = sum(w * line.score for w, line in zip(weights, lines)) / sum(weights)
    worst = sorted(lines, key=lambda line: line.score)[:worst_n]
    return VideoLegibility(score=round(grade, 1), line_count=len(lines), worst=worst)
