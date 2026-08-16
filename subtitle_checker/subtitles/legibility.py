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

from dataclasses import dataclass, field

import numpy as np

from subtitle_checker.artifacts import SubtitleEvent
from subtitle_checker.subtitles.masks import binarize, text_mask

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

# How many of the least legible lines to surface for a channel to inspect. Only
# lines that fall below the Clear band count - a caption that already reads
# clearly is not a problem to list, even on a short video with few lines.
DEFAULT_WORST_N = 5

# Readability bands for the time breakdown a channel asked for: how many minutes
# of the video read easily, sit borderline, or wash out. The two cutoffs tier the
# per-line scores, and the report colours the whole-video banner off the same two
# numbers, so the legibility read speaks one scale end to end.
BAND_POOR = 50.0  # below this a line is hard to read
BAND_CLEAR = 80.0  # at or above this a line reads easily; between the two = mixed


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


# Diagnostic factors behind the score - not part of the grade, but what a channel
# needs to know *why* a line reads poorly and how to fix it. Two factors carry
# the signal (calibrated on real Hindi footage: guideline clips vs low originals);
# background clutter was measured too and did not track legibility, so it is left
# out. The Michelson score above stays the grade; these only drive recommendations.
#
# WCAG relative-luminance contrast ratio, stroke pixels vs background. Broadcast
# (BBC) asks 5:1; below ~3:1 a white line over a bright scene is genuinely hard.
# Measured: guideline clips ~5.6-5.8:1, low originals ~2.7-3.2:1.
CONTRAST_RATIO_GOOD = 5.0
# Per-line text height as a fraction of frame height. The broadcast rule of thumb
# is ~8% line height (16:9); measured guideline clips run 6-7%, cramped multi-line
# originals ~4%. Below the target the text is small enough to slow a reader.
LINE_HEIGHT_GOOD = 0.06

# sRGB values at or below this are linearised by a plain divide, not the power
# curve (the WCAG piecewise transfer function).
_SRGB_LINEAR_CUTOFF = 0.03928


def legibility_ratio(crop: np.ndarray) -> float | None:
    """WCAG relative-luminance contrast ratio of the text strokes vs background.

    The bright text pixels (the binarised stroke mask) against the rest of the
    crop, as the standard ``(L_light + 0.05) / (L_dark + 0.05)`` on linearised
    luminance. Grayscale stands in for luminance - the band is sampled gray, and
    gray is itself a luma projection. Returns None when the crop has no clear
    stroke-and-background split to compare (all bright or all dark).
    """
    if crop.size == 0:
        return None
    stroke = binarize(crop)
    if stroke.sum() < 10 or (~stroke).sum() < 10:
        return None
    gray = crop.astype(np.float64) / 255.0
    lin = np.where(
        gray <= _SRGB_LINEAR_CUTOFF, gray / 12.92, ((gray + 0.055) / 1.055) ** 2.4
    )
    text = float(lin[stroke].mean())
    background = float(lin[~stroke].mean())
    hi, lo = max(text, background), min(text, background)
    return (hi + 0.05) / (lo + 0.05)


def _text_block_px(crop: np.ndarray) -> int:
    """Native-pixel height of the text block in the crop (stroke rows extent)."""
    mask = text_mask(crop)
    rows = np.where(mask.any(axis=1))[0]
    return int(rows[-1] - rows[0] + 1) if rows.size else 0


def line_height_frac(crop: np.ndarray, line_count: int, frame_h: float) -> float | None:
    """Per-line text height as a fraction of the frame height.

    The text block spans ``line_count`` stacked lines, so its height is divided
    back out to a single line, then taken relative to the full frame height - a
    resolution- and aspect-independent measure of how large the caption renders.
    Returns None when nothing was measurable.
    """
    px = _text_block_px(crop)
    if px == 0 or frame_h <= 0:
        return None
    return (px / max(line_count, 1)) / frame_h


@dataclass
class LineLegibility:
    """One line's legibility: where it is, what it reads, how legible it is."""

    start: float
    end: float
    text: str
    contrast: float
    score: float


@dataclass
class LegibilityBand:
    """One readability tier and how much subtitle time falls in it."""

    label: str  # Clear / Mixed / Poor
    low: float  # inclusive score floor of the band (0 for Poor)
    high: float  # score ceiling of the band (100 for Clear)
    seconds: float  # total subtitle time in this band
    share: float  # fraction of measured subtitle time, 0..1
    line_count: int


@dataclass
class VideoLegibility:
    """A whole-video legibility grade with the least legible lines called out."""

    score: float
    line_count: int
    worst: list[LineLegibility]
    bands: list[LegibilityBand] = field(default_factory=list)


def video_legibility(
    events: list[SubtitleEvent], worst_n: int = DEFAULT_WORST_N
) -> VideoLegibility | None:
    """Grade a whole video's subtitle legibility from its measured events.

    Reads the per-line ``legibility`` contrast set at detection time; events with
    no measured contrast, or with no subtitle text OCR could read, are skipped -
    contrast is a pixel measurement, so a band with no readable caption would
    otherwise score the scene behind it, not a subtitle. The grade is the
    duration-weighted mean of the per-line scores - a long low-contrast caption
    hurts a viewer more than a brief flash - and ``worst`` lists the least
    legible lines that fall below the Clear band, so a channel only sees captions
    worth inspecting (empty when every line reads clearly). Returns None when no
    line could be measured.
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
        if e.legibility is not None and e.text.strip()
    ]
    if not lines:
        return None

    weights = [max(line.end - line.start, 1e-6) for line in lines]
    grade = sum(w * line.score for w, line in zip(weights, lines)) / sum(weights)
    below_clear = [
        line for line in sorted(lines, key=lambda line: line.score)
        if line.score < BAND_CLEAR
    ]
    worst = below_clear[:worst_n]
    return VideoLegibility(
        score=round(grade, 1),
        line_count=len(lines),
        worst=worst,
        bands=_legibility_bands(lines),
    )


def _band_label(score: float) -> str:
    """Which readability band a per-line score falls in."""
    if score >= BAND_CLEAR:
        return "Clear"
    if score >= BAND_POOR:
        return "Mixed"
    return "Poor"


def _legibility_bands(lines: list[LineLegibility]) -> list[LegibilityBand]:
    """Total subtitle time in each readability band, best to worst.

    Each line's on-screen duration is added to its band, so the result answers
    the channel's question directly: how many minutes of the video read clearly,
    are borderline, or wash out. All three bands are always returned, at zero
    seconds when empty, so the report shows a complete picture.
    """
    order = [
        ("Clear", BAND_CLEAR, 100.0),
        ("Mixed", BAND_POOR, BAND_CLEAR),
        ("Poor", 0.0, BAND_POOR),
    ]
    seconds = {label: 0.0 for label, _, _ in order}
    counts = {label: 0 for label, _, _ in order}
    total = 0.0
    for line in lines:
        duration = max(line.end - line.start, 0.0)
        total += duration
        label = _band_label(line.score)
        seconds[label] += duration
        counts[label] += 1
    return [
        LegibilityBand(
            label=label,
            low=low,
            high=high,
            seconds=round(seconds[label], 1),
            share=(seconds[label] / total if total > 0 else 0.0),
            line_count=counts[label],
        )
        for label, low, high in order
    ]
