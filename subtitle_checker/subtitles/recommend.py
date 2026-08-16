"""Actionable legibility recommendations.

The legibility grade tells a channel how readable a caption is; this tells them
what to change to raise it. Two calibrated factors carry the signal (see
legibility.py): the WCAG contrast ratio of the text against its background, and
the per-line text height. For every line below the Clear band we name the factor
short of its target and the plain fix - a thicker outline or a box for low
contrast, a larger font or fewer lines for small text. Contrast and size are
what separated guideline clips from low originals on real footage; background
clutter was measured and did not track legibility, so it is not recommended on.

Additive by design: this reads factors already stored on each event and never
changes the grade, the bands, or any existing number.
"""

from __future__ import annotations

from dataclasses import dataclass

from subtitle_checker.artifacts import SubtitleEvent
from subtitle_checker.subtitles.legibility import (
    BAND_CLEAR,
    CONTRAST_RATIO_GOOD,
    LINE_HEIGHT_GOOD,
    line_score,
)


@dataclass
class LineAdvice:
    """One below-Clear line, why it reads poorly, and how to fix it."""

    start: float
    end: float
    text: str
    tips: list[str]
    low_contrast: bool
    small_text: bool


def _contrast_tip(event: SubtitleEvent) -> str | None:
    if event.legibility_ratio is None or event.legibility_ratio >= CONTRAST_RATIO_GOOD:
        return None
    return (
        f"low contrast ({event.legibility_ratio:.1f}:1, aim {CONTRAST_RATIO_GOOD:.0f}:1) "
        "- thicken the dark outline or put the text on a semi-opaque box"
    )


def _size_tip(event: SubtitleEvent) -> str | None:
    if event.line_height_frac is None or event.line_height_frac >= LINE_HEIGHT_GOOD:
        return None
    return (
        f"small text ({event.line_height_frac * 100:.0f}% of frame height, "
        f"aim {LINE_HEIGHT_GOOD * 100:.0f}%) - use a larger font or fewer lines per screen"
    )


def legibility_advice(events: list[SubtitleEvent]) -> list[LineAdvice]:
    """Per-line fixes for every below-Clear line with an actionable factor.

    Only lines that fall below the Clear band are advised - a caption that
    already reads clearly needs no fix - and only when a measured factor is
    actually short of its target, so the list stays specific.
    """
    advice: list[LineAdvice] = []
    for event in events:
        if event.legibility is None or not event.text.strip():
            continue
        if line_score(event.legibility) >= BAND_CLEAR:
            continue
        contrast = _contrast_tip(event)
        size = _size_tip(event)
        tips = [t for t in (contrast, size) if t]
        if tips:
            advice.append(
                LineAdvice(
                    start=event.start,
                    end=event.end,
                    text=event.text,
                    tips=tips,
                    low_contrast=contrast is not None,
                    small_text=size is not None,
                )
            )
    return advice


def advice_summary(advice: list[LineAdvice]) -> str | None:
    """One-line headline of the dominant issue across the advised lines."""
    if not advice:
        return None
    contrast_n = sum(a.low_contrast for a in advice)
    size_n = sum(a.small_text for a in advice)
    parts = []
    if contrast_n:
        parts.append(f"{contrast_n} low-contrast")
    if size_n:
        parts.append(f"{size_n} small-text")
    lead = "contrast" if contrast_n >= size_n else "text size"
    return f"{len(advice)} line(s) below Clear, mainly {lead} ({', '.join(parts)})"
