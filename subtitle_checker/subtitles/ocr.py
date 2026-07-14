"""OCR engines for subtitle bands.

EasyOCR is the default (best results on scene-overlaid Devanagari in our
tests); the Protocol keeps the engine swappable without touching the rest of
the pipeline.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class OcrEngine(Protocol):
    def read(self, band: np.ndarray) -> tuple[str, float]:
        """Grayscale band image → (text, confidence 0..1)."""
        ...


# EasyOCR returns one box per text region. Bright chrome that survives into a
# crop - an animated channel logo, sequin sparkle - comes back as its own
# boxes, and they read as punctuation, Latin, or digits (`"^7`, `१/ /`,
# `177374`, `"डद"`). A real subtitle box is dominated by Devanagari letters, so
# a per-box content test drops the junk without disturbing the actual line.
_DEVANAGARI_LANGS = {"hi", "mr", "ne", "sa"}
_BLOCK_START, _BLOCK_END = "ऀ", "ॿ"  # Devanagari block
_DIGIT_START, _DIGIT_END = "०", "९"  # ०-९, in the block but not letters
_MIN_LETTERS = 2
_MIN_FRACTION = 0.6


def _is_devanagari_line(text: str) -> bool:
    """True when a recognised box is real Devanagari text, not chrome/logo junk.

    Requires at least two Devanagari letters (a stray glyph is not a line) and
    that Devanagari makes up most of the box (so a couple of letters wrapped in
    quotes and slashes - a misread logo - does not pass).
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    in_block = [c for c in chars if _BLOCK_START <= c <= _BLOCK_END]
    letters = [c for c in in_block if not (_DIGIT_START <= c <= _DIGIT_END)]
    return len(letters) >= _MIN_LETTERS and len(in_block) / len(chars) >= _MIN_FRACTION


class EasyOcrEngine:
    """Wraps easyocr; the heavy import and model load wait until first use.

    For Devanagari languages the chrome/logo junk boxes are filtered out of the
    result (see _is_devanagari_line); other scripts are read unfiltered.
    """

    def __init__(self, langs: list[str] | None = None) -> None:
        self._langs = langs or ["hi"]
        self._reader = None
        self._drop_junk = bool(set(self._langs) & _DEVANAGARI_LANGS)

    def read(self, band: np.ndarray) -> tuple[str, float]:
        if self._reader is None:
            import easyocr  # pulls in torch - keep it off module import

            self._reader = easyocr.Reader(self._langs, verbose=False)
        results = self._reader.readtext(band, detail=1, paragraph=False)
        if self._drop_junk:
            results = [r for r in results if _is_devanagari_line(r[1])]
        if not results:
            return "", 0.0
        # reading order: top-to-bottom, then left-to-right
        results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
        text = " ".join(r[1] for r in results)
        confidence = float(np.mean([float(r[2]) for r in results]))
        return text, confidence
