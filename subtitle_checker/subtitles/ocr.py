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


class EasyOcrEngine:
    """Wraps easyocr; the heavy import and model load wait until first use."""

    def __init__(self, langs: list[str] | None = None) -> None:
        self._langs = langs or ["hi"]
        self._reader = None

    def read(self, band: np.ndarray) -> tuple[str, float]:
        if self._reader is None:
            import easyocr  # pulls in torch — keep it off module import

            self._reader = easyocr.Reader(self._langs, verbose=False)
        results = self._reader.readtext(band, detail=1, paragraph=False)
        if not results:
            return "", 0.0
        # reading order: top-to-bottom, then left-to-right
        results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
        text = " ".join(r[1] for r in results)
        confidence = float(np.mean([float(r[2]) for r in results]))
        return text, confidence
