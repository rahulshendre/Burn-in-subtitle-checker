"""OCR engines for subtitle bands.

EasyOCR is the default (best results on scene-overlaid Devanagari in our
tests, and it runs on-device); the Protocol keeps the engine swappable without
touching the rest of the pipeline. SarvamVisionOcr is an opt-in quality engine
that reads hard Devanagari bands more accurately at the cost of a cloud call.
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


# Sarvam Vision returns a per-block confidence, but on our subtitle bands it is
# flat (~0.33-0.49 whether the read is clean or hard - it does not track
# legibility), so it cannot gate trust or feed a legibility score. The reads
# themselves are reliable, so a fixed trusted confidence is returned instead:
# high enough to clear the OCR-trust gates in Stage 3. A legibility signal must
# come from a different measurement, not this value.
SARVAM_VISION_TRUSTED_CONF = 0.99


def _vision_text(blocks: list[str]) -> tuple[str, float]:
    """Join Sarvam Vision text blocks into one line, dropping chrome/logo boxes.

    Sarvam reads the whole crop, so a channel logo that survived Stage-1
    chrome subtraction (`DD Free Dish`, `TATA PLAY`) comes back as its own
    block; the same Devanagari content test used for EasyOCR drops it. A logo
    merged into a real line's block is lost with it - the EasyOCR limit too.
    """
    lines = [b for b in blocks if _is_devanagari_line(b)]
    if not lines:
        return "", 0.0
    return " ".join(lines), SARVAM_VISION_TRUSTED_CONF


class SarvamVisionOcr:
    """Sarvam Vision document-intelligence OCR - an opt-in quality engine.

    Reads Devanagari off hard scene-overlaid bands more accurately than EasyOCR
    (it fixes garbles EasyOCR makes on ornate or bright backgrounds), at the
    cost of one cloud job per band. Reads SARVAM_API_KEY from the environment
    and never stores it. The heavy SDK import and client build wait until first
    use, so importing this module stays cheap and offline.
    """

    def __init__(self, lang: str = "hi-IN") -> None:
        self._lang = lang
        self._client = None

    def read(self, band: np.ndarray) -> tuple[str, float]:
        if self._client is None:
            import os

            from sarvamai import SarvamAI

            key = os.environ.get("SARVAM_API_KEY")
            if not key:
                raise RuntimeError("SARVAM_API_KEY not set in the environment")
            self._client = SarvamAI(api_subscription_key=key)
        return _vision_text(_sarvam_vision_blocks(self._client, band, self._lang))


def _sarvam_vision_blocks(client, band: np.ndarray, lang: str) -> list[str]:
    """Run one document-intelligence job on a band image -> its text blocks.

    The band is written to a temporary PNG (the job takes a file), OCR'd, and
    the result ZIP's per-page JSON is read back in reading order.
    """
    import json
    import tempfile
    import zipfile
    from pathlib import Path

    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        png = tmp / "band.png"
        Image.fromarray(band).save(png)
        job = client.document_intelligence.create_job(language=lang, output_format="md")
        job.upload_file(str(png))
        job.start()
        job.wait_until_complete(poll_interval=1.5, timeout=120)
        zpath = tmp / "out.zip"
        job.download_output(str(zpath))
        extracted = tmp / "unzipped"
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(extracted)
        json_dir = extracted / "metadata" if (extracted / "metadata").exists() else extracted
        texts: list[str] = []
        for jf in sorted(json_dir.glob("*.json")):
            data = json.loads(jf.read_text(encoding="utf-8"))
            for block in sorted(data.get("blocks", []), key=lambda b: b.get("reading_order", 0)):
                text = block.get("text", "").strip()
                if text:
                    texts.append(text)
    return texts
