"""OCR engines for subtitle bands.

EasyOCR is the default (best results on scene-overlaid Devanagari in our
tests, and it runs on-device); the Protocol keeps the engine swappable without
touching the rest of the pipeline. SarvamVisionOcr is an opt-in quality engine
that reads hard Devanagari bands more accurately at the cost of a cloud call.
"""

from __future__ import annotations

import re
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

# One Vision job runs per subtitle band, so a long clip fires dozens back to
# back and the endpoint answers a burst with 429 Rate limit exceeded - which,
# unhandled, crashes the whole run mid-way. Back off and retry, mirroring the
# ASR cross-check (match.asr): honour the server's Retry-After when present,
# else exponential backoff. A 429 that survives every retry still raises.
SARVAM_VISION_MAX_RETRIES = 5
SARVAM_VISION_BACKOFF_S = 2.0


# Fed a band with no legible subtitle, Sarvam Vision sometimes narrates the
# picture instead of transcribing it, e.g. `यह छवि एक ग्रेस्केल (black and white)
# है जिसमें एक व्यक्ति के हाथ पर दो मोतियों की मालाएँ दिखाई दे रही हैं।` (this image
# is grayscale, showing pearl garlands on a hand). That caption is not a
# subtitle and must not become a subtitle line. It carries give-away markers a
# dialogue line never has: an English rendering term reported in the OCR
# output, or an image/scene referent that opens the sentence. On Marathi bands
# the narration comes back in Marathi (`प्रतिमामध्ये ... दिसत आहे`), so the
# Marathi referents sit alongside the Hindi ones.
_DESCRIBE_MARKERS = (
    "ग्रेस्केल",  # grayscale, transliterated
    "कृष्णधवल",  # black-and-white, Marathi
    "grayscale",
    "greyscale",
    "black and white",
)
_DESCRIBE_OPENERS = (
    "यह छवि",  # this image ...
    "यह दृश्य",  # this scene ...
    "यह तस्वीर",  # this picture ...
    "इस छवि",
    "इस तस्वीर",
    "छवि में",  # in the image ...
    "तस्वीर में",  # in the picture ...
    "प्रतिमामध्ये",  # Marathi: in the image ...
    "प्रतिमे",  # stem: प्रतिमेत / प्रतिमेमध्ये
    "या प्रतिमे",
    "या चित्रात",  # Marathi: in this picture ...
    "या छायाचित्रात",  # Marathi: in this photograph ...
    "चित्रात",
)
# A narration runs to several sentences; one burned subtitle block never does
# (the guidelines cap a whole screen at 70 characters). Past this length a
# block is a description whatever language or opener it uses.
_DESCRIBE_MIN_CHARS = 150


def _is_image_description(text: str) -> bool:
    """True when Sarvam Vision narrated the frame instead of reading a subtitle.

    Recognised by markers a dialogue subtitle never carries: an English
    rendering term anywhere in the text (`black and white`, `grayscale`), an
    image/scene referent that opens the sentence (`यह छवि ...`, `यह दृश्य ...`,
    Marathi `प्रतिमामध्ये ...`), or a length no single subtitle block reaches.
    Kept tight - the openers must lead - so a line that merely mentions an image
    is not mistaken for a caption.
    """
    if len(text.strip()) > _DESCRIBE_MIN_CHARS:
        return True
    low = text.lower()
    if any(m in low for m in _DESCRIBE_MARKERS):
        return True
    stripped = text.lstrip()
    return any(stripped.startswith(o) for o in _DESCRIBE_OPENERS)


# A channel logo Sarvam Vision merges into the same block as the subtitle
# (e.g. `खुशी की क्या ही बात है? TATA PLAY`) rides along with the Devanagari line,
# so the block-level content test keeps it. The logos on this footage are
# Latin-script (`TATA PLAY`, `DD Free Dish`, `MELBON`) while a burned Hindi
# subtitle is pure Devanagari, so a run of Latin letters inside an
# otherwise-Devanagari line is the logo and is removed. Truncated OCR reads
# (`TATA PL`, `TATA P`) fall out of the same rule. Only Devanagari-majority
# blocks reach here (a Latin-majority block is dropped whole upstream), so this
# never strips a line that is legitimately another script.
_LATIN_RUN = re.compile(r"[A-Za-z]+(?:\s+[A-Za-z]+)*")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([?!.,।])")


def _strip_channel_logo(text: str) -> str:
    """Remove a Latin-script channel logo merged into a Devanagari subtitle line.

    Cuts any run of Latin words out of the line, then tidies the gap it left:
    collapses the doubled spaces and drops a space stranded before punctuation.
    A pure Devanagari line has no Latin run and comes back unchanged.
    """
    cleaned = _LATIN_RUN.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)


def _vision_text(blocks: list[str]) -> tuple[str, float]:
    """Join Sarvam Vision text blocks into one line, dropping non-subtitle boxes.

    Sarvam reads the whole crop, so a channel logo that survived Stage-1
    chrome subtraction (`DD Free Dish`, `TATA PLAY`) comes back as its own
    block; the same Devanagari content test used for EasyOCR drops it. A logo
    merged into a real line's block instead is stripped in place (see
    _strip_channel_logo). Image-description captions (see _is_image_description)
    are dropped as well, so a narrated frame does not surface as a spurious
    subtitle line.
    """
    lines = [b for b in blocks if _is_devanagari_line(b) and not _is_image_description(b)]
    lines = [stripped for b in lines if (stripped := _strip_channel_logo(b))]
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


def _start_vision_job(client, png_path: str, lang: str):
    """Create + start one Vision job, retrying past a 429 rate limit.

    A long clip fires one job per band in quick succession and the endpoint
    429s a burst. Retry with backoff (Retry-After if the server sends one, else
    exponential) so the run rides out the limit instead of crashing. A 429 that
    outlasts every retry propagates.
    """
    import time

    from sarvamai.errors.too_many_requests_error import TooManyRequestsError

    for attempt in range(SARVAM_VISION_MAX_RETRIES + 1):
        try:
            job = client.document_intelligence.create_job(
                language=lang, output_format="md"
            )
            job.upload_file(png_path)
            job.start()
            return job
        except TooManyRequestsError:
            if attempt >= SARVAM_VISION_MAX_RETRIES:
                raise
            time.sleep(_vision_retry_after(attempt))
    raise AssertionError("unreachable")  # loop returns or raises every path


def _vision_retry_after(attempt: int) -> float:
    """Seconds to wait before retrying a 429 - exponential backoff by attempt."""
    return SARVAM_VISION_BACKOFF_S * (2**attempt)


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

    # Windows cannot delete a file the SDK may still hold open; a leftover temp
    # file must not fail the OCR.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        png = tmp / "band.png"
        Image.fromarray(band).save(png)
        job = _start_vision_job(client, str(png), lang)
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
