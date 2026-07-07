"""Stage 3 secondary signal: ASR cross-check.

Forced alignment (align.py) catches gross mismatches but not single-word errors
— a swapped word barely moves the alignment score. An Indic-specialised ASR
does catch them: transcribe the audio under a subtitle and compare the words.
Sarvam Saarika reads Devanagari off noisy PlanetRead audio far better than
Whisper (the six-month finding), so it is the default engine; the AsrEngine
Protocol keeps it swappable and lets the pipeline and tests run with no network.

The comparison is rapidfuzz token_set_ratio — order-insensitive, which suits
Indic word-order drift and the fact that ASR and OCR tokenise a little
differently. A low ratio on a line alignment let pass is a word-level
TEXT_MISMATCH. Precision-first: only trusted lines (speech under them, decent
OCR, long enough) are checked, and a blank transcript abstains.
"""

from __future__ import annotations

import io
import os
import wave
from typing import Protocol

import numpy as np
from rapidfuzz.fuzz import token_set_ratio

from subtitle_checker.artifacts import AudioRegion, CheckResult, SubtitleEvent, Verdict
from subtitle_checker.match.structural import event_has_speech

SAMPLE_RATE = 16_000
# A little grace so the first/last word of a line is not clipped from the window.
WINDOW_PAD_S = 0.3
# token_set_ratio is 0-100; below this the heard and written words disagree
# enough to flag. Precision-first placeholder — retune from real Sarvam
# transcripts once the key is wired (see the alignment_eval pattern).
MIN_TOKEN_RATIO = 60.0
# Match alignment's trust gates: do not cross-check garbled OCR or tiny lines.
MIN_OCR_CONF = 0.5
MIN_WORDS = 3

SARVAM_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL = "saarika:v2.5"


class AsrEngine(Protocol):
    def transcribe(self, audio: np.ndarray) -> str:
        """Mono 16 kHz float32 window -> transcript text ("" if nothing heard)."""
        ...


class SarvamAsr:
    """Sarvam Saarika speech-to-text.

    Reads SARVAM_API_KEY from the environment and never stores it. `lang` is
    Sarvam's BCP-47 code (hi-IN, kn-IN, mr-IN). The sync endpoint is short-audio
    only, which is exactly what a per-subtitle window is.
    """

    def __init__(self, lang: str = "hi-IN") -> None:
        self._lang = lang

    def transcribe(self, audio: np.ndarray) -> str:
        import requests

        key = os.environ.get("SARVAM_API_KEY")
        if not key:
            raise RuntimeError("SARVAM_API_KEY not set in the environment")
        resp = requests.post(
            SARVAM_URL,
            headers={"api-subscription-key": key},
            files={"file": ("audio.wav", _to_wav(audio), "audio/wav")},
            data={"model": SARVAM_MODEL, "language_code": self._lang},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("transcript", "").strip()


def _to_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> io.BytesIO:
    """Encode mono float32 [-1, 1] samples as 16-bit PCM WAV in memory."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    buf.seek(0)
    return buf


def check_asr(
    events: list[SubtitleEvent],
    audio: np.ndarray,
    regions: list[AudioRegion],
    engine: AsrEngine,
    min_ratio: float = MIN_TOKEN_RATIO,
    min_ocr_conf: float = MIN_OCR_CONF,
    min_words: int = MIN_WORDS,
    sample_rate: int = SAMPLE_RATE,
    pad: float = WINDOW_PAD_S,
) -> list[CheckResult]:
    """Flag speech-covered lines whose heard words differ from the subtitle."""
    results: list[CheckResult] = []
    for event in events:
        if not event_has_speech(event, regions):
            continue  # no speech to transcribe — structural's call
        if event.confidence < min_ocr_conf or len(event.text.split()) < min_words:
            continue  # untrusted or too short to compare word-for-word
        w0 = max(0.0, event.start - pad)
        w1 = min(len(audio) / sample_rate, event.end + pad)
        window = audio[int(w0 * sample_rate) : int(w1 * sample_rate)]
        if not window.size:
            continue
        heard = engine.transcribe(window)
        if not heard:
            continue  # ASR heard nothing — abstain rather than accuse
        ratio = token_set_ratio(event.text, heard)
        if ratio < min_ratio:
            results.append(
                CheckResult(
                    start=event.start,
                    end=event.end,
                    verdict=Verdict.TEXT_MISMATCH,
                    reason=f"heard words differ from the subtitle (match {ratio:.0f}%)",
                    subtitle_text=event.text,
                    heard_text=heard,
                    score=ratio / 100.0,
                )
            )
    return results
