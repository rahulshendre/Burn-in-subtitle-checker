"""Stage 3 secondary signal: ASR cross-check.

Forced alignment (align.py) catches gross mismatches but not single-word errors
- a swapped word barely moves the alignment score. An Indic-specialised ASR
does catch them: transcribe the audio under a subtitle and compare the words.
Sarvam reads Devanagari off noisy PlanetRead audio far better than Whisper (the
six-month finding), so Sarvam Saaras v4 is the default engine; the AsrEngine
Protocol keeps it swappable and lets the pipeline and tests run with no network.

The comparison is rapidfuzz token_set_ratio - order-insensitive, which suits
Indic word-order drift and the fact that ASR and OCR tokenise a little
differently. A low ratio on a line alignment let pass is a word-level
TEXT_MISMATCH. Precision-first: only trusted lines (speech under them, decent
OCR, long enough) are checked, and a blank transcript abstains.
"""

from __future__ import annotations

import io
import os
import wave
from dataclasses import replace
from typing import Protocol

import numpy as np
from rapidfuzz.fuzz import token_set_ratio

from subtitle_checker.artifacts import AudioRegion, CheckResult, SubtitleEvent, Verdict
from subtitle_checker.match.scoring import combined_score
from subtitle_checker.match.structural import event_has_speech

SAMPLE_RATE = 16_000
# A little grace so the first/last word of a line is not clipped from the window.
WINDOW_PAD_S = 0.3
# token_set_ratio is 0-100. Live Sarvam on real audio: correct lines cluster
# 82-100, gross divergence ~30, and single-word swaps sit at 75-93 - overlapping
# correct, because OCR and ASR already disagree ~15% on spelling and word order.
# So the cut flags only gross mismatches: it spares correct lines and misses
# subtle single-word swaps (those are surfaced heard-vs-written in the report for
# the editor, not auto-flagged - no full-line text metric separates them).
MIN_TOKEN_RATIO = 65.0
# Match alignment's trust gates: do not cross-check garbled OCR or tiny lines.
MIN_OCR_CONF = 0.5
MIN_WORDS = 3
# A gross mismatch needs a long enough line behind it, mirroring alignment's
# MIN_MISMATCH_SPAN. A caption under ~1.5 s is too little audio for the ASR to
# transcribe reliably, so a low word-match on it is as likely a garbled short
# window as a wrong subtitle. Guideline-compliant one-line captions run this
# short, so without the floor they produced most of the false flags. The line
# still appears in the ledger with its heard-vs-written for the editor - it is
# only held back from the auto-flag list.
MIN_MISMATCH_SPAN = 1.5
# Fewest words a transcript of a MISSING span needs before it is offered as the
# suggested caption text. A one- or two-word blurt is as likely an ASR latch as a
# real line; below this the tool keeps its honest "could not determine" rather
# than propose a fragment. Same spirit as MIN_HEARD_WORDS in report.suggest.
MIN_SUGGEST_WORDS = 3

SARVAM_URL = "https://api.sarvam.ai/speech-to-text"
# Saaras v4 is Sarvam's current STT model (v3 remains available; v4 adds Global
# English and expanded language support, same transcribe mode, same endpoint).
SARVAM_MODEL = "saaras:v4"
SARVAM_MODE = "transcribe"
# The sync endpoint rate-limits a fast burst of per-line calls with a 429. A long
# clip has dozens of lines, so a naive loop trips it and crashes the run mid-way.
# Back off and retry instead: honour the server's Retry-After when it sends one,
# else exponential backoff. A 429 that survives every retry still raises.
SARVAM_MAX_RETRIES = 5
SARVAM_BACKOFF_S = 2.0


class AsrEngine(Protocol):
    def transcribe(self, audio: np.ndarray) -> str:
        """Mono 16 kHz float32 window -> transcript text ("" if nothing heard)."""
        ...


class SarvamAsr:
    """Sarvam Saaras v4 speech-to-text (transcribe mode).

    Reads SARVAM_API_KEY from the environment and never stores it. `lang` is
    Sarvam's BCP-47 code (hi-IN, kn-IN, mr-IN). The sync endpoint is short-audio
    only, which is exactly what a per-subtitle window is.
    """

    def __init__(self, lang: str = "hi-IN") -> None:
        self._lang = lang

    def transcribe(self, audio: np.ndarray) -> str:
        import time

        import requests

        key = os.environ.get("SARVAM_API_KEY")
        if not key:
            raise RuntimeError("SARVAM_API_KEY not set in the environment")
        wav = _to_wav(audio).getvalue()
        for attempt in range(SARVAM_MAX_RETRIES + 1):
            resp = requests.post(
                SARVAM_URL,
                headers={"api-subscription-key": key},
                # Rebuild the file object each attempt - the previous POST consumed it.
                files={"file": ("audio.wav", io.BytesIO(wav), "audio/wav")},
                data={
                    "model": SARVAM_MODEL,
                    "mode": SARVAM_MODE,
                    "language_code": self._lang,
                },
                timeout=120,
            )
            if resp.status_code == 429 and attempt < SARVAM_MAX_RETRIES:
                time.sleep(_retry_after(resp, attempt))
                continue
            resp.raise_for_status()
            return resp.json().get("transcript", "").strip()
        raise AssertionError("unreachable")  # loop returns or raises every path


def _retry_after(resp: object, attempt: int) -> float:
    """Seconds to wait before retrying a 429 - server's Retry-After, else backoff."""
    header = getattr(resp, "headers", {}).get("Retry-After")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    return SARVAM_BACKOFF_S * (2**attempt)


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


def _heard_line(
    event: SubtitleEvent,
    audio: np.ndarray,
    regions: list[AudioRegion],
    engine: AsrEngine,
    min_ocr_conf: float,
    min_words: int,
    sample_rate: int,
    pad: float,
) -> tuple[str, float] | None:
    """Transcribe one trusted, speech-covered line -> (heard, token_set_ratio).

    Returns None when the line is not worth comparing: no speech beneath it,
    untrusted or too-short OCR, an empty window, or the ASR heard nothing. The
    shared trust gate behind both the flags and the report ledger.
    """
    if not event_has_speech(event, regions):
        return None  # no speech to transcribe - structural's call
    if event.confidence < min_ocr_conf or len(event.text.split()) < min_words:
        return None  # untrusted or too short to compare word-for-word
    w0 = max(0.0, event.start - pad)
    w1 = min(len(audio) / sample_rate, event.end + pad)
    window = audio[int(w0 * sample_rate) : int(w1 * sample_rate)]
    if not window.size:
        return None
    heard = engine.transcribe(window)
    if not heard:
        return None  # ASR heard nothing - abstain rather than accuse
    return heard, token_set_ratio(event.text, heard)


def transcribe_lines(
    events: list[SubtitleEvent],
    audio: np.ndarray,
    regions: list[AudioRegion],
    engine: AsrEngine,
    min_ratio: float = MIN_TOKEN_RATIO,
    min_ocr_conf: float = MIN_OCR_CONF,
    min_words: int = MIN_WORDS,
    min_span: float = MIN_MISMATCH_SPAN,
    sample_rate: int = SAMPLE_RATE,
    pad: float = WINDOW_PAD_S,
) -> list[CheckResult]:
    """Transcribe every comparable line for the report's heard-vs-written ledger.

    One CheckResult per trusted, speech-covered line - OK when the heard words
    match the subtitle, TEXT_MISMATCH when they diverge grossly - each carrying
    heard_text. A low match on a line shorter than ``min_span`` stays OK (too
    little audio to trust the mismatch), still shown in the ledger but held back
    from the flags. The OK rows are what let an editor eyeball the subtle
    single-word errors that sit below the auto-flag noise floor (see the module
    docstring).
    """
    results: list[CheckResult] = []
    for event in events:
        heard_ratio = _heard_line(
            event, audio, regions, engine, min_ocr_conf, min_words, sample_rate, pad
        )
        if heard_ratio is None:
            continue
        heard, ratio = heard_ratio
        match = ratio / 100.0
        if ratio >= min_ratio:
            verdict = Verdict.OK
            reason = f"heard words match the subtitle (match {ratio:.0f}%)"
        elif event.end - event.start < min_span:
            verdict = Verdict.OK
            reason = f"line too short to flag a mismatch (match {ratio:.0f}%)"
        else:
            verdict = Verdict.TEXT_MISMATCH
            reason = f"heard words differ from the subtitle (match {ratio:.0f}%)"
        results.append(
            CheckResult(
                start=event.start,
                end=event.end,
                verdict=verdict,
                reason=reason,
                subtitle_text=event.text,
                heard_text=heard,
                score=match,
                ocr_confidence=event.confidence,
                combined_score=combined_score(event.confidence, match),
            )
        )
    return results


def skipped_lines(
    events: list[SubtitleEvent],
    results: list[CheckResult],
    regions: list[AudioRegion] | None = None,
    *,
    min_ocr_conf: float = MIN_OCR_CONF,
    min_words: int = MIN_WORDS,
) -> list[tuple[SubtitleEvent, str]]:
    """Detected lines that got no verdict row, each with why they were declined.

    Mirrors _heard_line's trust gate so the report can show an editor that a
    line was noticed and passed over deliberately, not missed. Without
    ``regions`` the speech test is skipped (the artifact may be absent).
    """

    def claimed(e: SubtitleEvent) -> bool:
        return any(
            abs(r.start - e.start) < 0.05 and abs(r.end - e.end) < 0.05 for r in results
        )

    skipped: list[tuple[SubtitleEvent, str]] = []
    for event in events:
        if claimed(event):
            continue
        if regions is not None and not event_has_speech(event, regions):
            reason = "no speech under this line"
        elif event.confidence < min_ocr_conf:
            reason = (
                f"OCR read too unreliable to compare (confidence {event.confidence:.2f})"
            )
        elif len(event.text.split()) < min_words:
            reason = "too short to compare word-for-word"
        else:
            reason = "nothing was transcribed for this line"
        skipped.append((event, reason))
    return skipped


def transcribe_missing(
    flags: list[CheckResult],
    audio: np.ndarray,
    engine: AsrEngine,
    min_words: int = MIN_SUGGEST_WORDS,
    sample_rate: int = SAMPLE_RATE,
    pad: float = WINDOW_PAD_S,
) -> list[CheckResult]:
    """Fill in what the audio says under each MISSING_SUBTITLE span.

    A missing-subtitle flag is raised by structural (structural.py) from the voice
    detector alone - it knows speech is there but never transcribed it, so the
    report can only say "not transcribed". This runs ASR on those spans so an
    editor sees what the caption should have said, as an unverified best guess.

    ASR is the weak side of the tool, so this is deliberately a suggestion and
    never a verdict: the flag stays MISSING_SUBTITLE, only its ``heard_text`` is
    filled, and only when the transcript carries enough words to be worth reading.
    Non-MISSING flags are returned untouched. A MISSING span is by definition a
    speech region (that is what raised it), so no music gate is needed here.
    """
    out: list[CheckResult] = []
    for f in flags:
        if f.verdict is not Verdict.MISSING_SUBTITLE:
            out.append(f)
            continue
        w0 = max(0.0, f.start - pad)
        w1 = min(len(audio) / sample_rate, f.end + pad)
        window = audio[int(w0 * sample_rate) : int(w1 * sample_rate)]
        heard = engine.transcribe(window) if window.size else ""
        if len(heard.split()) < min_words:
            out.append(f)  # too little heard to suggest - keep the honest blank
            continue
        out.append(replace(f, heard_text=heard))
    return out


def check_asr(
    events: list[SubtitleEvent],
    audio: np.ndarray,
    regions: list[AudioRegion],
    engine: AsrEngine,
    min_ratio: float = MIN_TOKEN_RATIO,
    min_ocr_conf: float = MIN_OCR_CONF,
    min_words: int = MIN_WORDS,
    min_span: float = MIN_MISMATCH_SPAN,
    sample_rate: int = SAMPLE_RATE,
    pad: float = WINDOW_PAD_S,
) -> list[CheckResult]:
    """Flag speech-covered lines whose heard words differ from the subtitle.

    The flag subset of transcribe_lines: only the gross TEXT_MISMATCH rows, for
    the eval harness and the pipeline's flag list.
    """
    return [
        r
        for r in transcribe_lines(
            events, audio, regions, engine,
            min_ratio, min_ocr_conf, min_words, min_span, sample_rate, pad,
        )
        if r.verdict is Verdict.TEXT_MISMATCH
    ]
