"""Forced alignment — Stage 3's primary signal.

The subtitle text is known and trustworthy (Stage 1 OCR). The question is not
"what was said?" but "was THIS line spoken here?" — verification, not open
transcription. Forced alignment answers it: score the known words against the
audio under the subtitle. A low score means those words were not spoken in that
window.

Scoring known text sidesteps the whole hallucination failure class that sank
open ASR (Whisper) on this footage. The aligner is a CTC model (torchaudio's
MMS_FA bundle) covering Hindi, Marathi, and Kannada; the Protocol keeps it
swappable and lets the pipeline and the tests run without torch — exactly how
SileroVad sits behind VoiceActivityDetector in audio.vad.

MMS aligns romanized text, so the concrete aligner transliterates Devanagari
with uroman before scoring. That transliteration is the known risk flagged in
PLAN_V2; it lives inside MmsAligner, not in the pure scorer, so a script-native
aligner could drop in without touching anything else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from subtitle_checker.artifacts import SubtitleEvent

SAMPLE_RATE = 16_000

# Small grace on the audio window so a word starting a hair before the subtitle
# appears is not clipped. Kept tight on purpose: a wide window feeds the aligner
# neighbouring speech that the subtitle's own words must then also cover, which
# drags the score down. (Stage 2 pads generously to bridge blinks; alignment
# wants the true onset, so it pads far less.)
WINDOW_PAD_S = 0.2


@dataclass
class WordSpan:
    """One aligned word: where in the window it landed and how confident."""

    text: str
    start: float  # seconds from the window start
    end: float
    score: float  # CTC posterior, 0..1


@dataclass
class AlignmentScore:
    """How well one subtitle event's words align to the audio beneath it."""

    start: float  # subtitle event span, absolute seconds
    end: float
    text: str
    ocr_confidence: float  # Stage-1 OCR confidence for text; gates trust in a low score
    score: float | None  # frame-weighted mean CTC confidence 0..1; None = unalignable
    aligned_start: float  # first aligned word, absolute seconds
    aligned_end: float  # last aligned word, absolute seconds


class ForcedAligner(Protocol):
    def align(self, audio: np.ndarray, text: str) -> list[WordSpan]:
        """Align an expected transcript line against a window of mono 16 kHz
        float32 audio. Returns one span per aligned word with times relative to
        the window start, or an empty list when nothing could be aligned."""
        ...


def _mean_score(spans: list[WordSpan]) -> float:
    """Frame-count-weighted mean of per-word scores.

    A longer word carries more acoustic evidence, so it weighs more — one short
    filler word scoring low should not sink an otherwise solid line. Falls back
    to a plain mean when every span is zero-length.
    """
    if not spans:
        return 0.0
    weights = [max(s.end - s.start, 0.0) for s in spans]
    total = sum(weights)
    if total <= 0:
        return sum(s.score for s in spans) / len(spans)
    return sum(s.score * w for s, w in zip(spans, weights)) / total


def score_event(
    event: SubtitleEvent,
    audio: np.ndarray,
    aligner: ForcedAligner,
    sample_rate: int = SAMPLE_RATE,
    pad: float = WINDOW_PAD_S,
) -> AlignmentScore:
    """Align one subtitle event's text against its slice of the audio track."""
    w0 = max(0.0, event.start - pad)
    w1 = min(len(audio) / sample_rate, event.end + pad)
    window = audio[int(w0 * sample_rate) : int(w1 * sample_rate)]
    spans = aligner.align(window, event.text) if window.size else []
    if not spans:
        # Could not align at all — empty text, or a window too short to fit the
        # line's tokens under the CTC length rule. That is not a mismatch, so
        # score is None and the verdict layer marks it UNCHECKABLE, never
        # TEXT_MISMATCH.
        return AlignmentScore(
            start=event.start,
            end=event.end,
            text=event.text,
            ocr_confidence=event.confidence,
            score=None,
            aligned_start=event.start,
            aligned_end=event.end,
        )
    return AlignmentScore(
        start=event.start,
        end=event.end,
        text=event.text,
        ocr_confidence=event.confidence,
        score=_mean_score(spans),
        aligned_start=w0 + spans[0].start,
        aligned_end=w0 + spans[-1].end,
    )


def score_events(
    events: list[SubtitleEvent],
    audio: np.ndarray,
    aligner: ForcedAligner,
    sample_rate: int = SAMPLE_RATE,
    pad: float = WINDOW_PAD_S,
) -> list[AlignmentScore]:
    """Alignment score for every subtitle event, in order."""
    return [score_event(e, audio, aligner, sample_rate, pad) for e in events]


class MmsAligner:
    """torchaudio MMS_FA forced aligner.

    The heavy imports and the model load wait until first use (mirrors
    SileroVad). Devanagari is romanized with uroman first because the MMS
    dictionary is Latin; `lang` is the ISO 639-3 code uroman expects (hin, mar,
    kan).
    """

    def __init__(self, lang: str = "hin") -> None:
        self._lang = lang
        self._model = None
        self._tokenizer = None
        self._aligner = None
        self._uroman = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from torchaudio.pipelines import MMS_FA as bundle

        # with_star=False: the subtitle text is expected to match the windowed
        # audio, so no "*" token to absorb unrelated speech. Revisit if a padded
        # window's neighbouring speech proves to drag scores (smoke test).
        self._model = bundle.get_model(with_star=False)
        self._model.eval()
        self._tokenizer = bundle.get_tokenizer()
        self._aligner = bundle.get_aligner()

    def _romanize(self, text: str) -> list[str]:
        from uroman import Uroman

        if self._uroman is None:
            self._uroman = Uroman()
        latin = self._uroman.romanize_string(text, lcode=self._lang).lower()
        latin = re.sub(r"[^a-z' ]", " ", latin)
        return latin.split()

    def align(self, audio: np.ndarray, text: str) -> list[WordSpan]:
        import torch

        words = self._romanize(text)
        if not words or audio.size == 0:
            return []
        self._ensure_loaded()
        waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            emission, _ = self._model(waveform)
            try:
                token_spans = self._aligner(emission[0], self._tokenizer(words))
            except RuntimeError as exc:
                # CTC needs at least one audio frame per target token (plus
                # adjacent repeats). A very short event carrying a full line of
                # text — a Stage-1 flash or split duplicate — violates that. It
                # is unverifiable here, not wrong, so return no spans.
                if "targets length is too long" in str(exc):
                    return []
                raise
        seconds_per_frame = waveform.size(1) / emission.size(1) / SAMPLE_RATE
        out: list[WordSpan] = []
        for word, spans in zip(words, token_spans):
            if not spans:
                continue
            frames = sum(len(s) for s in spans) or 1
            score = sum(s.score * len(s) for s in spans) / frames
            out.append(
                WordSpan(
                    text=word,
                    start=spans[0].start * seconds_per_frame,
                    end=spans[-1].end * seconds_per_frame,
                    score=float(score),
                )
            )
        return out
