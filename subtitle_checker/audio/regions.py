"""Turn VAD output into a gap-free timeline of labelled audio regions.

Speech spans come from the VAD. Everything the VAD leaves out is a gap; each
gap is classified MUSIC or SILENCE by its energy. The result covers the whole
track end to end, so a downstream structural check can ask "what is under this
subtitle?" for any timestamp and always get an answer.

SONG is deliberately not emitted yet - separating sung vocals from a backing
score needs source separation (Stage 2's optional step). Emitting a SONG label
we can't yet stand behind would be dishonest; MUSIC/SPEECH is the coarse split
the plan calls for.
"""

from __future__ import annotations

import numpy as np

from subtitle_checker.artifacts import AudioRegion, AudioKind

from .vad import SAMPLE_RATE, VoiceActivityDetector

WINDOW_S = 0.1
# Median 100 ms-window RMS above this reads as music; below, as silence.
# Coarse by design; tuned on the Hindi clips and refined during validation.
MUSIC_RMS_FLOOR = 0.015


def _median_window_rms(samples: np.ndarray, sample_rate: int) -> float:
    """Median RMS across 100 ms windows - ignores one-off transient spikes."""
    if samples.size == 0:
        return 0.0
    win = max(1, int(WINDOW_S * sample_rate))
    n = samples.size // win
    if n == 0:
        return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    frames = samples[: n * win].astype(np.float64).reshape(n, win)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return float(np.median(rms))


def _classify_gap(
    audio: np.ndarray, start: float, end: float, sample_rate: int, music_rms: float
) -> AudioKind:
    lo = int(start * sample_rate)
    hi = int(end * sample_rate)
    energy = _median_window_rms(audio[lo:hi], sample_rate)
    return AudioKind.MUSIC if energy >= music_rms else AudioKind.SILENCE


def label_regions(
    audio: np.ndarray,
    vad: VoiceActivityDetector,
    sample_rate: int = SAMPLE_RATE,
    music_rms: float = MUSIC_RMS_FLOOR,
) -> list[AudioRegion]:
    """Full-timeline speech / music / silence regions for the whole track."""
    duration = len(audio) / sample_rate
    if duration <= 0:
        return []

    spans = sorted(
        (max(0.0, s), min(duration, e)) for s, e in vad.speech_spans(audio) if e > s
    )

    regions: list[AudioRegion] = []
    cursor = 0.0
    for start, end in spans:
        if start > cursor:  # non-speech gap before this speech span
            kind = _classify_gap(audio, cursor, start, sample_rate, music_rms)
            regions.append(AudioRegion(cursor, start, kind))
        regions.append(AudioRegion(max(start, cursor), end, AudioKind.SPEECH))
        cursor = max(cursor, end)
    if cursor < duration:  # trailing gap after the last speech span
        kind = _classify_gap(audio, cursor, duration, sample_rate, music_rms)
        regions.append(AudioRegion(cursor, duration, kind))

    return _merge_adjacent(regions)


def _merge_adjacent(regions: list[AudioRegion]) -> list[AudioRegion]:
    """Fold touching regions of the same kind into one span."""
    merged: list[AudioRegion] = []
    for r in regions:
        if merged and merged[-1].kind == r.kind and abs(merged[-1].end - r.start) < 1e-6:
            merged[-1] = AudioRegion(merged[-1].start, r.end, r.kind)
        else:
            merged.append(r)
    return merged
