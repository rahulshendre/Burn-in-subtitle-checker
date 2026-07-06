"""Voice activity detection over an audio track.

Silero VAD is the default; the Protocol keeps it swappable and lets the rest of
the pipeline (and the tests) run without the torch model. Input is mono 16 kHz
float32 (see ingest.audio_track); output is speech spans in seconds.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

SAMPLE_RATE = 16_000


class VoiceActivityDetector(Protocol):
    def speech_spans(self, audio: np.ndarray) -> list[tuple[float, float]]:
        """Mono 16 kHz float32 → list of (start, end) speech spans in seconds."""
        ...


class SileroVad:
    """Wraps silero-vad; the torch import and model load wait until first use."""

    def __init__(self, threshold: float = 0.5, min_silence_ms: int = 300) -> None:
        self._threshold = threshold
        self._min_silence_ms = min_silence_ms
        self._model = None

    def speech_spans(self, audio: np.ndarray) -> list[tuple[float, float]]:
        import torch  # pulls in the heavy stack — keep it off module import
        from silero_vad import get_speech_timestamps, load_silero_vad

        if self._model is None:
            self._model = load_silero_vad()
        stamps = get_speech_timestamps(
            torch.from_numpy(audio),
            self._model,
            sampling_rate=SAMPLE_RATE,
            threshold=self._threshold,
            min_silence_duration_ms=self._min_silence_ms,
            return_seconds=True,
        )
        return [(float(s["start"]), float(s["end"])) for s in stamps]
