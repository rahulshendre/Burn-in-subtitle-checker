"""Voice activity detection over an audio track.

Silero VAD is the default; the Protocol keeps it swappable and lets the rest of
the pipeline (and the tests) run without the torch model. Input is mono 16 kHz
float32 (see ingest.audio_track); output is speech spans in seconds.
"""

from __future__ import annotations

from pathlib import Path
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
        import torch  # pulls in the heavy stack - keep it off module import
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


_ONNX_MODEL = Path(__file__).parent / "models" / "silero_vad.onnx"
_WINDOW = 512  # samples per model step at 16 kHz
_CONTEXT = 64  # samples of the previous window the model sees


class SileroOnnxVad:
    """Silero VAD on onnxruntime - same model, no torch.

    The torch build pulls in ~450 MB, too much for the on-device app, so this
    runs Silero's published ONNX export (MIT, shipped under models/) and ports
    get_speech_timestamps with the settings SileroVad uses: no maximum speech
    length, 250 ms minimum speech, 30 ms padding. Output matches SileroVad span
    for span on the project clips.
    """

    def __init__(self, threshold: float = 0.5, min_silence_ms: int = 300) -> None:
        self._threshold = threshold
        self._min_silence_ms = min_silence_ms
        self._session = None

    def _probs(self, audio: np.ndarray) -> list[float]:
        import onnxruntime  # keep the runtime off module import

        if self._session is None:
            opts = onnxruntime.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            self._session = onnxruntime.InferenceSession(str(_ONNX_MODEL), sess_options=opts)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, _CONTEXT), dtype=np.float32)
        sr = np.array(SAMPLE_RATE, dtype=np.int64)
        probs = []
        for start in range(0, len(audio), _WINDOW):
            chunk = audio[start : start + _WINDOW].astype(np.float32)
            if len(chunk) < _WINDOW:
                chunk = np.pad(chunk, (0, _WINDOW - len(chunk)))
            x = np.concatenate([context, chunk[None, :]], axis=1)
            out, state = self._session.run(None, {"input": x, "state": state, "sr": sr})
            context = x[:, -_CONTEXT:]
            probs.append(float(out.reshape(-1)[0]))
        return probs

    def speech_spans(self, audio: np.ndarray) -> list[tuple[float, float]]:
        probs = self._probs(audio)
        return _timestamps(probs, len(audio), self._threshold, self._min_silence_ms)


def _timestamps(
    probs: list[float], n_samples: int, threshold: float, min_silence_ms: int
) -> list[tuple[float, float]]:
    """Silero's get_speech_timestamps with no maximum speech length, in seconds."""
    min_speech = SAMPLE_RATE * 250 / 1000
    pad = SAMPLE_RATE * 30 / 1000
    min_silence = SAMPLE_RATE * min_silence_ms / 1000
    neg_threshold = max(threshold - 0.15, 0.01)

    speeches: list[dict] = []
    current: dict = {}
    triggered = False
    temp_end = 0
    for i, prob in enumerate(probs):
        cur = _WINDOW * i
        if prob >= threshold and temp_end:
            temp_end = 0
        if prob >= threshold and not triggered:
            triggered = True
            current["start"] = cur
            continue
        if prob < neg_threshold and triggered:
            if not temp_end:
                temp_end = cur
            if cur - temp_end < min_silence:
                continue
            current["end"] = temp_end
            if current["end"] - current["start"] > min_speech:
                speeches.append(current)
            current, temp_end, triggered = {}, 0, False
    if current and n_samples - current["start"] > min_speech:
        current["end"] = n_samples
        speeches.append(current)

    for i, sp in enumerate(speeches):
        if i == 0:
            sp["start"] = int(max(0, sp["start"] - pad))
        if i != len(speeches) - 1:
            gap = speeches[i + 1]["start"] - sp["end"]
            if gap < 2 * pad:
                sp["end"] += int(gap // 2)
                speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - gap // 2))
            else:
                sp["end"] = int(min(n_samples, sp["end"] + pad))
                speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - pad))
        else:
            sp["end"] = int(min(n_samples, sp["end"] + pad))

    length = n_samples / SAMPLE_RATE
    return [
        (float(max(round(sp["start"] / SAMPLE_RATE, 1), 0)),
         float(min(round(sp["end"] / SAMPLE_RATE, 1), length)))
        for sp in speeches
    ]
