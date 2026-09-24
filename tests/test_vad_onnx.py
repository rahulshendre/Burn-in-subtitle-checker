"""Tests for the torch-free Silero VAD port."""

from __future__ import annotations

import numpy as np
import pytest

from subtitle_checker.audio.vad import _WINDOW, SAMPLE_RATE, _timestamps


def _probs(pattern: list[tuple[float, float]]) -> list[float]:
    """Per-window probabilities from (seconds, prob) runs."""
    out: list[float] = []
    for seconds, prob in pattern:
        out += [prob] * int(seconds * SAMPLE_RATE / _WINDOW)
    return out


def test_one_speech_run_is_padded_and_in_seconds():
    probs = _probs([(1.0, 0.0), (2.0, 0.9), (1.0, 0.0)])
    spans = _timestamps(probs, len(probs) * _WINDOW, 0.5, 300)
    assert len(spans) == 1
    start, end = spans[0]
    assert start == pytest.approx(1.0, abs=0.1)
    assert end == pytest.approx(3.0, abs=0.1)


def test_short_pause_does_not_split_speech():
    probs = _probs([(1.0, 0.9), (0.2, 0.0), (1.0, 0.9), (1.0, 0.0)])
    assert len(_timestamps(probs, len(probs) * _WINDOW, 0.5, 300)) == 1


def test_long_pause_splits_and_blips_are_dropped():
    probs = _probs([(1.0, 0.9), (0.6, 0.0), (1.0, 0.9), (0.6, 0.0), (0.1, 0.9), (1.0, 0.0)])
    assert len(_timestamps(probs, len(probs) * _WINDOW, 0.5, 300)) == 2


def test_onnx_model_runs_on_silence():
    pytest.importorskip("onnxruntime")
    from subtitle_checker.audio.vad import SileroOnnxVad

    assert SileroOnnxVad().speech_spans(np.zeros(SAMPLE_RATE * 2, dtype=np.float32)) == []
