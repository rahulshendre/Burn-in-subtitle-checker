"""Region labelling logic, driven by a fake VAD so no model is needed."""

from __future__ import annotations

import numpy as np

from subtitle_checker.artifacts import AudioKind
from subtitle_checker.audio.regions import label_regions
from subtitle_checker.audio.vad import SAMPLE_RATE


class FakeVad:
    """Returns a fixed set of speech spans, ignoring the audio."""

    def __init__(self, spans: list[tuple[float, float]]) -> None:
        self._spans = spans

    def speech_spans(self, audio: np.ndarray) -> list[tuple[float, float]]:
        return self._spans


def _audio(duration: float, amplitude: float = 0.0) -> np.ndarray:
    n = int(duration * SAMPLE_RATE)
    if amplitude == 0.0:
        return np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(0)
    return (rng.standard_normal(n) * amplitude).astype(np.float32)


def test_timeline_is_gap_free_and_ordered() -> None:
    audio = _audio(10.0)
    regions = label_regions(audio, FakeVad([(2.0, 4.0), (6.0, 8.0)]))
    assert regions[0].start == 0.0
    assert abs(regions[-1].end - 10.0) < 1e-6
    for a, b in zip(regions, regions[1:]):
        assert abs(a.end - b.start) < 1e-6  # no holes, no overlaps


def test_speech_spans_become_speech_regions() -> None:
    audio = _audio(10.0)
    regions = label_regions(audio, FakeVad([(2.0, 4.0), (6.0, 8.0)]))
    speech = [r for r in regions if r.kind == AudioKind.SPEECH]
    assert [(r.start, r.end) for r in speech] == [(2.0, 4.0), (6.0, 8.0)]


def test_quiet_gap_is_silence_loud_gap_is_music() -> None:
    # loud 0-3 s, then silent to the end; speech spans split the gaps apart
    audio = np.concatenate([_audio(3.0, amplitude=0.2), _audio(7.0, amplitude=0.0)])
    regions = label_regions(audio, FakeVad([(3.0, 4.0), (7.0, 8.0)]))
    kinds = [(round(r.start, 1), round(r.end, 1), r.kind) for r in regions]
    assert (0.0, 3.0, AudioKind.MUSIC) in kinds
    assert (4.0, 7.0, AudioKind.SILENCE) in kinds
    assert (3.0, 4.0, AudioKind.SPEECH) in kinds
    assert (8.0, 10.0, AudioKind.SILENCE) in kinds


def test_no_speech_gives_single_gap_region() -> None:
    regions = label_regions(_audio(5.0), FakeVad([]))
    assert len(regions) == 1
    assert regions[0].kind == AudioKind.SILENCE
    assert (regions[0].start, round(regions[0].end, 1)) == (0.0, 5.0)


def test_empty_audio_gives_no_regions() -> None:
    assert label_regions(np.zeros(0, dtype=np.float32), FakeVad([])) == []


def test_spans_clamped_and_touching_speech_merged() -> None:
    # VAD over-runs the end and returns two touching spans
    audio = _audio(5.0)
    regions = label_regions(audio, FakeVad([(1.0, 3.0), (3.0, 9.0)]))
    speech = [r for r in regions if r.kind == AudioKind.SPEECH]
    assert len(speech) == 1
    assert (speech[0].start, round(speech[0].end, 1)) == (1.0, 5.0)
