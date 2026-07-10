"""ASR cross-check logic over a scripted engine — no network, real rapidfuzz."""

from __future__ import annotations

import wave

import numpy as np

from subtitle_checker.artifacts import AudioKind, AudioRegion, SubtitleEvent, Verdict
from subtitle_checker.match.asr import _to_wav, check_asr, transcribe_lines

SPEECH = [AudioRegion(0.0, 6.0, AudioKind.SPEECH)]
MUSIC = [AudioRegion(0.0, 6.0, AudioKind.MUSIC)]
AUDIO = np.zeros(16_000 * 8, dtype=np.float32)
LINE = SubtitleEvent(1.0, 4.0, "एक दो तीन चार", 0.9)


class ScriptedAsr:
    """Returns preset transcripts in call order; counts calls."""

    def __init__(self, *transcripts: str) -> None:
        self._q = list(transcripts)
        self.calls = 0

    def transcribe(self, audio: np.ndarray) -> str:
        self.calls += 1
        return self._q.pop(0)


def test_heard_matches_subtitle_is_unflagged() -> None:
    assert check_asr([LINE], AUDIO, SPEECH, ScriptedAsr("एक दो तीन चार")) == []


def test_heard_differs_is_text_mismatch() -> None:
    results = check_asr([LINE], AUDIO, SPEECH, ScriptedAsr("पाँच छह सात आठ"))
    assert len(results) == 1
    assert results[0].verdict is Verdict.TEXT_MISMATCH
    assert results[0].subtitle_text == "एक दो तीन चार"
    assert results[0].heard_text == "पाँच छह सात आठ"


def test_blank_transcript_abstains() -> None:
    assert check_asr([LINE], AUDIO, SPEECH, ScriptedAsr("")) == []


def test_event_without_speech_is_not_transcribed() -> None:
    engine = ScriptedAsr("पाँच छह सात आठ")
    assert check_asr([LINE], AUDIO, MUSIC, engine) == []
    assert engine.calls == 0  # no speech → never spent an API call


def test_low_confidence_and_short_lines_are_skipped() -> None:
    events = [
        SubtitleEvent(1.0, 4.0, "एक दो तीन चार", 0.2),  # low OCR conf
        SubtitleEvent(1.0, 4.0, "एक दो", 0.9),  # too few words
    ]
    engine = ScriptedAsr()
    assert check_asr(events, AUDIO, SPEECH, engine) == []
    assert engine.calls == 0


def test_transcribe_lines_emits_ok_row_with_heard_text() -> None:
    rows = transcribe_lines([LINE], AUDIO, SPEECH, ScriptedAsr("एक दो तीन चार"))
    assert len(rows) == 1
    assert rows[0].verdict is Verdict.OK
    assert rows[0].subtitle_text == "एक दो तीन चार"
    assert rows[0].heard_text == "एक दो तीन चार"  # heard kept even when it matches


def test_transcribe_lines_flags_gross_divergence() -> None:
    rows = transcribe_lines([LINE], AUDIO, SPEECH, ScriptedAsr("पाँच छह सात आठ"))
    assert rows[0].verdict is Verdict.TEXT_MISMATCH
    assert rows[0].heard_text == "पाँच छह सात आठ"


def test_transcribe_lines_skips_untrusted_lines() -> None:
    assert transcribe_lines([LINE], AUDIO, MUSIC, ScriptedAsr("x")) == []


def test_check_asr_keeps_only_mismatches() -> None:
    events = [LINE, SubtitleEvent(4.0, 7.0, "एक दो तीन चार", 0.9)]
    engine = ScriptedAsr("पाँच छह सात आठ", "एक दो तीन चार")  # first differs, second matches
    flags = check_asr(events, AUDIO, SPEECH, engine)
    assert len(flags) == 1
    assert flags[0].verdict is Verdict.TEXT_MISMATCH


def test_to_wav_is_valid_pcm() -> None:
    buf = _to_wav(np.zeros(1600, dtype=np.float32))
    with wave.open(buf, "rb") as w:
        assert w.getframerate() == 16_000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 1600
