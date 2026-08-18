"""Tests for the whole-transcript audio-vs-script aligner."""

from __future__ import annotations

import numpy as np

from subtitle_checker.artifacts import SubtitleEvent, Verdict
from subtitle_checker.match.script_align import (
    align_script,
    transcribe_full,
)


def _cue(text: str, start: float, end: float) -> SubtitleEvent:
    return SubtitleEvent(start=start, end=end, text=text, confidence=1.0)


def test_perfect_match_is_ok() -> None:
    events = [_cue("राम घर आया", 0.0, 2.0)]
    results = align_script(events, ["राम", "घर", "आया"])
    assert results[0].verdict is Verdict.OK
    assert results[0].combined_score == 100.0


def test_boundary_bleed_across_cues_still_matches_each_cue() -> None:
    # One sentence split across two short cues; the heard stream is continuous.
    # Each cue must get its own words back, not the neighbour's.
    events = [_cue("राम घर", 0.0, 1.0), _cue("जल्दी आया", 1.0, 2.0)]
    heard = ["राम", "घर", "जल्दी", "आया"]
    results = align_script(events, heard)
    assert [r.verdict for r in results] == [Verdict.OK, Verdict.OK]
    assert "राम" in results[0].heard_text and "आया" in results[1].heard_text


def test_leading_extra_word_does_not_shift_cues() -> None:
    # A stray heard word before the dialogue (music bleed) must not push every
    # cue's audio onto the wrong cue.
    events = [_cue("राम घर", 0.0, 1.0), _cue("जल्दी आया", 1.0, 2.0)]
    heard = ["तो", "राम", "घर", "जल्दी", "आया"]
    results = align_script(events, heard)
    assert [r.verdict for r in results] == [Verdict.OK, Verdict.OK]


def test_wrong_words_flag_text_mismatch() -> None:
    events = [_cue("बिल्कुल सही जवाब है यह", 0.0, 3.0)]
    results = align_script(events, ["कुछ", "और", "ही", "सुना", "गया"])
    assert results[0].verdict is Verdict.TEXT_MISMATCH
    assert results[0].combined_score < 65.0


def test_single_spelling_variant_stays_ok() -> None:
    # ASR spelling drift on one token (समधन vs समदन) must not flag a good line.
    events = [_cue("आप गलत कह रही हैं समधन", 0.0, 3.0)]
    results = align_script(events, ["आप", "गलत", "कह", "रही", "हैं", "समदन"])
    assert results[0].verdict is Verdict.OK


def test_short_cue_never_flagged() -> None:
    # A one-word cue is folded into a neighbour by the ASR; too little to judge.
    events = [_cue("हाँ?", 0.0, 0.5)]
    results = align_script(events, [])
    assert results[0].verdict is Verdict.OK
    assert "too short" in results[0].reason


def test_transcribe_full_concatenates_chunks_in_order() -> None:
    class FakeAsr:
        def __init__(self, replies: list[str]) -> None:
            self._replies = replies
            self.calls = 0

        def transcribe(self, audio: np.ndarray) -> str:
            reply = self._replies[self.calls]
            self.calls += 1
            return reply

    audio = np.zeros(int(16_000 * 45), dtype=np.float32)  # 45 s -> 3 chunks at 20 s
    engine = FakeAsr(["पहला दूसरा", "तीसरा", "चौथा शब्द"])
    words = transcribe_full(audio, engine)
    assert words == ["पहला", "दूसरा", "तीसरा", "चौथा", "शब्द"]
    assert engine.calls == 3
