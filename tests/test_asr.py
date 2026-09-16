"""ASR cross-check logic over a scripted engine - no network, real rapidfuzz."""

from __future__ import annotations

import wave

import numpy as np

from subtitle_checker.artifacts import (
    AudioKind,
    AudioRegion,
    CheckResult,
    SubtitleEvent,
    Verdict,
)
from subtitle_checker.match.asr import (
    _to_wav,
    check_asr,
    skipped_lines,
    transcribe_lines,
    transcribe_missing,
)

SPEECH = [AudioRegion(0.0, 6.0, AudioKind.SPEECH)]
MUSIC = [AudioRegion(0.0, 6.0, AudioKind.MUSIC)]
AUDIO = np.zeros(16_000 * 8, dtype=np.float32)
LINE = SubtitleEvent(1.0, 4.0, "एक दो तीन चार", 0.9)
# A guideline-short one-line caption (0.75 s) - a real batch2 false-flag case,
# where the ASR garbled the tiny audio window while the subtitle was correct.
SHORT_LINE = SubtitleEvent(1.0, 1.75, "तोशु के तानों को", 0.99)


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


def test_short_line_mismatch_is_not_flagged() -> None:
    # Below MIN_MISMATCH_SPAN a low match is too likely a garbled short window
    # to accuse the subtitle, so it is held back from the flags.
    assert check_asr([SHORT_LINE], AUDIO, SPEECH, ScriptedAsr("पाँच छह सात आठ")) == []


def test_short_line_mismatch_still_shown_in_ledger() -> None:
    # Held back from the flags (not a TEXT_MISMATCH), but the words do not match
    # either - too short to verify. It becomes UNCHECKABLE so the editor still
    # sees the heard-vs-written, without it sitting in the "matching" list.
    rows = transcribe_lines([SHORT_LINE], AUDIO, SPEECH, ScriptedAsr("पाँच छह सात आठ"))
    assert len(rows) == 1
    assert rows[0].verdict is Verdict.UNCHECKABLE
    assert rows[0].heard_text == "पाँच छह सात आठ"
    assert "too short" in rows[0].reason


def test_skipped_lines_reason_per_gate() -> None:
    regions = [
        AudioRegion(0.0, 10.0, AudioKind.SPEECH),
        AudioRegion(10.0, 20.0, AudioKind.MUSIC),
    ]
    events = [
        SubtitleEvent(1.0, 3.0, "पहली सही लाइन है", 0.9),  # claimed by a result row
        SubtitleEvent(4.0, 6.0, "गड़बड़ पाठ यहाँ है", 0.2),  # untrusted OCR
        SubtitleEvent(7.0, 9.0, "दो शब्द", 0.9),  # too short
        SubtitleEvent(12.0, 14.0, "गाने के ऊपर वाली लाइन", 0.9),  # over music
        SubtitleEvent(6.2, 6.9, "सुनाई नहीं दिया कुछ", 0.9),  # trusted, never heard
    ]
    results = [
        CheckResult(1.0, 3.0, Verdict.OK, "ok", "पहली सही लाइन है", "पहली सही लाइन है")
    ]
    reasons = {e.text: reason for e, reason in skipped_lines(events, results, regions)}
    assert "पहली सही लाइन है" not in reasons
    assert "confidence 0.20" in reasons["गड़बड़ पाठ यहाँ है"]
    assert reasons["दो शब्द"] == "too short to compare word-for-word"
    assert reasons["गाने के ऊपर वाली लाइन"] == "no speech under this line"
    assert reasons["सुनाई नहीं दिया कुछ"] == "nothing was transcribed for this line"


def test_skipped_lines_without_regions_skips_speech_test() -> None:
    events = [SubtitleEvent(1.0, 2.0, "बिना क्षेत्र सूचना के", 0.1)]
    (_, reason), = skipped_lines(events, [], None)
    assert "confidence 0.10" in reason


def _missing(start: float, end: float) -> CheckResult:
    return CheckResult(start, end, Verdict.MISSING_SUBTITLE, "speech, no subtitle")


def test_missing_gets_transcript_filled() -> None:
    engine = ScriptedAsr("यहाँ कुछ बोला गया")
    (r,) = transcribe_missing([_missing(1.0, 4.0)], AUDIO, engine)
    assert r.verdict is Verdict.MISSING_SUBTITLE  # still a flag, not a verdict change
    assert r.heard_text == "यहाँ कुछ बोला गया"
    assert engine.calls == 1


def test_missing_with_too_few_heard_words_stays_blank() -> None:
    # a one-word blurt is as likely an ASR latch as a real line - keep it blank
    (r,) = transcribe_missing([_missing(1.0, 4.0)], AUDIO, ScriptedAsr("शब्द"))
    assert r.heard_text == ""


def test_missing_with_nothing_heard_stays_blank() -> None:
    (r,) = transcribe_missing([_missing(1.0, 4.0)], AUDIO, ScriptedAsr(""))
    assert r.heard_text == ""


def test_non_missing_flags_are_untouched() -> None:
    # only missing spans get transcribed; a mismatch or orphan is passed through
    engine = ScriptedAsr()  # would IndexError if called
    flags = [
        CheckResult(1.0, 4.0, Verdict.TEXT_MISMATCH, "diff", "लाइन", "सुना"),
        CheckResult(5.0, 7.0, Verdict.ORPHAN_SUBTITLE, "silence", "अनाथ"),
    ]
    out = transcribe_missing(flags, AUDIO, engine)
    assert out == flags
    assert engine.calls == 0


def test_to_wav_is_valid_pcm() -> None:
    buf = _to_wav(np.zeros(1600, dtype=np.float32))
    with wave.open(buf, "rb") as w:
        assert w.getframerate() == 16_000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 1600


class _Resp:
    def __init__(self, status: int, headers: dict | None = None) -> None:
        self.status_code = status
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        import requests

        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self) -> dict:
        return {"transcript": " नमस्ते "}


def test_sarvam_retries_on_429_then_succeeds(monkeypatch) -> None:
    import requests

    from subtitle_checker.match.asr import SarvamAsr

    calls = {"n": 0}
    slept: list[float] = []

    def fake_post(*_args, **_kwargs) -> _Resp:
        calls["n"] += 1
        return _Resp(429 if calls["n"] < 3 else 200)

    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))

    assert SarvamAsr().transcribe(np.zeros(1600, dtype=np.float32)) == "नमस्ते"
    assert calls["n"] == 3  # two 429s retried, third succeeded
    assert len(slept) == 2


def test_sarvam_raises_after_exhausting_retries(monkeypatch) -> None:
    import requests

    from subtitle_checker.match.asr import SarvamAsr

    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(429))
    monkeypatch.setattr("time.sleep", lambda s: None)

    try:
        SarvamAsr().transcribe(np.zeros(1600, dtype=np.float32))
    except requests.HTTPError as exc:
        assert "429" in str(exc)
    else:
        raise AssertionError("expected HTTPError after retries exhausted")


def test_retry_after_honours_header_then_backoff() -> None:
    from subtitle_checker.match.asr import SARVAM_BACKOFF_S, _retry_after

    assert _retry_after(_Resp(429, {"Retry-After": "7"}), attempt=0) == 7.0
    assert _retry_after(_Resp(429), attempt=2) == SARVAM_BACKOFF_S * 4
