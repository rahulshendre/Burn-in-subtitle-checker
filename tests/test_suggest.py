"""Correction-suggestion logic - who gets a suggestion and who honestly does not.

Pure logic, no rendering: given a check result, decide whether the heard text is
worth offering as the correct wording. The gate is deliberately conservative
because ASR is the weak side of the pipeline (see report.suggest).
"""

from __future__ import annotations

from subtitle_checker.artifacts import CheckResult, Verdict
from subtitle_checker.report.suggest import suggest_correction


def _mismatch(heard: str) -> CheckResult:
    return CheckResult(
        start=5.0,
        end=7.5,
        verdict=Verdict.TEXT_MISMATCH,
        reason="heard words differ",
        subtitle_text="हम सब से नज़रे",
        heard_text=heard,
        score=0.21,
    )


def test_mismatch_with_clean_transcript_gets_a_suggestion():
    s = suggest_correction(_mismatch("हम सब की नज़र"))
    assert s.confident is True
    assert s.text == "हम सब की नज़र"


def test_suggestion_strips_surrounding_whitespace():
    s = suggest_correction(_mismatch("  हम सब की नज़र  "))
    assert s.text == "हम सब की नज़र"


def test_empty_transcript_declines_to_suggest():
    s = suggest_correction(_mismatch(""))
    assert s.confident is False
    assert s.text == ""
    assert "Could not determine" in s.note


def test_single_word_transcript_declines_to_suggest():
    # one stray heard word under a mismatch is as likely noise as a correction
    s = suggest_correction(_mismatch("नज़र"))
    assert s.confident is False
    assert s.text == ""


def test_structural_flags_have_no_correction():
    # a missing / orphan / uncheckable line has no mis-transcribed text to fix
    for verdict in (
        Verdict.MISSING_SUBTITLE,
        Verdict.ORPHAN_SUBTITLE,
        Verdict.UNCHECKABLE,
    ):
        r = CheckResult(1.0, 3.0, verdict, "gap", heard_text="कुछ आवाज़ यहाँ")
        s = suggest_correction(r)
        assert s.confident is False
        assert s.text == ""


def test_ok_line_gets_no_suggestion():
    r = CheckResult(
        9.0, 11.0, Verdict.OK, "matches",
        subtitle_text="कैसे मिला", heard_text="कैसे मिला",
    )
    assert suggest_correction(r).confident is False
