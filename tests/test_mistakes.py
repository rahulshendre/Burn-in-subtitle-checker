"""The mistakes-only page - errors and corrections, nothing else.

Evidence is faked so the renderer runs in isolation, the same way the full
report is tested (see test_report).
"""

from __future__ import annotations

from subtitle_checker.artifacts import CheckResult, Verdict
from subtitle_checker.report.mistakes import render_mistakes


class FakeEvidence:
    def frame_png(self, t: float) -> bytes:
        return b"PNGDATA"

    def audio_clip(self, start: float, end: float) -> tuple[bytes, str]:
        return b"MP3DATA", "audio/mpeg"


class NullEvidence:
    def frame_png(self, t: float) -> None:
        return None

    def audio_clip(self, start: float, end: float) -> None:
        return None


def _sample() -> list[CheckResult]:
    return [
        CheckResult(
            start=1.0, end=3.0,
            verdict=Verdict.MISSING_SUBTITLE,
            reason="speech with no subtitle",
        ),
        CheckResult(
            start=5.0, end=7.5,
            verdict=Verdict.TEXT_MISMATCH,
            reason="heard words differ",
            subtitle_text="हम सब से नज़रे",
            heard_text="हम सब की नज़र",
            score=0.21, ocr_confidence=0.8, combined_score=38.7,
        ),
        CheckResult(
            start=9.0, end=11.0,
            verdict=Verdict.OK,
            reason="matches",
            subtitle_text="कैसे मिला पायेंगे",
            heard_text="कैसे मिला पायेंगे",
            score=0.95, ocr_confidence=0.6, combined_score=84.5,
        ),
    ]


def test_document_is_self_contained_html():
    out = render_mistakes(_sample(), FakeEvidence(), title="Demo", generated="2026-09-01 12:00")
    assert out.startswith("<!DOCTYPE html>")
    assert out.rstrip().endswith("</body></html>")
    assert "data:image/png;base64," in out
    assert 'src="http' not in out and "<link" not in out


def test_only_flags_appear_ok_lines_excluded():
    out = render_mistakes(_sample(), FakeEvidence(), title="Demo")
    # two flags -> two cards; the OK line is not shown at all
    assert out.count('class="card"') == 2
    assert "कैसे मिला पायेंगे" not in out  # the OK line's text is absent


def test_no_full_report_sections():
    # this page is errors only - none of the full report's context blocks
    out = render_mistakes(_sample(), FakeEvidence(), title="Demo")
    assert "Matching lines" not in out
    assert "Subtitle legibility" not in out
    assert "Guideline compliance" not in out
    assert "Skipped lines" not in out


def test_mismatch_shows_a_suggested_correction():
    out = render_mistakes(_sample(), FakeEvidence(), title="Demo")
    assert "Suggested correction" in out
    assert "हम सब की नज़र" in out  # the heard text offered as the fix


def test_structural_flag_declines_a_suggestion():
    out = render_mistakes(_sample(), FakeEvidence(), title="Demo")
    # the missing-subtitle flag has no heard line, so it says so honestly
    assert "Could not determine the correct text" in out


def test_garbled_short_transcript_declines_a_suggestion():
    results = [
        CheckResult(
            2.0, 4.0, Verdict.TEXT_MISMATCH, "heard words differ",
            subtitle_text="बहुत लंबी पंक्ति यहाँ", heard_text="ठीक",
            score=0.2, ocr_confidence=0.8,
        )
    ]
    out = render_mistakes(results, FakeEvidence(), title="Demo")
    assert "Suggested correction" not in out
    assert "Could not determine the correct text" in out


def test_headline_counts_the_mistakes():
    out = render_mistakes(_sample(), FakeEvidence(), title="Demo")
    assert "2 subtitle mistakes to review" in out


def test_summary_groups_mistakes_by_kind():
    # the sample has one text mismatch and one missing subtitle - both counted
    out = render_mistakes(_sample(), FakeEvidence(), title="Demo")
    assert "Text mismatch 1" in out
    assert "Missing subtitle 1" in out
    # OK is never a mistake kind, so it is not chipped
    assert "OK 1" not in out


def test_no_summary_chips_when_clean():
    oks = [CheckResult(0.0, 2.0, Verdict.OK, "ok", "अ", "अ")]
    out = render_mistakes(oks, FakeEvidence(), title="Clean")
    assert '<div class="chips">' not in out


def test_no_mistakes_message():
    oks = [CheckResult(0.0, 2.0, Verdict.OK, "ok", "अ", "अ")]
    out = render_mistakes(oks, FakeEvidence(), title="Clean")
    assert "No mistakes found" in out
    assert out.count('class="card"') == 0


def test_matra_diff_is_marked_in_written_and_heard():
    out = render_mistakes(_sample(), FakeEvidence(), title="Demo")
    assert 'mark class="diff"' in out  # the mismatch's differing aksharas highlighted


def test_null_evidence_degrades():
    out = render_mistakes(_sample(), NullEvidence(), title="Demo")
    assert "no frame" in out
    assert "<audio" not in out


def test_script_source_label():
    out = render_mistakes(_sample(), FakeEvidence(), title="Demo", source_label="script")
    assert "Written (script)" in out
    assert "Written (OCR)" not in out
