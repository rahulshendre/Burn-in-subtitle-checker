"""Stage 4 renderer tests - pure HTML, no ffmpeg or network.

Evidence is faked so the renderer is exercised in isolation: the media it embeds
is opaque bytes here; the ffmpeg-backed extractor is tested separately.
"""

from __future__ import annotations

from subtitle_checker.artifacts import CheckResult, SubtitleEvent, Verdict
from subtitle_checker.report.html import _diff_texts, _grapheme_clusters, render_report


class FakeEvidence:
    """Returns fixed media for every span."""

    def frame_png(self, t: float) -> bytes:
        return b"PNGDATA"

    def audio_clip(self, start: float, end: float) -> tuple[bytes, str]:
        return b"MP3DATA", "audio/mpeg"


class NullEvidence:
    """No media available - the renderer must degrade gracefully."""

    def frame_png(self, t: float) -> None:
        return None

    def audio_clip(self, start: float, end: float) -> None:
        return None


def _sample() -> list[CheckResult]:
    return [
        CheckResult(
            start=1.0,
            end=3.0,
            verdict=Verdict.MISSING_SUBTITLE,
            reason="speech with no subtitle",
        ),
        CheckResult(
            start=5.0,
            end=7.5,
            verdict=Verdict.TEXT_MISMATCH,
            reason="heard words differ",
            subtitle_text="हम सब से नज़रे",
            heard_text="हम सब की नज़र",
            score=0.21,
            ocr_confidence=0.8,
            combined_score=38.7,
        ),
        CheckResult(
            start=9.0,
            end=11.0,
            verdict=Verdict.OK,
            reason="matches",
            subtitle_text="कैसे मिला पायेंगे",
            heard_text="कैसे मिला पायेंगे",
            score=0.95,
            ocr_confidence=0.6,
            combined_score=84.5,
        ),
    ]


def test_document_is_self_contained_html():
    out = render_report(_sample(), FakeEvidence(), title="Demo", generated="2026-07-10 12:00")
    assert out.startswith("<!DOCTYPE html>")
    assert '<meta charset="utf-8">' in out
    assert out.rstrip().endswith("</body></html>")
    # media is embedded, not linked
    assert "data:image/png;base64," in out
    assert "data:audio/mpeg;base64," in out
    assert 'src="http' not in out and "<link" not in out


def test_flags_render_worst_first_ok_excluded_from_cards():
    out = render_report(_sample(), FakeEvidence(), title="Demo")
    # two flags -> two cards; the OK row is not a card
    assert out.count('class="card"') == 2
    # text mismatch (worst) card comes before the missing-subtitle card
    assert out.index("Text mismatch") < out.index("Missing subtitle")


def test_ok_rows_go_to_ledger_table():
    out = render_report(_sample(), FakeEvidence(), title="Demo")
    assert "Matching lines" in out and "<table>" in out
    # the OK line's text lands in the ledger, its Devanagari preserved verbatim
    assert "कैसे मिला पायेंगे" in out


def test_combined_score_shown_on_card():
    out = render_report(_sample(), FakeEvidence(), title="Demo")
    # the fused score headlines the card, with the OCR + audio breakdown beside it
    assert "score 39" in out
    assert "OCR 80%" in out and "audio 21%" in out


def test_ledger_has_score_column():
    out = render_report(_sample(), FakeEvidence(), title="Demo")
    assert "<th>Score</th>" in out
    assert ">84<" in out  # the OK line's combined score, rounded for the cell


def test_missing_subtitle_placeholder_when_no_text():
    out = render_report(_sample(), FakeEvidence(), title="Demo")
    assert "- no subtitle -" in out  # MISSING flag has empty subtitle_text


def test_no_flags_message():
    oks = [CheckResult(0.0, 2.0, Verdict.OK, "ok", "अ", "अ")]
    out = render_report(oks, FakeEvidence(), title="Clean")
    assert "No flags" in out
    assert out.count('class="card"') == 0


def test_null_evidence_degrades():
    out = render_report(_sample(), NullEvidence(), title="Demo")
    assert "no frame" in out  # frame placeholder
    assert "<audio" not in out  # audio omitted, not a broken tag


def test_skipped_lines_render_with_reason():
    skipped = [
        (
            SubtitleEvent(12.0, 14.0, "और बुरा तो तब होगा", 0.32),
            "OCR read too unreliable to compare (confidence 0.32)",
        )
    ]
    out = render_report(_sample(), FakeEvidence(), title="Demo", skipped=skipped)
    assert "Skipped lines (1)" in out
    assert "और बुरा तो तब होगा" in out
    assert "confidence 0.32" in out


def test_no_skipped_section_when_none_given():
    out = render_report(_sample(), FakeEvidence(), title="Demo")
    assert "Skipped lines" not in out


def test_grapheme_clusters_keep_matra_with_base():
    # each akshara is one unit: र + ी is a single cluster, not two codepoints
    assert _grapheme_clusters("हमारी") == ["ह", "मा", "री"]


def test_matra_diff_marks_only_the_changed_akshara():
    written_html, heard_html = _diff_texts("हमारि", "हमारी")
    assert '<mark class="diff">रि</mark>' in written_html
    assert '<mark class="diff">री</mark>' in heard_html
    # the shared aksharas are left unmarked
    assert written_html.startswith("हमा")


def test_identical_texts_are_not_marked():
    written_html, heard_html = _diff_texts("कैसे मिला", "कैसे मिला")
    assert "mark" not in written_html and "mark" not in heard_html


def test_ledger_highlights_a_matra_difference():
    results = [
        CheckResult(
            0.0, 2.0, Verdict.OK, "matches",
            subtitle_text="हमारि वजह से", heard_text="हमारी वजह से",
        )
    ]
    out = render_report(results, FakeEvidence(), title="Demo")
    assert 'mark class="diff"' in out
