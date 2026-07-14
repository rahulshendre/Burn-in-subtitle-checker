"""Stage 4 renderer tests - pure HTML, no ffmpeg or network.

Evidence is faked so the renderer is exercised in isolation: the media it embeds
is opaque bytes here; the ffmpeg-backed extractor is tested separately.
"""

from __future__ import annotations

from subtitle_checker.artifacts import CheckResult, SubtitleEvent, Verdict
from subtitle_checker.report.html import render_report


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
        ),
        CheckResult(
            start=9.0,
            end=11.0,
            verdict=Verdict.OK,
            reason="matches",
            subtitle_text="कैसे मिला पायेंगे",
            heard_text="कैसे मिला पायेंगे",
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


def test_score_shown_as_percent():
    out = render_report(_sample(), FakeEvidence(), title="Demo")
    assert "match 21%" in out


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
