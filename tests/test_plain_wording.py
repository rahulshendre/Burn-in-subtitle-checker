"""The reports are read by editors and production staff: no engineering terms.

Every verdict is rendered into both reports and the visible text is checked, and
the reason strings the checks write are scanned at the source so a new technical
phrase cannot slip back in.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from subtitle_checker.artifacts import CheckResult, SubtitleEvent, Verdict
from subtitle_checker.report.html import render_report
from subtitle_checker.report.mistakes import render_mistakes

_JARGON = (
    "OCR", "ASR", "Orphan", "orphan", "Uncheckable", "uncheckable", "mismatch",
    "transcri", "verifiable", "Heard (", "Written (", "beneath",
)


class _Evidence:
    def frame_png(self, t: float) -> bytes:
        return b"PNG"

    def audio_clip(self, start: float, end: float) -> tuple[bytes, str]:
        return b"MP3", "audio/mpeg"


def _every_verdict() -> list[CheckResult]:
    return [
        CheckResult(1.0, 3.0, Verdict.TEXT_MISMATCH, "The audio doesn't match the subtitle",
                    subtitle_text="हमारे गाँव में", heard_text="हमारे शहर में बिजली की कमी नहीं है",
                    score=0.43, ocr_confidence=0.9, combined_score=57.0),
        CheckResult(4.0, 6.0, Verdict.MISSING_SUBTITLE, "Someone is speaking but there is no subtitle",
                    heard_text="आप सब का धन्यवाद"),
        CheckResult(7.0, 9.0, Verdict.ORPHAN_SUBTITLE, "There is a subtitle here but no speech",
                    subtitle_text="चलो अब आगे चलते हैं"),
        CheckResult(10.0, 12.0, Verdict.TIMING_DRIFT, "The subtitle appears late",
                    subtitle_text="ठीक है"),
        CheckResult(13.0, 15.0, Verdict.UNCHECKABLE, "Music is playing, so the speech couldn't be checked",
                    subtitle_text="गाना बज रहा है"),
        CheckResult(16.0, 18.0, Verdict.OK, "The audio matches the subtitle",
                    subtitle_text="कैसे मिला पायेंगे", heard_text="कैसे मिला पायेंगे",
                    score=0.95, ocr_confidence=0.6, combined_score=84.5),
    ]


def _visible(page: str) -> str:
    page = re.sub(r"<style.*?</style>|<script.*?</script>", "", page, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page))


def _jargon_in(text: str) -> list[str]:
    return [word for word in _JARGON if word in text]


@pytest.mark.parametrize("source_label", ["OCR", "SRT"])
def test_full_report_uses_plain_words(source_label):
    skipped = [(SubtitleEvent(20.0, 21.0, "छोटी लाइन", 0.9), "Too short to check")]
    page = render_report(
        _every_verdict(), _Evidence(), title="Demo", skipped=skipped, source_label=source_label
    )
    assert _jargon_in(_visible(page)) == []


def test_mistakes_page_uses_plain_words():
    page = render_mistakes(_every_verdict(), _Evidence(), title="Demo")
    assert _jargon_in(_visible(page)) == []


def test_every_kind_of_flag_has_a_plain_label():
    text = _visible(render_report(_every_verdict(), _Evidence(), title="Demo"))
    for label in (
        "Subtitle doesn't match the audio",
        "Speech with no subtitle",
        "Subtitle with no speech",
        "Subtitle timing is off",
        "Couldn't check",
        "What the audio says",
        "Subtitle on screen",
    ):
        assert label in text, label


def test_the_ledger_and_cards_name_the_subtitle_source_plainly():
    on_screen = _visible(render_report(_every_verdict(), _Evidence(), title="Demo"))
    from_file = _visible(
        render_report(_every_verdict(), _Evidence(), title="Demo", source_label="SRT")
    )
    assert "Subtitle on screen" in on_screen and "Subtitle in the file" not in on_screen
    assert "Subtitle in the file" in from_file and "Subtitle on screen" not in from_file


def test_reason_strings_in_the_checks_are_plain():
    banned = ("ocr", "asr", "orphan", "uncheckable", "mismatch", "transcrib", "verifi",
              "beneath", "word-for-word")
    found = []
    for path in Path(__file__).parent.parent.joinpath("subtitle_checker", "match").glob("*.py"):
        for reason in re.findall(r'reason\s*=\s*f?"([^"]+)"', path.read_text(encoding="utf-8")):
            found.append((path.name, reason))
            hits = [word for word in banned if word in reason.lower()]
            assert not hits, f"{path.name}: {reason!r} uses {hits}"
            assert "%" not in reason, f"{path.name}: {reason!r} repeats the match figure"
    assert len(found) >= 10, "the scan should see the checks' reason strings"
