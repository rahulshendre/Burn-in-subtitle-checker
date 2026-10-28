"""The numbers in a report must agree with each other and say what they measure.

One card used to show four percentages for one measurement, and the bar said
"3 flagged" under a headline of "4 flag(s)". These tests render the reports the
way a user sees them and read the figures back out of the visible text.
"""

from __future__ import annotations

import re

import pytest

from subtitle_checker.artifacts import CheckResult, Verdict
from subtitle_checker.report.html import CONFIDENCE_EXPLAINED, accuracy_stats, render_report
from subtitle_checker.report.mistakes import render_mistakes


class _Evidence:
    def frame_png(self, t: float) -> bytes:
        return b"PNG"

    def audio_clip(self, start: float, end: float) -> tuple[bytes, str]:
        return b"MP3", "audio/mpeg"


_KINDS = {
    "mismatch": Verdict.TEXT_MISMATCH,
    "missing": Verdict.MISSING_SUBTITLE,
    "orphan": Verdict.ORPHAN_SUBTITLE,
    "unchecked": Verdict.UNCHECKABLE,
    "ok": Verdict.OK,
}


def _mix(**counts: int) -> list[CheckResult]:
    results, t = [], 0.0
    for kind, n in counts.items():
        for _ in range(n):
            results.append(
                CheckResult(t, t + 2.0, _KINDS[kind], "reason", subtitle_text="टेक्स्ट",
                            heard_text="टेक्स्ट" if kind == "ok" else "आवाज़ यहाँ है")
            )
            t += 3.0
    return results


def _visible(page: str) -> str:
    page = re.sub(r"<style.*?</style>|<script.*?</script>", "", page, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page).replace("&middot;", "|"))


def _number(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    assert match, f"{pattern!r} not found in: {text[:300]}"
    return int(match.group(1))


def test_the_reported_case_ten_lines_one_cannot_be_checked():
    # The review's example: 6 matched, 3 flagged, 1 could not be checked.
    results = _mix(mismatch=1, missing=1, orphan=1, unchecked=1, ok=6)
    full = _visible(render_report(results, _Evidence(), title="Demo"))
    mistakes = _visible(render_mistakes(results, _Evidence(), title="Demo"))

    for page in (full, mistakes):
        assert "10 lines" in page
        assert "6 matched" in page
        assert "3 flagged" in page
        assert "1 could not be checked" in page
        # 6 of the 9 lines that could be checked, and the page says so
        assert "67% match rate (of the 9 lines that could be checked)" in page
    assert "3 flag(s) for review (and 1 line that could not be checked)" in full
    assert "3 subtitle mistakes to review (and 1 line that could not be checked)" in mistakes


@pytest.mark.parametrize(
    "counts",
    [
        dict(mismatch=1, ok=4),
        dict(mismatch=2, missing=1, orphan=3, ok=2),
        dict(unchecked=2, ok=5),
        dict(unchecked=3),
        dict(mismatch=1, unchecked=1),
        dict(ok=7),
    ],
)
def test_flag_counts_agree_everywhere_on_both_pages(counts):
    results = _mix(**counts)
    stats = accuracy_stats(results)
    flagged = counts.get("mismatch", 0) + counts.get("missing", 0) + counts.get("orphan", 0)

    assert stats["matched"] + stats["flagged"] + stats["uncheckable"] == stats["total"]
    assert stats["flagged"] == flagged

    full = _visible(render_report(results, _Evidence(), title="Demo"))
    mistakes = _visible(render_mistakes(results, _Evidence(), title="Demo"))
    assert _number(r"(\d+) flagged", full) == flagged
    assert _number(r"(\d+) flagged", mistakes) == flagged
    if flagged:
        assert _number(r"(\d+) flag\(s\) for review", full) == flagged
        assert _number(r"(\d+) subtitle mistakes? to review", mistakes) == flagged
    else:
        assert "No flags" in full and "No mistakes found" in mistakes


def test_the_match_rate_says_when_lines_were_left_out_of_it():
    with_unchecked = _visible(render_report(_mix(ok=1, unchecked=1), _Evidence(), title="Demo"))
    without = _visible(render_report(_mix(ok=1, mismatch=1), _Evidence(), title="Demo"))
    assert "of the 1 line that could be checked" in with_unchecked
    assert "could be checked" not in without


def test_a_card_shows_one_number_and_the_inputs_are_not_repeated():
    flag = CheckResult(
        1.0, 3.0, Verdict.TEXT_MISMATCH, "The audio doesn't match the subtitle",
        subtitle_text="हमारे गाँव में पानी", heard_text="हमारे शहर में बिजली की कमी नहीं है",
        score=0.43, ocr_confidence=0.91, combined_score=57.0,
    )
    for page in (
        render_report([flag], _Evidence(), title="Demo"),
        render_mistakes([flag], _Evidence(), title="Demo"),
    ):
        card = re.search(r'<article class="card".*?</article>', page, re.S).group(0)
        text = _visible(card)
        assert text.count("%") == 1, text
        assert "Confidence 57%" in text
        assert "match score" not in text
        assert "OCR" not in text and "ASR" not in text


def test_the_ledger_shows_the_same_single_number():
    ok = CheckResult(
        1.0, 3.0, Verdict.OK, "The audio matches the subtitle", subtitle_text="ठीक है",
        heard_text="ठीक है", score=0.95, ocr_confidence=0.6, combined_score=84.5,
    )
    text = _visible(render_report([ok], _Evidence(), title="Demo"))
    assert "Confidence" in text
    assert "84%" in text or "85%" in text
    assert "OCR" not in text and "ASR" not in text


def test_a_line_with_no_audio_signal_gets_no_made_up_number():
    flag = CheckResult(1.0, 3.0, Verdict.ORPHAN_SUBTITLE, "r", subtitle_text="टेक्स्ट")
    card = re.search(r'<article class="card".*?</article>',
                     render_report([flag], _Evidence(), title="Demo"), re.S).group(0)
    assert "Confidence" not in card and "%" not in _visible(card)


def test_confidence_is_explained_on_both_pages():
    results = _mix(mismatch=1, ok=1)
    for page in (
        render_report(results, _Evidence(), title="Demo"),
        render_mistakes(results, _Evidence(), title="Demo"),
    ):
        assert CONFIDENCE_EXPLAINED in _visible(page)
