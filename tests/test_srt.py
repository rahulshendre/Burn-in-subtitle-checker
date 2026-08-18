"""Tests for the SRT/VTT script parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitle_checker.subtitles.srt import parse_script

_BASIC = """1
00:00:00,010 --> 00:00:01,680
अरे, माँ!

2
00:00:01,680 --> 00:00:03,340
थीं ना कि आज
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_parses_index_timeline_and_text(tmp_path: Path) -> None:
    events = parse_script(_write(tmp_path, "s.srt", _BASIC))
    assert len(events) == 2
    first = events[0]
    assert first.start == pytest.approx(0.01)
    assert first.end == pytest.approx(1.68)
    assert first.text == "अरे, माँ!"
    assert first.confidence == 1.0  # authored text is fully trusted, no OCR


def test_double_blank_lines_between_cues(tmp_path: Path) -> None:
    # A docx-to-txt export leaves two blank lines between cues; both cues survive.
    doubled = _BASIC.replace("\n\n", "\n\n\n")
    events = parse_script(_write(tmp_path, "s.txt", doubled))
    assert len(events) == 2


def test_multiline_cue_joins_with_space(tmp_path: Path) -> None:
    text = "1\n00:00:00,000 --> 00:00:02,000\nजल्दी चलो।\nचलो।\n"
    events = parse_script(_write(tmp_path, "s.srt", text))
    assert len(events) == 1
    assert events[0].text == "जल्दी चलो। चलो।"


def test_vtt_header_and_dot_separator(tmp_path: Path) -> None:
    text = "WEBVTT\n\nNOTE speaker map\n\n00:00:01.000 --> 00:00:02.500\nहाँ\n"
    events = parse_script(_write(tmp_path, "s.vtt", text))
    assert len(events) == 1  # header and NOTE skipped, dot millis parsed
    assert events[0].start == pytest.approx(1.0)
    assert events[0].end == pytest.approx(2.5)


def test_short_millisecond_field_is_padded(tmp_path: Path) -> None:
    text = "1\n00:00:00,01 --> 00:00:01,7\nहाँ\n"
    events = parse_script(_write(tmp_path, "s.srt", text))
    assert events[0].start == pytest.approx(0.01)  # ,01 -> 010 ms
    assert events[0].end == pytest.approx(1.7)  # ,7 -> 700 ms


def test_drops_empty_and_zero_span_cues(tmp_path: Path) -> None:
    text = (
        "1\n00:00:01,000 --> 00:00:02,000\n\n"  # no text
        "2\n00:00:03,000 --> 00:00:03,000\nहाँ\n"  # zero span
    )
    assert parse_script(_write(tmp_path, "s.srt", text)) == []


def test_no_cues_returns_empty(tmp_path: Path) -> None:
    assert parse_script(_write(tmp_path, "s.srt", "just prose, no timings\n")) == []


def test_events_sorted_by_start(tmp_path: Path) -> None:
    text = (
        "2\n00:00:05,000 --> 00:00:06,000\nदूसरा\n\n"
        "1\n00:00:01,000 --> 00:00:02,000\nपहला\n"
    )
    events = parse_script(_write(tmp_path, "s.srt", text))
    assert [e.text for e in events] == ["पहला", "दूसरा"]
