"""Parse an authored subtitle file (SRT / VTT) into subtitle events.

The audio-vs-script pipeline takes the authored subtitle file as its reference
instead of reading burned-in pixels, so there is no OCR and every line is fully
trusted: each event carries confidence 1.0 and no legibility measures (those
come from pixels this pipeline never looks at).

Tolerant of the quirks real files carry: a WEBVTT header and NOTE blocks, a
byte-order mark, comma or dot as the millisecond separator, and the extra blank
lines a docx-to-txt export leaves between cues. Cues are split on blank-line
boundaries, so a run of blank lines between two cues collapses to one gap.
"""

from __future__ import annotations

import re
from pathlib import Path

from subtitle_checker.artifacts import SubtitleEvent

# HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT); hours and millis may be short.
_TIME = r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
_TIMELINE = re.compile(rf"{_TIME}\s*-->\s*{_TIME}")
# One or more blank (whitespace-only) lines separate cues; collapses the double
# blanks a docx export leaves behind.
_CUE_SPLIT = re.compile(r"\n[ \t]*\n")


def _seconds(hh: str, mm: str, ss: str, ms: str) -> float:
    """Timestamp fields to seconds; pads a short millisecond field (7 -> 700 ms)."""
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms.ljust(3, "0")) / 1000.0


def parse_script(path: Path) -> list[SubtitleEvent]:
    """Read an SRT/VTT file into time-ordered subtitle events (confidence 1.0).

    Lines with no parseable timeline (the WEBVTT header, NOTE blocks, a stray
    cue index) are skipped. A cue with no text or a non-positive span is dropped.
    """
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    events: list[SubtitleEvent] = []
    for block in _CUE_SPLIT.split(raw):
        lines = block.split("\n")
        timeline = next((ln for ln in lines if _TIMELINE.search(ln)), None)
        if timeline is None:
            continue
        m = _TIMELINE.search(timeline)
        assert m is not None  # guarded by the `next` above
        start = _seconds(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _seconds(m.group(5), m.group(6), m.group(7), m.group(8))
        after = lines[lines.index(timeline) + 1 :]
        text = " ".join(ln.strip() for ln in after if ln.strip()).strip()
        if end > start and text:
            events.append(SubtitleEvent(start=start, end=end, text=text, confidence=1.0))

    events.sort(key=lambda e: e.start)
    return events
