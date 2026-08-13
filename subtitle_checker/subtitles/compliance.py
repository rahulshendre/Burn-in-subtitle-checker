"""Guideline-compliance check for burned-in subtitles.

PlanetRead sets presentation guidelines for its Hindi subtitles - font, sizes,
spacing, line and character limits. Most are authoring-tool settings (font name,
point sizes, word and line spacing) that a finished, burned-in video does not
carry: the checker sees rendered pixels, not the project that made them, so it
cannot recover a point size or a font name after the fact. Two rules, though, are
measurable straight off the frame - how many characters a line holds, and how
many lines stack on screen - and those are exactly the ones an editor slips on.

This grades each readable subtitle against those two caps and reports how many
lines follow the standard, so a channel gets a concrete pass/fail against its own
guidelines instead of a subjective read.

The line count is measured from the subtitle mask (the OCR text joins wrapped
lines into one string, so it cannot be split back into visual lines), and drives
the two-lines-on-screen cap. The character cap is checked against the whole
caption's on-screen budget - two lines at the per-line limit - rather than per
visual line: recovering the exact wrap point of a burned-in caption from pixels
is not reliable, so the tool checks the total it can measure well instead of
guessing a split it cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from subtitle_checker.artifacts import SubtitleEvent

# The two caps from PlanetRead's Hindi subtitle guidelines the tool can verify
# from a burned-in frame.
MAX_CHARS_PER_LINE = 35
MAX_LINES = 2
# The most characters a compliant caption can hold on screen: two lines at the
# per-line cap. Checked against the whole caption because a burned-in caption's
# wrap point cannot be recovered reliably from pixels.
MAX_CHARS_ON_SCREEN = MAX_CHARS_PER_LINE * MAX_LINES


@dataclass
class LineCompliance:
    """One subtitle line graded against the checkable guidelines."""

    start: float
    end: float
    text: str
    char_count: int
    line_count: int | None
    chars_ok: bool  # against the whole-caption budget, always measurable from the text
    lines_ok: bool | None  # None when the line count could not be measured from the mask

    @property
    def compliant(self) -> bool:
        """True unless a measured check failed - an unmeasured check makes no claim."""
        return self.chars_ok is not False and self.lines_ok is not False

    def failures(self) -> list[str]:
        """Human-readable reasons this line breaks the guidelines, if any."""
        reasons = []
        if self.lines_ok is False:
            reasons.append(f"{self.line_count} lines on screen (max {MAX_LINES})")
        if self.chars_ok is False:
            reasons.append(f"{self.char_count} characters (max {MAX_CHARS_ON_SCREEN})")
        return reasons


@dataclass
class VideoCompliance:
    """How a whole video's subtitles measure against the guidelines."""

    graded: int  # readable lines graded
    compliant: int  # lines that pass every measured check
    chars_pass: int
    chars_measured: int
    lines_pass: int
    lines_measured: int
    violations: list[LineCompliance] = field(default_factory=list)

    @property
    def share(self) -> float:
        """Fraction of graded lines that follow the guidelines, 0..1."""
        return self.compliant / self.graded if self.graded else 0.0


def line_compliance(event: SubtitleEvent) -> LineCompliance:
    """Grade one event against the character and line-count caps.

    Characters are checked against the whole-caption budget (two lines at the
    per-line cap), always measurable from the OCR text. The line-count cap needs
    the mask measurement; when that is missing the tool makes no claim on it.
    """
    text = event.text.strip()
    lines = event.line_count
    return LineCompliance(
        start=event.start,
        end=event.end,
        text=text,
        char_count=len(text),
        line_count=lines,
        chars_ok=len(text) <= MAX_CHARS_ON_SCREEN,
        lines_ok=lines <= MAX_LINES if lines else None,
    )


def check_compliance(events: list[SubtitleEvent]) -> VideoCompliance | None:
    """Grade a video's readable subtitles against the checkable guidelines.

    Only lines OCR could read are graded - a band with no readable caption has no
    text to measure. Returns None when nothing readable was found.
    """
    lines = [line_compliance(e) for e in events if e.text.strip()]
    if not lines:
        return None
    return VideoCompliance(
        graded=len(lines),
        compliant=sum(1 for line in lines if line.compliant),
        chars_pass=sum(1 for line in lines if line.chars_ok is True),
        chars_measured=sum(1 for line in lines if line.chars_ok is not None),
        lines_pass=sum(1 for line in lines if line.lines_ok is True),
        lines_measured=sum(1 for line in lines if line.lines_ok is not None),
        violations=[line for line in lines if not line.compliant],
    )
