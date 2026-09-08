"""The mistakes-only page - just the errors, for a production team.

The full report (report.html) is the diagnostic surface: flags, the pass ledger,
legibility, compliance, skipped lines, all the evidence to judge a run. A
production team does not want to read all of that. Management's ask was plain:
now that the tool finds the mistakes, give a page that shows *only* the mistakes,
so the team can see what to fix and move on.

So this renders one focused list: every flagged line, worst first, each with the
subtitle as written, the tool's suggested correction (report.suggest), the frame,
and a snippet to hear it. No pass ledger, no legibility grade, no compliance, no
skipped section - none of the context the full report carries. If a run raised no
flags, the page says so in one line.

It reuses the full report's rendering helpers (frames, audio, timestamps, the
Devanagari matra-level diff, the verdict colours and the shared stylesheet) so
the two pages read as one tool, and there is a single place to change how a
subtitle frame or a matra difference is drawn.
"""

from __future__ import annotations

import html

from subtitle_checker.artifacts import CheckResult, Verdict
from subtitle_checker.report.html import (
    _AUDIO_PAD_S,
    _STYLE,
    _VERDICT_COLOR,
    _VERDICT_LABEL,
    _VERDICT_ORDER,
    Evidence,
    _accuracy_bar,
    _audio_html,
    _frame_html,
    _ts,
    _written_heard,
)
from subtitle_checker.report.suggest import suggest_correction


def render_mistakes(
    results: list[CheckResult],
    evidence: Evidence,
    *,
    title: str,
    generated: str | None = None,
    source_label: str = "OCR",
) -> str:
    """Render only the flagged mistakes into a self-contained HTML page.

    ``results`` may be the full ledger or flags only; either way just the
    non-OK rows appear, worst-first, each with a suggested correction. Passing
    lines, legibility, compliance and skipped lines are all left out - this page
    is the errors and nothing else.
    """
    from datetime import datetime

    flags = [r for r in results if r.verdict is not Verdict.OK]
    flags.sort(key=lambda r: (_VERDICT_ORDER.index(r.verdict), r.start))
    stamp = generated or datetime.now().strftime("%Y-%m-%d %H:%M")

    parts = [
        _head(title),
        _header(title, results, flags, stamp),
        _mistakes_section(flags, evidence, source_label),
        "</body></html>",
    ]
    return "\n".join(parts)


def _header(title: str, results: list[CheckResult], flags: list[CheckResult], stamp: str) -> str:
    n = len(flags)
    headline = (
        f"{n} subtitle mistake{'s' if n != 1 else ''} to review"
        if n
        else "No mistakes found - every checked line matches the audio"
    )
    return (
        f"<header><h1>{html.escape(title)}</h1>"
        f'<p class="sub">Subtitle mistakes to fix &middot; {html.escape(stamp)}</p>'
        f"{_accuracy_bar(results)}"
        f'<p class="headline">{headline}</p>'
        f"{_summary_chips(flags)}"
        '<p class="note">Only the lines that need fixing are listed. For each one: '
        "the subtitle as written, what the audio says, and the suggested correction.</p>"
        "</header>"
    )


def _summary_chips(flags: list[CheckResult]) -> str:
    """A one-glance count of the mistakes grouped by kind.

    The 'how many' and 'summary of mistakes' asks, answered at the top before any
    card: each mistake type that occurred, with its count, worst kind first.
    """
    if not flags:
        return ""
    counts = {v: 0 for v in _VERDICT_ORDER}
    for r in flags:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    chips = "".join(
        f'<span class="chip" style="background:{_VERDICT_COLOR[v]}">'
        f"{_VERDICT_LABEL[v]} {counts[v]}</span>"
        for v in _VERDICT_ORDER
        if v is not Verdict.OK and counts.get(v)
    )
    return f'<div class="chips">{chips}</div>'


def _mistakes_section(
    flags: list[CheckResult], evidence: Evidence, source_label: str
) -> str:
    if not flags:
        return '<section><p class="empty">Nothing to fix on this clip.</p></section>'
    cards = "\n".join(_mistake_card(r, evidence, source_label) for r in flags)
    return f'<section class="flags">{cards}</section>'


def _mistake_card(r: CheckResult, evidence: Evidence, source_label: str) -> str:
    color = _VERDICT_COLOR[r.verdict]
    frame = _frame_html(evidence.frame_png((r.start + r.end) / 2.0))
    audio = _audio_html(
        evidence.audio_clip(max(0.0, r.start - _AUDIO_PAD_S), r.end + _AUDIO_PAD_S)
    )
    written, heard = _written_heard(
        r,
        no_subtitle='<em class="none">- no subtitle -</em>',
        not_heard='<em class="none">- not transcribed -</em>',
    )
    return (
        f'<article class="card" style="border-left:6px solid {color}">'
        '<div class="card-head">'
        f'<span class="badge" style="background:{color}">{_VERDICT_LABEL[r.verdict]}</span>'
        f'<span class="tspan">{_ts(r.start)} - {_ts(r.end)}</span></div>'
        '<div class="card-body">'
        f'<div class="frame">{frame}</div>'
        '<div class="detail">'
        '<div class="texts">'
        f'<div class="col"><h4>Written ({source_label})</h4><p class="deva">{written}</p></div>'
        f'<div class="col"><h4>Heard (ASR)</h4><p class="deva">{heard}</p></div>'
        "</div>"
        f"{_suggestion_html(r)}{audio}"
        "</div></div></article>"
    )


def _suggestion_html(r: CheckResult) -> str:
    """The suggested text for a mistake, or an honest "could not tell".

    A corrected line and a filled-in missing line use different headings (see
    report.suggest), so the editor is not told a best-guess is a confirmed fix.
    """
    s = suggest_correction(r)
    if s.confident:
        conf_html = (
            f' <span class="suggest-conf">(match score: {s.asr_confidence:.0%})</span>'
            if s.asr_confidence is not None else ""
        )
        return (
            f'<div class="suggest"><h4>{html.escape(s.heading)}{conf_html}</h4>'
            f'<p class="deva suggest-text">{html.escape(s.text)}</p>'
            f'<p class="suggest-note">{html.escape(s.note)}</p></div>'
        )
    return f'<p class="suggest-none">{html.escape(s.note)}</p>'


def _head(title: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title>{_STYLE}{_EXTRA_STYLE}</head><body>"
    )


# A little extra styling for the suggestion block, layered on the shared sheet so
# the correction reads as the actionable part of each card.
_EXTRA_STYLE = """<style>
  .suggest { margin:.7rem 0 .5rem; padding:.6rem .8rem; background:#eef7f0;
             border:1px solid #cfe8d8; border-radius:5px; }
  .suggest h4 { margin:0 0 .25rem; font-size:.75rem; text-transform:uppercase;
                letter-spacing:.04em; color:#2a7; }
  .suggest-text { margin:0; font-weight:600; }
  .suggest-note { margin:.3rem 0 0; color:#678; font-size:.82rem; }
  .suggest-none { margin:.7rem 0 .5rem; padding:.55rem .8rem; background:#f6f6f6;
                  border:1px solid #e4e4e4; border-radius:5px; color:#777;
                  font-size:.88rem; }
</style>"""
