"""Stage 4 - render check results into a self-contained HTML report.

The report is the editor-facing surface of the whole pipeline: one HTML file,
no server, no external assets. Every flag becomes a card carrying the evidence a
non-technical reviewer needs to judge it in seconds - a frame from the subtitle,
the written text beside what the ASR heard, a short audio snippet to play, and
the verdict reason. Flags are grouped worst-first; when the run also carries the
per-line ledger (OK rows with heard-vs-written), a scan table lists the matching
lines so the editor can still catch the subtle word errors the pipeline
deliberately does not auto-flag - they sit below the OCR/ASR noise floor (see
match.asr).

Evidence (frames, audio) is supplied through the Evidence protocol so this
module stays pure and testable: the ffmpeg-backed implementation lives in
report.evidence, and tests pass a fake.
"""

from __future__ import annotations

import base64
import difflib
import html
import unicodedata
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from subtitle_checker.artifacts import CheckResult, SubtitleEvent, Verdict

if TYPE_CHECKING:
    from subtitle_checker.subtitles.legibility import VideoLegibility


class Evidence(Protocol):
    """Supplies the media a card embeds. Any method may return None if absent."""

    def frame_png(self, t: float) -> bytes | None:
        """A single video frame at time ``t`` seconds, PNG-encoded."""
        ...

    def audio_clip(self, start: float, end: float) -> tuple[bytes, str] | None:
        """Audio for the span [start, end] as (encoded bytes, MIME type)."""
        ...


# Worst first: how flags are grouped, and each verdict's colour and wording.
_VERDICT_ORDER = [
    Verdict.TEXT_MISMATCH,
    Verdict.MISSING_SUBTITLE,
    Verdict.ORPHAN_SUBTITLE,
    Verdict.TIMING_DRIFT,
    Verdict.UNCHECKABLE,
    Verdict.OK,
]
_VERDICT_LABEL = {
    Verdict.TEXT_MISMATCH: "Text mismatch",
    Verdict.MISSING_SUBTITLE: "Missing subtitle",
    Verdict.ORPHAN_SUBTITLE: "Orphan subtitle",
    Verdict.TIMING_DRIFT: "Timing drift",
    Verdict.UNCHECKABLE: "Uncheckable",
    Verdict.OK: "OK",
}
_VERDICT_COLOR = {
    Verdict.TEXT_MISMATCH: "#c0392b",
    Verdict.MISSING_SUBTITLE: "#d35400",
    Verdict.ORPHAN_SUBTITLE: "#8e44ad",
    Verdict.TIMING_DRIFT: "#16a085",
    Verdict.UNCHECKABLE: "#7f8c8d",
    Verdict.OK: "#27ae60",
}

# A little air around the subtitle span so the snippet plays with room, not clipped.
_AUDIO_PAD_S = 0.4


def render_report(
    results: list[CheckResult],
    evidence: Evidence,
    *,
    title: str,
    generated: str | None = None,
    skipped: list[tuple[SubtitleEvent, str]] | None = None,
    legibility: VideoLegibility | None = None,
) -> str:
    """Render check results into one self-contained HTML document.

    ``results`` may be flags only (as ``check`` saves them) or the full per-line
    ledger (OK rows carrying heard-vs-written). Flags become evidence cards
    worst-first; OK rows, if present, become a scan table below. ``skipped``
    lists detected lines the checker declined to verify, with reasons, so the
    editor sees they were passed over deliberately rather than missed.
    ``legibility``, when supplied, adds the whole-video presentation grade and
    its least legible lines - a channel-facing read separate from the mismatch
    check.
    """
    flags = [r for r in results if r.verdict is not Verdict.OK]
    oks = [r for r in results if r.verdict is Verdict.OK]
    flags.sort(key=lambda r: (_VERDICT_ORDER.index(r.verdict), r.start))
    oks.sort(key=lambda r: r.start)
    stamp = generated or datetime.now().strftime("%Y-%m-%d %H:%M")

    parts = [
        _head(title),
        _summary(title, results, stamp),
        _legibility_banner(legibility),
        _flags_section(flags, evidence),
        _ledger_section(oks, evidence),
        _legibility_section(legibility, evidence),
        _skipped_section(sorted(skipped or [], key=lambda s: s[0].start), evidence),
        _foot(),
    ]
    return "\n".join(parts)


def _summary(title: str, results: list[CheckResult], stamp: str) -> str:
    counts = {v: 0 for v in _VERDICT_ORDER}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    n_flags = sum(c for v, c in counts.items() if v is not Verdict.OK)
    chips = "".join(
        f'<span class="chip" style="background:{_VERDICT_COLOR[v]}">'
        f"{_VERDICT_LABEL[v]} {counts[v]}</span>"
        for v in _VERDICT_ORDER
        if counts.get(v)
    )
    headline = (
        f"{n_flags} flag(s) for review"
        if n_flags
        else "No flags - every checked line matches the audio"
    )
    return (
        f'<header><h1>{html.escape(title)}</h1>'
        f'<p class="sub">Burn-in subtitle checker · {html.escape(stamp)}</p>'
        f'<p class="headline">{headline}</p>'
        f'<div class="chips">{chips}</div>'
        "<p class=\"note\">Each card shows the subtitle frame, the written text beside "
        "what the ASR heard, and an audio snippet to verify. Subtle single-word errors "
        "are not auto-flagged - scan the heard-vs-written column below to catch them.</p>"
        "</header>"
    )


def _flags_section(flags: list[CheckResult], evidence: Evidence) -> str:
    if not flags:
        return '<section><p class="empty">No flags raised on this clip.</p></section>'
    cards = "\n".join(_card(r, evidence) for r in flags)
    return f'<section class="flags"><h2>Flags</h2>{cards}</section>'


def _card(r: CheckResult, evidence: Evidence) -> str:
    color = _VERDICT_COLOR[r.verdict]
    frame = _frame_html(evidence.frame_png((r.start + r.end) / 2.0))
    audio = _audio_html(
        evidence.audio_clip(max(0.0, r.start - _AUDIO_PAD_S), r.end + _AUDIO_PAD_S)
    )
    score = _score_html(r)
    written, heard = _written_heard(
        r, no_subtitle='<em class="none">- no subtitle -</em>',
        not_heard='<em class="none">- not transcribed -</em>',
    )
    return (
        f'<article class="card" style="border-left:6px solid {color}">'
        f'<div class="card-head">'
        f'<span class="badge" style="background:{color}">{_VERDICT_LABEL[r.verdict]}</span>'
        f'<span class="tspan">{_ts(r.start)} - {_ts(r.end)}</span>{score}</div>'
        f'<div class="card-body">'
        f'<div class="frame">{frame}</div>'
        f'<div class="detail">'
        f'<div class="texts">'
        f'<div class="col"><h4>Subtitle (written)</h4><p class="deva">{written}</p></div>'
        f'<div class="col"><h4>Heard (ASR)</h4><p class="deva">{heard}</p></div>'
        f"</div>"
        f'<p class="reason">{html.escape(r.reason)}</p>{audio}'
        f"</div></div></article>"
    )


def _score_breakdown(r: CheckResult) -> list[str]:
    """The OCR and audio confidences that fold into the combined score."""
    bits = []
    if r.ocr_confidence is not None:
        bits.append(f"OCR {r.ocr_confidence:.0%}")
    if r.score is not None:
        bits.append(f"audio {r.score:.0%}")
    return bits


def _score_html(r: CheckResult) -> str:
    """Card-head score: the one combined confidence with its OCR + audio parts."""
    if r.combined_score is None:
        return ""
    bits = _score_breakdown(r)
    detail = f' <small>{" &middot; ".join(bits)}</small>' if bits else ""
    return f'<span class="score">score {r.combined_score:.0f}{detail}</span>'


def _score_cell(r: CheckResult) -> str:
    """Ledger score column: the combined number, breakdown on hover."""
    if r.combined_score is None:
        return '<td class="score-cell">-</td>'
    tip = html.escape(" · ".join(_score_breakdown(r)))
    return f'<td class="score-cell" title="{tip}">{r.combined_score:.0f}</td>'


def _ledger_section(oks: list[CheckResult], evidence: Evidence) -> str:
    if not oks:
        return ""
    rows = "\n".join(_ledger_row(r, evidence) for r in oks)
    return (
        '<section class="ledger"><h2>Matching lines - heard vs written</h2>'
        '<p class="note">These lines passed the automatic check. Skim the two columns '
        "for spelling or word swaps the tool cannot flag on its own.</p>"
        '<table><thead><tr><th>Time</th><th>Frame</th><th>Written</th>'
        "<th>Heard (ASR)</th><th>Score</th><th>Audio</th></tr></thead><tbody>"
        f"{rows}</tbody></table></section>"
    )


def _ledger_row(r: CheckResult, evidence: Evidence) -> str:
    thumb = _thumb_html(evidence.frame_png((r.start + r.end) / 2.0))
    audio = _audio_html(
        evidence.audio_clip(max(0.0, r.start - _AUDIO_PAD_S), r.end + _AUDIO_PAD_S)
    )
    written, heard = _written_heard(r, no_subtitle="-", not_heard='<em class="none">-</em>')
    return (
        f'<tr><td class="tspan">{_ts(r.start)}</td><td class="thumb">{thumb}</td>'
        f'<td class="deva">{written}</td><td class="deva">{heard}</td>'
        f"{_score_cell(r)}<td>{audio}</td></tr>"
    )


# Legibility grade colour bands: clear, mixed, poor.
_LEG_CLEAR = 85.0
_LEG_MIXED = 60.0


def _legibility_color(score: float) -> str:
    if score >= _LEG_CLEAR:
        return "#27ae60"
    if score >= _LEG_MIXED:
        return "#d35400"
    return "#c0392b"


def _legibility_headline(score: float) -> str:
    if score >= _LEG_CLEAR:
        return "Clear - the subtitles read easily on screen"
    if score >= _LEG_MIXED:
        return "Mixed - some lines are hard to read"
    return "Poor - many lines wash out against the background"


def _legibility_banner(leg: VideoLegibility | None) -> str:
    """The whole-video legibility grade, front and centre for a channel."""
    if leg is None:
        return ""
    color = _legibility_color(leg.score)
    return (
        '<section class="legibility"><h2>Subtitle legibility</h2>'
        '<div class="leg-grade">'
        f'<span class="leg-score" style="color:{color}">{leg.score:.0f}'
        "<small>/100</small></span>"
        '<div class="leg-caption">'
        f'<p class="leg-headline">{_legibility_headline(leg.score)}</p>'
        f'<p class="note">How readable the subtitles are on screen across '
        f"{leg.line_count} line(s). Optical contrast between the text and its "
        "background stands in for how easily a viewer - or an OCR engine - can "
        "make out each word.</p></div></div></section>"
    )


def _legibility_section(leg: VideoLegibility | None, evidence: Evidence) -> str:
    """The least legible lines, so a channel sees exactly which captions fail."""
    if leg is None:
        return ""
    low = [line for line in leg.worst if line.score < 100]
    if not low:
        return ""
    rows = "\n".join(_legibility_row(line, evidence) for line in low)
    return (
        '<section class="leg-lines"><h2>Least legible lines</h2>'
        '<p class="note">The lowest-contrast subtitles in the video. A low score '
        "means the text and its background are close in brightness, so the words "
        "are hard to read.</p>"
        "<table><thead><tr><th>Time</th><th>Frame</th><th>Subtitle (OCR read)</th>"
        "<th>Legibility</th></tr></thead><tbody>"
        f"{rows}</tbody></table></section>"
    )


def _legibility_row(line, evidence: Evidence) -> str:
    thumb = _thumb_html(evidence.frame_png((line.start + line.end) / 2.0))
    text = html.escape(line.text.strip()) or '<em class="none">- unreadable -</em>'
    color = _legibility_color(line.score)
    return (
        f'<tr><td class="tspan">{_ts(line.start)}</td><td class="thumb">{thumb}</td>'
        f'<td class="deva">{text}</td>'
        f'<td class="score-cell" style="color:{color}">{line.score:.0f}</td></tr>'
    )


def _skipped_section(
    skipped: list[tuple[SubtitleEvent, str]], evidence: Evidence
) -> str:
    if not skipped:
        return ""
    rows = "\n".join(_skipped_row(e, reason, evidence) for e, reason in skipped)
    return (
        f'<section class="skipped"><h2>Skipped lines ({len(skipped)})</h2>'
        '<p class="note">These subtitles were detected but not verified '
        "word-for-word - checking them against the audio would risk a false "
        "alarm. Each row says why.</p>"
        '<table><thead><tr><th>Time</th><th>Frame</th><th>Subtitle (OCR read)</th>'
        "<th>Why skipped</th></tr></thead><tbody>"
        f"{rows}</tbody></table></section>"
    )


def _skipped_row(e: SubtitleEvent, reason: str, evidence: Evidence) -> str:
    thumb = _thumb_html(evidence.frame_png((e.start + e.end) / 2.0))
    text = html.escape(e.text.strip()) or '<em class="none">- unreadable -</em>'
    return (
        f'<tr><td class="tspan">{_ts(e.start)}</td><td class="thumb">{thumb}</td>'
        f'<td class="deva">{text}</td><td class="why">{html.escape(reason)}</td></tr>'
    )


def _written_heard(r: CheckResult, *, no_subtitle: str, not_heard: str) -> tuple[str, str]:
    """The written and heard cells for a result, diff-marked when both exist.

    When a line has both a subtitle and a transcript, the differing clusters are
    highlighted so a matra or word difference jumps out (this is where the editor
    catches the subtle errors the tool does not auto-flag). When one side is
    absent - a missing or orphan line - there is nothing to diff, so a plain
    placeholder stands in.
    """
    written, heard = r.subtitle_text.strip(), r.heard_text.strip()
    if written and heard:
        return _diff_texts(written, heard)
    return html.escape(written) or no_subtitle, html.escape(heard) or not_heard


def _diff_texts(written: str, heard: str) -> tuple[str, str]:
    """Escaped HTML for both texts with the differing clusters marked.

    Diffing is over grapheme clusters, not codepoints: a matra (vowel sign) is a
    combining mark, so हमारी vs हमारि differs in one cluster (री vs रि), and
    marking the whole akshara is what shows an editor the exact wrong vowel. A
    codepoint diff would highlight a bare, meaningless combining mark instead.
    """
    w = _grapheme_clusters(written)
    h = _grapheme_clusters(heard)
    matcher = difflib.SequenceMatcher(a=w, b=h, autojunk=False)
    written_out: list[str] = []
    heard_out: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        changed = tag != "equal"
        written_out.append(_mark("".join(w[i1:i2]), changed))
        heard_out.append(_mark("".join(h[j1:j2]), changed))
    return "".join(written_out), "".join(heard_out)


def _mark(text: str, changed: bool) -> str:
    if not text:
        return ""
    escaped = html.escape(text)
    return f'<mark class="diff">{escaped}</mark>' if changed else escaped


def _grapheme_clusters(text: str) -> list[str]:
    """Split Devanagari text into user-perceived characters (aksharas).

    A base letter carries its following combining marks (matras, nukta,
    anusvara), and a virama binds the next consonant into the same conjunct
    cluster, so each element is one akshara the reader sees as a unit.
    """
    clusters: list[str] = []
    for ch in text:
        if clusters and _binds_to_previous(clusters[-1], ch):
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


def _binds_to_previous(previous: str, ch: str) -> bool:
    if unicodedata.category(ch) in ("Mn", "Mc", "Me"):
        return True  # a combining mark joins the base it follows
    if ch in ("‍", "‌"):
        return True  # ZWJ / ZWNJ hold a conjunct together
    return previous.endswith("्")  # a virama pulls the next consonant in


def _frame_html(png: bytes | None) -> str:
    if not png:
        return '<div class="noframe">no frame</div>'
    uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    return f'<img alt="subtitle frame" src="{uri}">'


def _thumb_html(png: bytes | None) -> str:
    if not png:
        return '<span class="none">-</span>'
    uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    return f'<img class="thumb-img" alt="subtitle frame" src="{uri}">'


def _audio_html(clip: tuple[bytes, str] | None) -> str:
    if not clip:
        return ""
    data, mime = clip
    uri = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    return f'<audio controls preload="none" src="{uri}"></audio>'


def _ts(t: float) -> str:
    return f"{int(t) // 60}:{t % 60:04.1f}"


def _head(title: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title>{_STYLE}</head><body>"
    )


def _foot() -> str:
    return "</body></html>"


_STYLE = """<style>
  :root { --ink:#222; --line:#e2e2e2; }
  body { font-family:-apple-system,"Segoe UI",Roboto,sans-serif; color:var(--ink);
         max-width:960px; margin:2rem auto; padding:0 1rem; line-height:1.5; }
  h1 { font-size:1.5rem; margin:0 0 .2rem; }
  h2 { font-size:1.15rem; margin:2rem 0 .8rem; border-bottom:2px solid #2a6;
       padding-bottom:.3rem; }
  .sub { color:#777; margin:0 0 .8rem; font-size:.9rem; }
  .headline { font-weight:600; font-size:1.05rem; margin:.4rem 0; }
  .chips { margin:.5rem 0 1rem; }
  .chip { display:inline-block; color:#fff; border-radius:10px; padding:.1rem .6rem;
          font-size:.8rem; margin:0 .3rem .3rem 0; }
  .note { color:#666; font-size:.88rem; }
  .empty { color:#2a6; font-weight:600; }
  .card { border:1px solid var(--line); border-radius:6px; margin:0 0 1.2rem;
          overflow:hidden; background:#fff; }
  .card-head { display:flex; align-items:center; gap:.7rem; padding:.5rem .8rem;
               background:#fafafa; border-bottom:1px solid var(--line); }
  .badge { color:#fff; border-radius:4px; padding:.15rem .5rem; font-size:.8rem;
           font-weight:600; }
  .tspan { font-variant-numeric:tabular-nums; color:#555; font-size:.9rem; }
  .score { margin-left:auto; color:#333; font-size:.9rem; font-weight:600; }
  .score small { color:#888; font-weight:normal; font-size:.8rem; }
  .score-cell { text-align:center; font-weight:600; font-variant-numeric:tabular-nums;
                cursor:help; }
  .leg-grade { display:flex; align-items:center; gap:1.2rem; flex-wrap:wrap;
               background:#fafafa; border:1px solid var(--line); border-radius:6px;
               padding:1rem 1.2rem; }
  .leg-score { font-size:3rem; font-weight:700; line-height:1;
               font-variant-numeric:tabular-nums; }
  .leg-score small { font-size:1.1rem; color:#aaa; font-weight:600; }
  .leg-caption { flex:1; min-width:240px; }
  .leg-headline { font-weight:600; margin:0 0 .3rem; }
  .card-body { display:flex; gap:1rem; padding:.8rem; flex-wrap:wrap; }
  .frame img { width:320px; max-width:100%; border-radius:4px; display:block; }
  .noframe { width:320px; height:120px; background:#f0f0f0; color:#aaa;
             display:flex; align-items:center; justify-content:center; border-radius:4px; }
  .detail { flex:1; min-width:280px; }
  .texts { display:flex; gap:1rem; flex-wrap:wrap; }
  .col { flex:1; min-width:130px; }
  .col h4 { margin:0 0 .2rem; font-size:.75rem; text-transform:uppercase;
            letter-spacing:.04em; color:#888; }
  .deva { margin:0; font-size:1.05rem; }
  mark.diff { background:#ffe08a; color:inherit; border-radius:2px; padding:0 .05em; }
  .none { color:#aaa; }
  .reason { color:#555; font-size:.9rem; margin:.7rem 0 .5rem; }
  audio { width:100%; max-width:340px; height:34px; margin-top:.3rem; }
  table { width:100%; border-collapse:collapse; font-size:.95rem; }
  th, td { text-align:left; padding:.45rem .5rem; border-bottom:1px solid var(--line);
           vertical-align:top; }
  th { font-size:.75rem; text-transform:uppercase; letter-spacing:.04em; color:#888; }
  td.thumb { width:160px; }
  td.why { color:#777; font-size:.88rem; }
  .thumb-img { width:150px; border-radius:3px; display:block; }
  td audio { width:190px; height:30px; margin:0; }
</style>"""
