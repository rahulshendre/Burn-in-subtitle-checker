"""Burn a subtitle event list onto a video with ffmpeg + libass.

Produces the labelled test videos for the evaluation harness: plant defects
with defects.plan_defects, burn the mutated list here, and the output file
is a video whose subtitle errors are known exactly.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from subtitle_checker.artifacts import SubtitleEvent

# Noto Sans Devanagari is the usual Linux install; macOS falls back through
# fontconfig to its bundled Devanagari fonts (e.g. Kohinoor Devanagari).
DEFAULT_FONT = "Noto Sans Devanagari"

_STYLE_FIELDS = (
    "Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, "
    "Bold, Outline, Shadow, Alignment, MarginL, MarginR, MarginV"
)
_EVENT_FIELDS = "Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"


def _ass_timestamp(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    centis = round(seconds * 100)
    hours, rem = divmod(centis, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _escape(text: str) -> str:
    """Neutralise ASS control characters; subtitle text should carry none."""
    cleaned = text.replace("\\", "").replace("{", "(").replace("}", ")")
    return cleaned.replace("\n", r"\N")


def events_to_ass(
    events: list[SubtitleEvent],
    font: str = DEFAULT_FONT,
    font_size: int = 52,
    play_res: tuple[int, int] = (1280, 720),
) -> str:
    """Render subtitle events as an ASS script (white text, bottom-centred)."""
    style = (
        f"Style: Default,{font},{font_size},&H00FFFFFF,&H00000000,&H00000000,"
        "0,2,0,2,40,40,40"
    )
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {play_res[0]}",
        f"PlayResY: {play_res[1]}",
        "",
        "[V4+ Styles]",
        f"Format: {_STYLE_FIELDS}",
        style,
        "",
        "[Events]",
        f"Format: {_EVENT_FIELDS}",
    ]
    lines = []
    for event in sorted(events, key=lambda e: e.start):
        start, end = _ass_timestamp(event.start), _ass_timestamp(event.end)
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{_escape(event.text)}")
    return "\n".join(header + lines) + "\n"


def burn_subtitles(
    video: Path,
    events: list[SubtitleEvent],
    out: Path,
    font: str = DEFAULT_FONT,
) -> Path:
    """Re-encode ``video`` with ``events`` rendered as burned-in subtitles."""
    video = Path(video).resolve()
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        ass_path = Path(tmp) / "burn.ass"
        ass_path.write_text(events_to_ass(events, font=font), encoding="utf-8")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vf",
            "ass=burn.ass",
            "-c:a",
            "copy",
            str(out),
        ]
        # cwd=tmp lets the filter reference the .ass by bare name, dodging
        # ffmpeg filter-argument escaping for absolute paths
        proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-15:])
        raise RuntimeError(f"ffmpeg failed burning subtitles:\n{tail}")
    return out
