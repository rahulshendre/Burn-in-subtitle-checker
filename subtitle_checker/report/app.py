"""Local app: pick a video, run the check, open the report - all in the browser.

The viewer in webui.py only shows reports that already exist. This is the tool
people in the org install and use themselves: the page uploads a video to a
server running on their own machine, the pipeline runs there, and the finished
report joins the list of past runs. Only the Sarvam calls leave the machine
(cropped subtitle bands for OCR, short audio windows for ASR); the video and
every report stay in the app's folder.

Stdlib only (http.server, threads), same as the viewer, so it packages into a
single on-device executable. One check runs at a time - a laptop cannot do
more, and the Sarvam rate limit would not allow it anyway. The page shell is a
pure function (render_app) so it is unit-tested without a socket.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

from subtitle_checker.report.webui import discover_reports

LANGUAGES = {"hi": "Hindi", "mr": "Marathi"}
KEY_ENV = "SARVAM_API_KEY"
_CONFIG = "config.json"
_CHUNK = 1 << 20

# The pipeline steps shown on the page, with where each sits on the progress bar
# (start, end percent; the upload fills 0-10 in the browser) and how long it
# usually takes per second of video. The timings come from a real full-Sarvam
# run (40 s Marathi clip: subtitles 49 s, audio 10 s, report 10 s), so the bar
# is an estimate that moves at the pace the pipeline really does.
STAGES = (
    ("Reading the subtitles", 10, 65, 1.25, 0.0),
    ("Checking the audio", 65, 85, 0.2, 5.0),
    ("Building the report", 85, 100, 0.2, 5.0),
)
# The pipeline reports progress by printing; these markers advance the stage, so
# the CLI stays the single code path with no progress plumbing.
_STAGE_MARKERS = (
    ("subtitle events", 1),
    ("script cue(s)", 1),
    ("flag(s)", 2),
)
_CAP = 0.95  # a step never shows full until the pipeline says it has finished


def _expected(stage: int, duration: float) -> float:
    _, _, _, per_second, fixed = STAGES[stage]
    return max(per_second * duration + fixed, 5.0)


def progress(stage: int, in_stage: float, duration: float) -> tuple[int, int]:
    """Estimated (percent, seconds left) for a job in `stage` for `in_stage` s.

    Within a step the bar fills at the expected pace but stops short of the next
    step until the pipeline actually moves on, so it never claims more than done.
    """
    _, start, end, _, _ = STAGES[stage]
    expected = _expected(stage, duration)
    frac = min(in_stage / expected, _CAP)
    percent = int(start + (end - start) * frac)
    left = max(expected - in_stage, 5.0) + sum(
        _expected(i, duration) for i in range(stage + 1, len(STAGES))
    )
    return percent, int(left)


def video_duration(video: Path) -> float:
    """Length of the video in seconds, or 0 when it cannot be read."""
    from subtitle_checker.subtitles.sampler import probe

    try:
        return probe(video).duration
    except Exception:  # an unreadable file fails later with a clearer message
        return 0.0


def default_home() -> Path:
    """Folder that holds uploads, finished runs and the saved key."""
    return Path.home() / "Subtitle Checker"


def load_key(home: Path) -> None:
    """Put a saved Sarvam key into the environment unless one is already set."""
    if os.environ.get(KEY_ENV):
        return
    try:
        key = json.loads((home / _CONFIG).read_text(encoding="utf-8")).get("sarvam_key", "")
    except (OSError, ValueError):
        return
    if key:
        os.environ[KEY_ENV] = key


def save_key(home: Path, key: str) -> None:
    """Save the key on this machine only (owner-readable) and use it right away."""
    home.mkdir(parents=True, exist_ok=True)
    path = home / _CONFIG
    path.write_text(json.dumps({"sarvam_key": key}), encoding="utf-8")
    path.chmod(0o600)
    os.environ[KEY_ENV] = key


def safe_name(name: str) -> str:
    """Reduce an uploaded filename to a plain, safe file name."""
    base = Path(name.replace("\\", "/")).name
    base = re.sub(r"[^\w.\- ]", "_", base).strip(" .")
    return base or "upload"


def resolve_inside(home: Path, rel: str) -> Path | None:
    """Map a URL path back to a file inside home; refuse anything outside it."""
    path = (home / unquote(rel)).resolve()
    if path.is_relative_to(home.resolve()) and path.is_file():
        return path
    return None


@dataclass
class Job:
    """The one check that is running (or last ran)."""

    state: str = "idle"  # idle | running | done | error
    stage: int = 0  # index into STAGES
    video: str = ""
    report: str = ""  # path relative to home, set when done
    error: str = ""
    started: float = 0.0
    stage_started: float = 0.0
    duration: float = 0.0  # video length, drives the time estimate
    log: list[str] = field(default_factory=list)

    def advance(self, stage: int) -> None:
        if stage > self.stage:
            self.stage, self.stage_started = stage, time.time()

    def as_json(self) -> dict:
        now = time.time()
        if self.state == "done":
            percent, left = 100, 0
        elif self.state == "running":
            percent, left = progress(self.stage, now - self.stage_started, self.duration)
        else:
            percent, left = 0, 0
        return {
            "state": self.state,
            "stage": self.stage,
            "stages": [s[0] for s in STAGES],
            "percent": percent,
            "left": left,
            "video": self.video,
            "report": self.report,
            "error": self.error,
            "elapsed": int(now - self.started) if self.started else 0,
            "log": self.log[-8:],
        }


class _StageWriter(io.TextIOBase):
    """Capture the pipeline's printed progress into the job."""

    def __init__(self, job: Job) -> None:
        self._job = job
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._job.log.append(line.rstrip())
            for marker, stage in _STAGE_MARKERS:
                if marker in line:
                    self._job.advance(stage)
        return len(text)


def run_check(job: Job, home: Path, video: Path, srt: Path | None, lang: str) -> None:
    """Run one check on the full Sarvam stack and record the outcome in job."""
    from subtitle_checker import cli

    out_dir = home / "runs" / f"{video.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    args = argparse.Namespace(
        video=str(video), lang=lang, out=str(out_dir), asr=True, ocr="sarvam-vision",
        script=str(srt) if srt else None,
    )
    writer = _StageWriter(job)
    try:
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            code = cli._run_script_check(args) if srt else cli._run_check(args)
        report = out_dir / f"{video.stem}_report.html"
        if code == 0 and report.exists():
            job.report = report.relative_to(home).as_posix()
            job.state = "done"
        else:
            job.state = "error"
            job.error = job.log[-1] if job.log else "the check did not produce a report"
    except Exception as exc:  # surface any pipeline failure on the page
        job.state, job.error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        # The report embeds the frames and audio it needs; the uploaded copies
        # would only fill the disk.
        for upload in (video, srt):
            if upload is not None:
                upload.unlink(missing_ok=True)


def past_runs(home: Path) -> list[dict]:
    """Finished reports under home, newest first, with their mistakes pages."""
    entries = discover_reports([home / "runs"]) if (home / "runs").is_dir() else []
    runs = []
    for e in entries:
        mistakes = e.path.with_name(e.path.name.replace("_report.html", "_mistakes.html"))
        runs.append({
            "label": e.label,
            "report": e.path.relative_to(home).as_posix(),
            "mistakes": mistakes.relative_to(home).as_posix() if mistakes.exists() else "",
            "when": e.path.stat().st_mtime,
        })
    runs.sort(key=lambda r: r["when"], reverse=True)
    return runs
