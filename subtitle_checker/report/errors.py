"""Turn a failed check into a message a person can act on.

The pipeline fails with developer errors (a subprocess command line, an HTTP
client dump). The app page shows the text below instead; anything not listed
here gets a generic message that points at app.log, where the full error is
written for whoever supports the user.
"""

from __future__ import annotations

import errno
import socket
import subprocess
import time
import traceback
from pathlib import Path

from subtitle_checker.report.keycheck import RUN_REJECTED, http_status, is_key_rejection

VIDEO_UNREADABLE = "This video file can't be read. Try exporting it again as MP4."
VIDEO_TOO_SLOW = (
    "Reading this video took too long. Try exporting it again as MP4, or use a shorter video."
)
NO_INTERNET = "Couldn't reach Sarvam. Check your internet connection and try again."
SARVAM_BUSY = "Sarvam is busy right now. Wait a minute and try again."
SARVAM_DOWN = "Sarvam had a problem on its side. Try again in a few minutes."
DISK_FULL = "This computer is out of disk space. Free some space and try again."
OUT_OF_MEMORY = (
    "The computer ran out of memory. Close other programs and try again, "
    "or use a shorter video."
)
SRT_EMPTY = "No subtitles could be read from that subtitle file. Check that it is a valid .srt file."
UNEXPECTED = (
    "Something went wrong. Please send the app.log file from your Subtitle Checker folder."
)

_VIDEO_TOOLS = ("ffmpeg", "ffprobe")


def _chain(exc: BaseException):
    """The error and the errors that caused it, so a wrapped failure is still seen."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _is_video_tool(exc: BaseException) -> bool:
    cmd = getattr(exc, "cmd", None)
    first = cmd[0] if isinstance(cmd, (list, tuple)) and cmd else cmd
    return isinstance(first, str) and Path(first).stem in _VIDEO_TOOLS


def _network_errors() -> tuple[type[BaseException], ...]:
    import requests

    kinds: tuple[type[BaseException], ...] = (
        requests.ConnectionError, requests.Timeout, ConnectionError, TimeoutError, socket.gaierror,
    )
    try:
        import httpx  # the Sarvam SDK's HTTP client

        kinds += (httpx.TransportError,)
    except ImportError:
        pass
    return kinds


def plain_error(exc: BaseException) -> str:
    """The message to show for a pipeline exception."""
    network = _network_errors()
    for err in _chain(exc):
        if is_key_rejection(err):
            return RUN_REJECTED
        if isinstance(err, subprocess.TimeoutExpired) and _is_video_tool(err):
            return VIDEO_TOO_SLOW
        if isinstance(err, subprocess.CalledProcessError) and _is_video_tool(err):
            return VIDEO_UNREADABLE
        status = http_status(err)
        if status == 429:
            return SARVAM_BUSY
        if status is not None and status >= 500:
            return SARVAM_DOWN
        if isinstance(err, OSError) and err.errno == errno.ENOSPC:
            return DISK_FULL
        if isinstance(err, MemoryError):
            return OUT_OF_MEMORY
        if isinstance(err, network):
            return NO_INTERNET
    return UNEXPECTED


def plain_exit_message(log: list[str]) -> str:
    """The message for a check that ended without a report and without raising.

    The command-line code prints its own explanation and returns an error code;
    only the cases a user can cause get their own wording.
    """
    text = " ".join(log).lower()
    if "no subtitle cues parsed" in text:
        return SRT_EMPTY
    return UNEXPECTED


def record_failure(home: Path, video: str, exc: BaseException | None, log: list[str]) -> None:
    """Append the full technical detail to app.log for whoever supports the user."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if exc is not None:
        detail = "".join(traceback.format_exception(exc))
    else:
        detail = "\n".join(log[-20:]) + "\n"
    try:
        home.mkdir(parents=True, exist_ok=True)
        with (home / "app.log").open("a", encoding="utf-8") as fh:
            fh.write(f"\n[{stamp}] check failed for {video}\n{detail}")
    except OSError:
        pass  # a full disk must not hide the message the user is about to see
