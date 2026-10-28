"""Tests for turning pipeline failures into messages a user can act on."""

from __future__ import annotations

import errno
import socket
import subprocess

import httpx
import pytest
import requests

from subtitle_checker.report import errors
from subtitle_checker.report.errors import plain_error, plain_exit_message, record_failure
from subtitle_checker.report.keycheck import RUN_REJECTED


class _Status(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"headers: {{'secret': 'x'}}, status_code: {status_code}, body: raw")
        self.status_code = status_code


def _ffprobe_failure() -> subprocess.CalledProcessError:
    cmd = ["ffprobe", "-v", "error", r"C:\Users\user\AppData\Local\Temp\uploads\clip.mp4"]
    return subprocess.CalledProcessError(1, cmd)


@pytest.mark.parametrize("tool", ["ffprobe", "ffmpeg", "/opt/homebrew/bin/ffprobe", "ffmpeg.exe"])
def test_a_video_the_tools_cannot_read_is_explained(tool):
    exc = subprocess.CalledProcessError(1, [tool, "-i", "clip.mp4"])
    assert plain_error(exc) == errors.VIDEO_UNREADABLE


def test_a_stuck_video_tool_is_explained():
    exc = subprocess.TimeoutExpired(["ffmpeg", "-i", "clip.mp4"], 600)
    assert plain_error(exc) == errors.VIDEO_TOO_SLOW


def test_the_real_ffprobe_failure_never_shows_the_command_line():
    message = plain_error(_ffprobe_failure())
    assert message == errors.VIDEO_UNREADABLE
    assert "ffprobe" not in message and "Users" not in message


def test_another_failing_command_is_not_called_a_video_problem():
    assert plain_error(subprocess.CalledProcessError(1, ["git", "status"])) == errors.UNEXPECTED


def test_a_rejected_key_points_at_the_change_button():
    assert plain_error(_Status(403)) == RUN_REJECTED
    assert plain_error(_Status(401)) == RUN_REJECTED
    assert "Change API key" in RUN_REJECTED


def test_rate_limits_and_server_errors_are_told_apart():
    assert plain_error(_Status(429)) == errors.SARVAM_BUSY
    assert plain_error(_Status(500)) == errors.SARVAM_DOWN
    assert plain_error(_Status(503)) == errors.SARVAM_DOWN


def test_requests_style_http_errors_are_read_too():
    err = requests.HTTPError("503 Server Error")
    err.response = type("R", (), {"status_code": 503})()
    assert plain_error(err) == errors.SARVAM_DOWN


@pytest.mark.parametrize(
    "exc",
    [
        requests.ConnectionError("down"),
        requests.Timeout("slow"),
        httpx.ConnectError("down"),
        httpx.ReadTimeout("slow"),
        socket.gaierror(8, "nodename nor servname provided"),
        ConnectionResetError("reset"),
        TimeoutError("timed out"),
    ],
)
def test_no_connection_is_explained(exc):
    assert plain_error(exc) == errors.NO_INTERNET


def test_a_wrapped_network_failure_is_still_recognised():
    try:
        try:
            raise requests.ConnectionError("down")
        except requests.ConnectionError as cause:
            raise RuntimeError("could not transcribe") from cause
    except RuntimeError as wrapped:
        assert plain_error(wrapped) == errors.NO_INTERNET


def test_a_full_disk_and_low_memory_are_explained():
    assert plain_error(OSError(errno.ENOSPC, "No space left on device")) == errors.DISK_FULL
    assert plain_error(MemoryError()) == errors.OUT_OF_MEMORY


@pytest.mark.parametrize("exc", [ValueError("bad shape (3, 0)"), KeyError("blocks"), _Status(400)])
def test_anything_unknown_gets_the_generic_message_without_technical_text(exc):
    message = plain_error(exc)
    assert message == errors.UNEXPECTED
    assert "app.log" in message
    assert "shape" not in message and "blocks" not in message and "400" not in message


def test_an_empty_subtitle_file_is_explained():
    log = ["note: x", "no subtitle cues parsed from: C:\\u\\script.srt"]
    assert plain_exit_message(log) == errors.SRT_EMPTY


def test_an_unexplained_error_exit_gets_the_generic_message():
    assert plain_exit_message(["something odd"]) == errors.UNEXPECTED
    assert plain_exit_message([]) == errors.UNEXPECTED


def test_the_technical_detail_is_kept_in_app_log(tmp_path):
    try:
        raise ValueError("bad shape (3, 0)")
    except ValueError as exc:
        record_failure(tmp_path, "clip.mp4", exc, ["a log line"])
    record_failure(tmp_path, "second.mp4", None, ["first", "last line of cli output"])

    text = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert "check failed for clip.mp4" in text
    assert "ValueError: bad shape (3, 0)" in text
    assert "Traceback" in text
    assert "check failed for second.mp4" in text  # appended, the first entry kept
    assert "last line of cli output" in text


def test_a_log_that_cannot_be_written_does_not_hide_the_message(tmp_path):
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("x")
    record_failure(blocker, "clip.mp4", ValueError("x"), [])  # must not raise
