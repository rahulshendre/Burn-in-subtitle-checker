"""Smoke tests for the CLI entrypoint."""

import pytest

from subtitle_checker import __version__
from subtitle_checker.cli import main


def test_version_flag_prints_version(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help_and_succeeds(capsys: pytest.CaptureFixture) -> None:
    assert main([]) == 0
    assert "check" in capsys.readouterr().out


def test_check_rejects_missing_video(capsys: pytest.CaptureFixture) -> None:
    assert main(["check", "--video", "no_such_clip.mp4"]) == 2
    assert "not found" in capsys.readouterr().err
