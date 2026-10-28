"""Tests for checking a Sarvam key on Save (no network: the POST is injected)."""

from __future__ import annotations

import pytest
import requests

from subtitle_checker.report.keycheck import (
    SAVE_REJECTED,
    SAVE_UNREACHABLE,
    check_key,
    is_key_rejection,
)


class _Resp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def _answering(status: int):
    def post(url, headers, timeout):
        assert "speech-to-text" in url
        assert headers == {"api-subscription-key": "k"}
        return _Resp(status)

    return post


def _failing(exc: Exception):
    def post(url, headers, timeout):
        raise exc

    return post


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_is_reported(status):
    assert check_key("k", _answering(status)) == SAVE_REJECTED


@pytest.mark.parametrize("status", [200, 400, 422, 429])
def test_any_answer_past_the_key_check_accepts_it(status):
    # An empty request gets "file required" (400) from Sarvam when the key is good.
    assert check_key("k", _answering(status)) is None


@pytest.mark.parametrize("status", [500, 502, 503])
def test_a_sarvam_outage_is_not_blamed_on_the_key(status):
    assert check_key("k", _answering(status)) == SAVE_UNREACHABLE


@pytest.mark.parametrize(
    "exc", [requests.ConnectionError("down"), requests.Timeout("slow")]
)
def test_no_connection_is_reported_as_unreachable(exc):
    assert check_key("k", _failing(exc)) == SAVE_UNREACHABLE


@pytest.mark.parametrize(
    "exc",
    [requests.exceptions.InvalidHeader("line break"), UnicodeEncodeError("latin-1", "क", 0, 1, "x")],
)
def test_text_that_cannot_be_a_key_is_rejected(exc):
    assert check_key("k", _failing(exc)) == SAVE_REJECTED


def test_a_key_with_a_line_break_is_rejected_before_any_request():
    # requests refuses the header while preparing the call, so this needs no network.
    assert check_key("abc\ndef") == SAVE_REJECTED


def test_key_rejection_is_read_from_both_client_styles():
    class SdkError(Exception):
        status_code = 403

    http_error = requests.HTTPError("403")
    http_error.response = _Resp(401)
    assert is_key_rejection(SdkError())
    assert is_key_rejection(http_error)


def test_other_failures_are_not_key_rejections():
    class ServerError(Exception):
        status_code = 500

    assert not is_key_rejection(ServerError())
    assert not is_key_rejection(RuntimeError("boom"))
    assert not is_key_rejection(requests.HTTPError("no response"))
