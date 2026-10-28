"""Check a Sarvam API key before it is saved, and recognise a rejected key later.

Sarvam checks the key before it looks at the request body, so an empty POST to
the speech-to-text endpoint answers 403 for a bad key and 400 ("file required")
for a good one. That costs no credit and sends no audio, so the key can be
tested the moment the user clicks Save instead of minutes into a check.
"""

from __future__ import annotations

from collections.abc import Callable

_REJECTED_STATUS = (401, 403)
_TIMEOUT_S = 15

SAVE_REJECTED = "Sarvam did not accept this key. Check that you copied all of it, then try again."
SAVE_UNREACHABLE = (
    "Could not reach Sarvam to check the key. Check your internet connection and try again."
)
RUN_REJECTED = "Your Sarvam API key was not accepted. Click Change API key to enter it again."


def check_key(key: str, post: Callable | None = None) -> str | None:
    """Return None when Sarvam accepts the key, else the message to show the user.

    A rate limit (429) still means the key was recognised. Network trouble and
    Sarvam server errors are reported as "could not check" so a good key is not
    called wrong; the key is only saved when this returns None.
    """
    import requests

    from subtitle_checker.match.asr import SARVAM_URL  # heavy imports stay off app start

    post = post or requests.post
    try:
        resp = post(SARVAM_URL, headers={"api-subscription-key": key}, timeout=_TIMEOUT_S)
    except (UnicodeError, requests.exceptions.InvalidHeader):
        return SAVE_REJECTED  # not a possible key: odd characters or a line break
    except requests.RequestException:
        return SAVE_UNREACHABLE
    if resp.status_code in _REJECTED_STATUS:
        return SAVE_REJECTED
    if resp.status_code >= 500:
        return SAVE_UNREACHABLE
    return None


def is_key_rejection(exc: BaseException) -> bool:
    """True when a failed Sarvam call was refused over the key (HTTP 401 or 403).

    Covers both client styles in the pipeline: `requests` (response.status_code)
    for speech-to-text and the Sarvam SDK (status_code on the error) for OCR.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in _REJECTED_STATUS
