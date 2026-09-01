"""The Devanagari junk filter that cleans chrome/logo leak out of OCR boxes.

Cases are the real EasyOCR boxes from the Gatha clip (animated TATA PLAY logo
bleeding into the crop) and their eyeballed truth.
"""

from subtitle_checker.subtitles.ocr import (
    SARVAM_VISION_TRUSTED_CONF,
    _is_devanagari_line,
    _is_image_description,
    _strip_channel_logo,
    _vision_text,
)

# The two real Sarvam Vision captions from the Mann guidelines clip, where it
# narrated a no-subtitle scene frame instead of reading a subtitle.
_DESCRIBE_1 = (
    "यह छवि एक ग्रेस्केल (black and white) है जिसमें एक व्यक्ति के हाथ पर दो मोतियों की "
    "मालाएँ दिखाई दे रही हैं। हाथ पर एक सफेद कपड़े की कढ़ाई है।"
)
_DESCRIBE_2 = (
    "यह छवि एक ग्रेस्केल (black and white) है जिसमें एक व्यक्ति के हाथ दिखाई दे रहे हैं, "
    "जिन्होंने एक सफेद कपड़े पर मोतियों की माला पहनी हुई है। यह दृश्य किसी पारंपरिक भारतीय "
    "शादी के परिधान का है।"
)


def test_keeps_real_subtitle_lines():
    assert _is_devanagari_line("एक मां को और क्या चाहिए")
    assert _is_devanagari_line("हम श्राप मुक्त हो गए हैं")
    assert _is_devanagari_line("और दूसरे भगवान।")  # trailing danda is Devanagari


def test_keeps_lines_with_light_punctuation():
    assert _is_devanagari_line("पुत्र, दुख हरने आए हो या फिर देने?")
    assert _is_devanagari_line("[भावुक पार्श्व संगीत]")  # brackets are a small fraction


def test_drops_logo_and_sparkle_junk():
    assert not _is_devanagari_line('"^7')
    assert not _is_devanagari_line("177374")
    assert not _is_devanagari_line("१/ /")  # Devanagari digit + slashes, no letters
    assert not _is_devanagari_line('"डद"')  # two letters, but half the box is quotes
    assert not _is_devanagari_line("1/7^7")


def test_drops_empty_and_single_glyph():
    assert not _is_devanagari_line("")
    assert not _is_devanagari_line("   ")
    assert not _is_devanagari_line("ढ")  # one stray letter is not a line


def test_vision_text_joins_lines_and_drops_logo_blocks():
    # Sarvam returns the subtitle and the leaked channel logo as separate blocks
    text, conf = _vision_text(["और बुरा तो तब होगा।", "DD Free Dish"])
    assert text == "और बुरा तो तब होगा।"  # logo block dropped
    assert conf == SARVAM_VISION_TRUSTED_CONF


def test_vision_text_empty_when_only_chrome():
    text, conf = _vision_text(["TATA PL", "177374"])
    assert text == ""
    assert conf == 0.0


def test_flags_image_description_captions():
    # Sarvam narrated the frame - English rendering term and image-referent opener
    assert _is_image_description(_DESCRIBE_1)
    assert _is_image_description(_DESCRIBE_2)
    assert _is_image_description("यह दृश्य किसी पारंपरिक भारतीय शादी का है।")


def test_keeps_dialogue_that_merely_mentions_an_image():
    # A line is only a caption when a referent opens it, not when one appears
    assert not _is_image_description("मेरी छवि आईने में धुंधली थी।")
    assert not _is_image_description("हम सबसे नज़रें कैसे मिला पाएँगे?")
    assert not _is_image_description("आज हमारे कारण")


def test_vision_text_drops_narrated_frame():
    # A block that is a picture caption must not become a subtitle line
    text, conf = _vision_text([_DESCRIBE_1])
    assert text == ""
    assert conf == 0.0


def test_vision_text_keeps_real_line_beside_a_caption():
    text, conf = _vision_text(["और बुरा तो तब होगा।", _DESCRIBE_2])
    assert text == "और बुरा तो तब होगा।"
    assert conf == SARVAM_VISION_TRUSTED_CONF


def test_strip_channel_logo_removes_merged_tata_play():
    # Real Sarvam Vision blocks from the without-guidelines clip, where the
    # TATA PLAY logo merged into the same block as the subtitle line.
    assert _strip_channel_logo("खुशी की क्या ही बात है? TATA PL") == "खुशी की क्या ही बात है?"
    assert _strip_channel_logo("नॉनस्टॉप बोले जा रहे हो। TATA PLAY") == "नॉनस्टॉप बोले जा रहे हो।"
    assert (
        _strip_channel_logo("मैंने हमेशा आपको सपोर्ट किया है कृष, हमेशा। TATA PLAY")
        == "मैंने हमेशा आपको सपोर्ट किया है कृष, हमेशा।"
    )


def test_strip_channel_logo_leaves_a_clean_line_untouched():
    assert _strip_channel_logo("हम ये घर छोड़कर जा रहे हैं।") == "हम ये घर छोड़कर जा रहे हैं।"
    assert _strip_channel_logo("मेरे बेटे को") == "मेरे बेटे को"


def test_vision_text_strips_logo_merged_into_a_real_line():
    # The merged-logo block passes the Devanagari test, so it must be cleaned
    # in place rather than dropped whole (dropping it would lose the subtitle).
    text, conf = _vision_text(["इस तरह इन्सल्ट करती हैं? TATA PLAY"])
    assert text == "इस तरह इन्सल्ट करती हैं?"
    assert conf == SARVAM_VISION_TRUSTED_CONF


class _FakeJob:
    def upload_file(self, _path) -> None:
        pass

    def start(self) -> None:
        pass


class _FakeDocIntel:
    """A create_job that 429s a set number of times, then returns a job."""

    def __init__(self, fails: int) -> None:
        self._fails = fails
        self.calls = 0

    def create_job(self, **_kwargs) -> _FakeJob:
        from sarvamai.errors.too_many_requests_error import TooManyRequestsError

        self.calls += 1
        if self.calls <= self._fails:
            raise TooManyRequestsError(headers={}, body={})
        return _FakeJob()


class _FakeClient:
    def __init__(self, fails: int) -> None:
        self.document_intelligence = _FakeDocIntel(fails)


def test_vision_job_retries_on_429_then_succeeds(monkeypatch) -> None:
    from subtitle_checker.subtitles.ocr import _start_vision_job

    slept: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    client = _FakeClient(fails=2)

    _start_vision_job(client, "band.png", "hi-IN")
    assert client.document_intelligence.calls == 3  # two 429s retried, third ok
    assert len(slept) == 2


def test_vision_job_raises_after_exhausting_retries(monkeypatch) -> None:
    import pytest

    from sarvamai.errors.too_many_requests_error import TooManyRequestsError

    from subtitle_checker.subtitles.ocr import (
        SARVAM_VISION_MAX_RETRIES,
        _start_vision_job,
    )

    monkeypatch.setattr("time.sleep", lambda _s: None)
    client = _FakeClient(fails=SARVAM_VISION_MAX_RETRIES + 1)

    with pytest.raises(TooManyRequestsError):
        _start_vision_job(client, "band.png", "hi-IN")
    assert client.document_intelligence.calls == SARVAM_VISION_MAX_RETRIES + 1


def test_vision_retry_after_grows_exponentially() -> None:
    from subtitle_checker.subtitles.ocr import (
        SARVAM_VISION_BACKOFF_S,
        _vision_retry_after,
    )

    assert _vision_retry_after(0) == SARVAM_VISION_BACKOFF_S
    assert _vision_retry_after(2) == SARVAM_VISION_BACKOFF_S * 4
