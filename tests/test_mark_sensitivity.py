"""The word matcher must see vowel signs, nukta and chandrabindu.

Whisper's non-English text normaliser strips Unicode marks, which hides matra
errors. Ours compares the raw strings, so a mark-only difference must always
score below an identical line. These tests pin that: if anyone adds a
normalising step that drops marks, the pairs below collapse to 100 and fail.
"""

from __future__ import annotations

import pytest
from rapidfuzz.fuzz import token_set_ratio

from subtitle_checker.match.asr import MIN_TOKEN_RATIO

# (written, heard) pairs that differ only by a combining mark.
MARK_ONLY = [
    pytest.param("हमारे परिवार की इज़्ज़त होगी", "हमारे परवार की इज़्ज़त होगी", id="hindi-matra-swap"),
    pytest.param("सबको पता चल जायेगा", "सबक पता चल जायेगा", id="hindi-matra-drop"),
    pytest.param("वो घर में हैं", "वो घर में है", id="hindi-chandrabindu-or-matra"),
    pytest.param("इज़्ज़त बहुत ज़रूरी है", "इजत बहुत जरूरी है", id="hindi-nukta-drop"),
    pytest.param("आणि आपल्या घरी मुलांना सांगा", "आणि आपल्या घरी मुलाना सांगा", id="marathi-matra-drop"),
    pytest.param("तुला ते कधी दिसले नाही", "तुला ते कधी दिसल नाही", id="marathi-matra-drop-verb"),
    pytest.param("ನಾನು ಮನೆ ಹೋಗಿ", "ನಾನು ಮನ ಹೋಗಿ", id="kannada-matra-drop"),
]


@pytest.mark.parametrize(("written", "heard"), MARK_ONLY)
def test_mark_only_difference_scores_below_identical(written: str, heard: str) -> None:
    assert token_set_ratio(written, written) == 100.0
    assert token_set_ratio(written, heard) < 100.0


@pytest.mark.parametrize(("written", "heard"), MARK_ONLY)
def test_mark_only_difference_is_not_auto_flagged(written: str, heard: str) -> None:
    # Documented limit: one vowel sign is smaller than ASR/OCR noise, so it stays
    # above the flag cut. The report's cluster-level highlight is what surfaces it.
    assert token_set_ratio(written, heard) >= MIN_TOKEN_RATIO
