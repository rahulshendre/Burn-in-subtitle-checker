"""Tests for the per-line score fusion (the combined OCR + audio confidence)."""

from subtitle_checker.match.scoring import W_MATCH, W_OCR, combined_score


def test_clean_line_scores_high() -> None:
    # Read well (OCR 0.9) and the audio matches (0.92) -> a confident line.
    assert combined_score(0.9, 0.92) > 90


def test_wrong_line_scores_low() -> None:
    # A wrong line: read clearly but the audio does not match it at all.
    assert combined_score(1.0, 0.0) < 40


def test_no_audio_signal_is_unscored() -> None:
    # Over music or silence there is nothing to match against - no number.
    assert combined_score(0.8, None) is None


def test_audio_match_outweighs_ocr() -> None:
    # An ASR-verified line with poor OCR still scores well; a well-read line the
    # audio disputes scores worse. Match is the stronger evidence of correctness.
    verified_poor_read = combined_score(0.2, 0.9)
    clean_read_disputed = combined_score(0.9, 0.2)
    assert verified_poor_read > clean_read_disputed


def test_missing_ocr_confidence_counts_as_zero() -> None:
    assert combined_score(None, 0.8) == round(100 * W_MATCH * 0.8, 1)


def test_weights_sum_to_one() -> None:
    assert W_MATCH + W_OCR == 1.0
    assert combined_score(1.0, 1.0) == 100.0
