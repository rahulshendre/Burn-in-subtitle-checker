"""Unit tests for band geometry; streaming is covered by the integration test."""

from subtitle_checker.subtitles.sampler import band_crop


def test_band_crop_is_even_and_bottom_anchored() -> None:
    crop_h, y = band_crop(720, band_top=0.70)
    assert crop_h % 2 == 0
    assert crop_h + y == 720
    assert crop_h == 216  # 30% of 720 = 216, already even


def test_band_crop_rounds_odd_heights_down() -> None:
    crop_h, y = band_crop(715, band_top=0.70)
    assert crop_h % 2 == 0
    assert crop_h + y == 715
