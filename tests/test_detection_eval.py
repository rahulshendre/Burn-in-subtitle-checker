"""Truth-vs-detected matching for the Stage 1 detection eval."""

import pytest

from subtitle_checker.artifacts import SubtitleEvent
from subtitle_checker.evaluation.detection import make_truth, match_detection


def ev(start: float, end: float, text: str = "पाठ") -> SubtitleEvent:
    return SubtitleEvent(start=start, end=end, text=text)


def test_make_truth_lays_lines_with_gaps() -> None:
    truth = make_truth(["एक", "दो"], start=4.0, line_s=3.0, gap_s=2.5)
    assert truth[0].start == 4.0 and truth[0].end == 7.0
    assert truth[1].start == 9.5 and truth[1].end == 12.5


def test_exact_detection_matches_everything() -> None:
    truth = make_truth(["एक", "दो"])
    report = match_detection(truth, list(truth))
    assert report.recall == 1.0
    assert report.mean_similarity == 1.0
    assert report.strays == [] and report.missed == []


def test_shifted_detection_reports_timing_error() -> None:
    truth = [ev(4.0, 7.0)]
    report = match_detection(truth, [ev(4.25, 6.75)])
    assert len(report.matches) == 1
    assert report.mean_start_error == pytest.approx(0.25)
    assert report.mean_end_error == pytest.approx(0.25)


def test_undetected_line_is_missed() -> None:
    truth = [ev(4.0, 7.0), ev(10.0, 13.0)]
    report = match_detection(truth, [ev(4.0, 7.0)])
    assert len(report.missed) == 1
    assert report.missed[0].start == 10.0
    assert report.recall == 0.5


def test_detection_without_truth_is_stray() -> None:
    report = match_detection([ev(4.0, 7.0)], [ev(4.0, 7.0), ev(20.0, 21.0)])
    assert len(report.strays) == 1
    assert report.strays[0].start == 20.0


def test_tiny_overlap_does_not_match() -> None:
    report = match_detection([ev(4.0, 7.0)], [ev(6.8, 9.0)])
    assert report.matches == []
    assert len(report.missed) == 1 and len(report.strays) == 1


def test_one_detection_matches_only_one_truth_line() -> None:
    # a single long detection spanning two truth lines can satisfy only one
    truth = [ev(4.0, 7.0, "एक"), ev(8.0, 11.0, "दो")]
    report = match_detection(truth, [ev(4.0, 11.0, "एक दो")])
    assert len(report.matches) == 1
    assert len(report.missed) == 1


def test_ocr_noise_lowers_similarity() -> None:
    truth = [ev(4.0, 7.0, "एक मां को और क्या चाहिए।")]
    noisy = [ev(4.0, 7.0, "एक मां को और क्या चाहिए। ँट_३0ऋ")]
    report = match_detection(truth, noisy)
    assert 0.5 < report.mean_similarity < 1.0
