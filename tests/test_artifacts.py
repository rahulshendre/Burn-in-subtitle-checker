"""Round-trip and validation tests for the stage artifact contract."""

from pathlib import Path

import pytest

from subtitle_checker.artifacts import (
    AudioKind,
    AudioRegion,
    CheckResult,
    SubtitleEvent,
    Verdict,
    load_artifact,
    save_artifact,
)


def test_subtitle_events_round_trip(tmp_path: Path) -> None:
    events = [
        SubtitleEvent(start=1.0, end=3.5, text="वो कहाँ गया था", confidence=0.92),
        SubtitleEvent(start=4.0, end=6.0, text="ठीक है भाई"),
    ]
    path = tmp_path / "events.json"
    save_artifact(path, "subtitle_events", events)

    kind, loaded = load_artifact(path)
    assert kind == "subtitle_events"
    assert loaded == events


def test_devanagari_stays_readable_in_file(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    save_artifact(path, "subtitle_events", [SubtitleEvent(0.0, 1.0, "ठीक है")])
    assert "ठीक है" in path.read_text(encoding="utf-8")


def test_audio_regions_round_trip_restores_enum(tmp_path: Path) -> None:
    regions = [AudioRegion(start=0.0, end=2.0, kind=AudioKind.SONG, confidence=0.8)]
    path = tmp_path / "regions.json"
    save_artifact(path, "audio_regions", regions)

    _, loaded = load_artifact(path)
    assert loaded[0].kind is AudioKind.SONG


def test_check_results_round_trip_restores_enum(tmp_path: Path) -> None:
    results = [
        CheckResult(
            start=10.2,
            end=12.9,
            verdict=Verdict.TEXT_MISMATCH,
            reason="alignment score below threshold",
            subtitle_text="वो कहाँ गया था",
            heard_text="वो कहाँ गई थी",
            score=0.61,
        )
    ]
    path = tmp_path / "results.json"
    save_artifact(path, "check_results", results)

    _, loaded = load_artifact(path)
    assert loaded[0].verdict is Verdict.TEXT_MISMATCH
    assert loaded[0].score == pytest.approx(0.61)


def test_unknown_kind_rejected_on_save(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown artifact kind"):
        save_artifact(tmp_path / "x.json", "nonsense", [])
