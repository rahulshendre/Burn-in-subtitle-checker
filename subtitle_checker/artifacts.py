"""Stage contracts for the subtitle checker pipeline.

Every pipeline stage reads one JSON artifact and writes another. The
dataclasses here define those artifacts. Keeping the contract in one module
lets any stage be cached, re-run, or swapped (a different OCR engine, a
different ASR backend) without touching the rest of the pipeline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

SCHEMA_VERSION = 1


class Verdict(str, Enum):
    """Outcome for one checked span of the video."""

    OK = "OK"
    TEXT_MISMATCH = "TEXT_MISMATCH"
    MISSING_SUBTITLE = "MISSING_SUBTITLE"
    ORPHAN_SUBTITLE = "ORPHAN_SUBTITLE"
    TIMING_DRIFT = "TIMING_DRIFT"
    UNCHECKABLE = "UNCHECKABLE"


class AudioKind(str, Enum):
    """Coarse label for a region of the audio track."""

    SPEECH = "speech"
    MUSIC = "music"
    SONG = "song"
    SILENCE = "silence"


@dataclass
class SubtitleEvent:
    """One burned-in subtitle line: when it appears, disappears, what it says."""

    start: float
    end: float
    text: str
    confidence: float = 1.0


@dataclass
class AudioRegion:
    """One labelled span of the audio track."""

    start: float
    end: float
    kind: AudioKind
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.kind = AudioKind(self.kind)


@dataclass
class CheckResult:
    """Verdict for one span, with the evidence an editor needs to judge it."""

    start: float
    end: float
    verdict: Verdict
    reason: str
    subtitle_text: str = ""
    heard_text: str = ""
    score: float | None = None

    def __post_init__(self) -> None:
        self.verdict = Verdict(self.verdict)


_REGISTRY = {
    "subtitle_events": SubtitleEvent,
    "audio_regions": AudioRegion,
    "check_results": CheckResult,
}


def save_artifact(path: Path, kind: str, items: list) -> None:
    """Write a stage artifact as UTF-8 JSON (Devanagari stays readable)."""
    if kind not in _REGISTRY:
        raise ValueError(f"unknown artifact kind: {kind!r}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "items": [asdict(item) for item in items],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_artifact(path: Path) -> tuple[str, list]:
    """Read a stage artifact back into typed dataclasses."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    kind = payload["kind"]
    if kind not in _REGISTRY:
        raise ValueError(f"unknown artifact kind: {kind!r}")
    cls = _REGISTRY[kind]
    return kind, [cls(**item) for item in payload["items"]]
