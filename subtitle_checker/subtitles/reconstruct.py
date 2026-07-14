"""Stage 1 orchestration: video → subtitle event timeline.

Pass A samples the band and accumulates per-pixel and region presence to find
chrome (static bugs and animated logos alike).
Pass B re-samples and detects events on chrome-subtracted masks.
Each surviving event is OCR'd once, at its middle frame, at native
resolution.
"""

from __future__ import annotations

from pathlib import Path

from subtitle_checker.artifacts import SubtitleEvent
from subtitle_checker.subtitles.events import (
    MAX_EVENT_S,
    RawEvent,
    chrome_mask,
    detect_events,
    presence_fields,
)
from subtitle_checker.subtitles.masks import text_mask
from subtitle_checker.subtitles.ocr import EasyOcrEngine, OcrEngine
from subtitle_checker.subtitles.sampler import (
    DEFAULT_BAND_TOP,
    DEFAULT_FPS,
    extract_band_frame,
    iter_band_frames,
)


def detect_raw_events(
    video: Path,
    fps: float = DEFAULT_FPS,
    band_top: float = DEFAULT_BAND_TOP,
    threshold: int | None = None,
    max_event_s: float = MAX_EVENT_S,
) -> list[RawEvent]:
    """Detect on-screen text spans without OCR (both sampling passes)."""
    kwargs = {} if threshold is None else {"threshold": threshold}

    presence, region = presence_fields(
        text_mask(frame, **kwargs) for _, frame in iter_band_frames(video, fps, band_top)
    )
    chrome = chrome_mask(presence, region)

    events = detect_events(
        ((t, text_mask(frame, **kwargs)) for t, frame in iter_band_frames(video, fps, band_top)),
        chrome=chrome,
    )
    # very long "events" are disclaimers or missed chrome, not dialogue
    return [e for e in events if e.end - e.start <= max_event_s]


def reconstruct_subtitles(
    video: Path,
    engine: OcrEngine | None = None,
    fps: float = DEFAULT_FPS,
    band_top: float = DEFAULT_BAND_TOP,
    threshold: int | None = None,
) -> list[SubtitleEvent]:
    """Full Stage 1: detect events, OCR each once, return the subtitle track.

    Events where OCR finds no text are kept with empty text and zero
    confidence - "something bright was there but unreadable" is a signal the
    matcher wants, not something to hide.
    """
    engine = engine or EasyOcrEngine()
    subtitles = []
    for raw in detect_raw_events(video, fps, band_top, threshold):
        band = extract_band_frame(video, raw.mid, band_top)
        text, confidence = engine.read(_crop_to_text(band, raw.bbox))
        subtitles.append(
            SubtitleEvent(start=raw.start, end=raw.end, text=text, confidence=confidence)
        )
    return subtitles


# native-resolution pixels of context left around the text crop
_CROP_PAD = 12


def _crop_to_text(band, bbox, detection_width: int = 640):
    """Crop the native band to the event's text bbox so OCR never sees the
    bright scenery (sequins, jewellery) around the subtitle."""
    if bbox is None:
        return band
    scale = band.shape[1] / detection_width
    r0, r1, c0, c1 = (int(v * scale) for v in bbox)
    r0 = max(r0 - _CROP_PAD, 0)
    r1 = min(r1 + _CROP_PAD, band.shape[0])
    c0 = max(c0 - _CROP_PAD, 0)
    c1 = min(c1 + _CROP_PAD, band.shape[1])
    if r1 <= r0 or c1 <= c0:
        return band
    return band[r0:r1, c0:c1]
