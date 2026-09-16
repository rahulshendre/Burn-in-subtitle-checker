"""Stage 1 orchestration: video → subtitle event timeline.

Pass A samples the band and accumulates per-pixel and region presence to find
chrome (static bugs and animated logos alike).
Pass B re-samples and detects events on chrome-subtracted masks.
Each surviving event is OCR'd once, at its middle frame, at native
resolution.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from subtitle_checker.artifacts import SubtitleEvent
from subtitle_checker.subtitles.events import (
    MAX_EVENT_S,
    RawEvent,
    chrome_mask,
    detect_events,
    presence_fields,
)
from subtitle_checker.subtitles.legibility import (
    contrast,
    legibility_ratio,
    line_height_frac,
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
        crop = _crop_to_text(band, raw.bbox)
        text, confidence = engine.read(crop)
        # extract_band_frame returns the band below band_top, so the full frame
        # height is the band's height scaled back up by the fraction it covers.
        frame_h = band.shape[0] / (1.0 - band_top) if band_top < 1.0 else band.shape[0]
        subtitles.append(
            SubtitleEvent(
                start=raw.start,
                end=raw.end,
                text=text,
                confidence=confidence,
                legibility=contrast(crop),
                line_count=raw.line_count,
                legibility_ratio=legibility_ratio(crop),
                line_height_frac=line_height_frac(crop, raw.line_count or 1, frame_h),
            )
        )
    return _merge_repeated(subtitles)


# A subtitle that briefly flickers off (a dropped frame or two mid-display)
# splits into two events with the same text a fraction of a second apart. A real
# repeated line sits further apart, so only bridge a short gap.
_REPEAT_GAP_S = 0.6


def _merge_repeated(events: list[SubtitleEvent]) -> list[SubtitleEvent]:
    """Collapse consecutive events with identical text separated by a small gap.

    One subtitle split by a flicker becomes one span again, and a span the split
    kept under the mismatch floor can clear it once rejoined. Text must match
    exactly - different OCR readings stay separate.
    """
    if not events:
        return events
    merged = [events[0]]
    for ev in events[1:]:
        prev = merged[-1]
        if ev.text and ev.text == prev.text and ev.start - prev.end <= _REPEAT_GAP_S:
            # Keep the earlier event's readings; only stretch the end forward.
            merged[-1] = replace(prev, end=ev.end)
        else:
            merged.append(ev)
    return merged


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
