"""Turn a stream of band masks into subtitle events.

Two passes over the sampled band:

1. Presence pass — how often each pixel is lit across the video. Pixels lit
   most of the time are chrome (channel bugs, watermarks, "TATA PLAY"), not
   subtitles, and get removed from every mask before detection.
2. Event pass — a run of consecutive frames whose masks stay similar is one
   subtitle event; the text changing (IoU drop) or vanishing closes it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from subtitle_checker.subtitles.masks import mask_iou

# Karaoke highlighting nudges the mask a little; a text change replaces most
# of it.
SAME_EVENT_IOU = 0.35
# Fewer lit pixels than this is noise or stray brightness, not a subtitle.
MIN_TEXT_PIXELS = 120
# A pixel lit for over this fraction of the whole video is chrome.
CHROME_PRESENCE = 0.5
# Shorter than this is a sampling blip, not a subtitle line.
MIN_EVENT_S = 0.4
# Longer than this is a disclaimer or other persistent text, not dialogue.
MAX_EVENT_S = 15.0


@dataclass
class RawEvent:
    """A detected on-screen text span, before OCR."""

    start: float
    end: float
    # union bounding box of the text pixels, in detection-scale coordinates:
    # (row0, row1, col0, col1) — lets OCR crop to the text and ignore the
    # bright scenery around it
    bbox: tuple[int, int, int, int] | None = None

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _union(a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None):
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])


def presence_fraction(masks: Iterable[np.ndarray]) -> np.ndarray:
    """Per-pixel fraction of frames in which the pixel is lit."""
    total: np.ndarray | None = None
    count = 0
    for mask in masks:
        if total is None:
            total = mask.astype(np.uint32)
        else:
            total += mask
        count += 1
    if total is None:
        raise ValueError("no frames sampled from video")
    return total / count


def chrome_mask(presence: np.ndarray, cutoff: float = CHROME_PRESENCE) -> np.ndarray:
    """Pixels lit persistently enough to be chrome rather than subtitles."""
    return presence >= cutoff


def detect_events(
    frames: Iterable[tuple[float, np.ndarray]],
    chrome: np.ndarray | None = None,
    same_event_iou: float = SAME_EVENT_IOU,
    min_text_pixels: int = MIN_TEXT_PIXELS,
    min_event_s: float = MIN_EVENT_S,
    stabilize: bool = True,
) -> list[RawEvent]:
    """One pass over (time, mask) pairs → raw subtitle events.

    With ``stabilize``, each mask is ANDed with the previous frame's raw
    mask: subtitle text persists across frames while sequins, jewellery and
    other bright sparkle move every frame — the AND keeps the text and kills
    the flicker (costs one sample of latency on event starts).
    """
    events: list[RawEvent] = []
    current_start: float | None = None
    current_bbox: tuple[int, int, int, int] | None = None
    reference: np.ndarray | None = None
    prev_t: float | None = None
    prev_raw: np.ndarray | None = None

    def close(end: float) -> None:
        nonlocal current_start, current_bbox, reference
        if current_start is not None and end - current_start >= min_event_s:
            events.append(RawEvent(start=current_start, end=end, bbox=current_bbox))
        current_start, current_bbox, reference = None, None, None

    for t, mask in frames:
        if chrome is not None:
            mask = np.logical_and(mask, np.logical_not(chrome))
        if stabilize:
            raw = mask
            if prev_raw is not None:
                mask = np.logical_and(mask, prev_raw)
            prev_raw = raw
        has_text = int(mask.sum()) >= min_text_pixels

        if not has_text:
            close(prev_t if prev_t is not None else t)
        elif reference is None:
            current_start, reference = t, mask
            current_bbox = _mask_bbox(mask)
        elif mask_iou(mask, reference) < same_event_iou:
            close(prev_t if prev_t is not None else t)
            current_start, reference = t, mask
            current_bbox = _mask_bbox(mask)
        else:
            # follow the highlight sweep so slow drift stays one event
            reference = mask
            current_bbox = _union(current_bbox, _mask_bbox(mask))
        prev_t = t

    if prev_t is not None:
        close(prev_t)
    return events
