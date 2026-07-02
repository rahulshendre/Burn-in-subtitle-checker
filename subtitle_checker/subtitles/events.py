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

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2


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
) -> list[RawEvent]:
    """One pass over (time, mask) pairs → raw subtitle events."""
    events: list[RawEvent] = []
    current_start: float | None = None
    reference: np.ndarray | None = None
    prev_t: float | None = None

    def close(end: float) -> None:
        nonlocal current_start, reference
        if current_start is not None and end - current_start >= min_event_s:
            events.append(RawEvent(start=current_start, end=end))
        current_start, reference = None, None

    for t, mask in frames:
        if chrome is not None:
            mask = np.logical_and(mask, np.logical_not(chrome))
        has_text = int(mask.sum()) >= min_text_pixels

        if not has_text:
            close(prev_t if prev_t is not None else t)
        elif reference is None:
            current_start, reference = t, mask
        elif mask_iou(mask, reference) < same_event_iou:
            close(prev_t if prev_t is not None else t)
            current_start, reference = t, mask
        else:
            # follow the highlight sweep so slow drift stays one event
            reference = mask
        prev_t = t

    if prev_t is not None:
        close(prev_t)
    return events
