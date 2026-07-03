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
# A pixel must be lit for this fraction of an event to count as its text.
# Subtitle glyphs hold still for the whole event; a scrolling ticker or a
# glinting logo lights any one pixel only briefly.
STABLE_PIXEL_FRACTION = 0.6
# A horizontal text block whose lit pixels were mostly transient is a
# scrolling ticker sweeping through, not a subtitle. (A ticker still leaks
# two stable streaks — its सूचना-style label chip and the Devanagari
# headline stroke, which scrolling never moves — so the block's *ratio* of
# stable to ever-lit pixels is the tell: real subtitles measure ~1.0,
# scrolling tickers ~0.2.)
MIN_CLUSTER_STABILITY = 0.5
# Row gaps up to this many detection-scale pixels stay one text block; the
# two lines of a wrapped subtitle sit closer than a subtitle sits to a
# bottom-edge ticker.
ROW_CLUSTER_GAP = 6


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


def _stable_text_bbox(
    lit_frames: np.ndarray, frame_count: int
) -> tuple[int, int, int, int] | None:
    """Bounding box of the event's own text, from its per-pixel lit counts.

    Splits the ever-lit rows into horizontal blocks, drops blocks that were
    mostly transient (scrolling tickers), and boxes the stable pixels of
    what remains.
    """
    stable = lit_frames >= max(frame_count * STABLE_PIXEL_FRACTION, 1)
    union = lit_frames > 0
    rows = np.flatnonzero(union.any(axis=1))
    if rows.size == 0:
        return None
    keep = np.zeros(union.shape[0], dtype=bool)
    splits = np.flatnonzero(np.diff(rows) > ROW_CLUSTER_GAP)
    for cluster in np.split(rows, splits + 1):
        r0, r1 = cluster[0], cluster[-1] + 1
        lit = int(union[r0:r1].sum())
        if lit and stable[r0:r1].sum() / lit >= MIN_CLUSTER_STABILITY:
            keep[r0:r1] = True
    masked = np.logical_and(stable, keep[:, None])
    return _mask_bbox(masked) or _mask_bbox(stable) or _mask_bbox(union)


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
    reference: np.ndarray | None = None
    lit_frames: np.ndarray | None = None
    frame_count = 0
    prev_t: float | None = None
    prev_raw: np.ndarray | None = None

    def close(end: float) -> None:
        nonlocal current_start, reference, lit_frames, frame_count
        if current_start is not None and end - current_start >= min_event_s:
            bbox = _stable_text_bbox(lit_frames, frame_count)
            events.append(RawEvent(start=current_start, end=end, bbox=bbox))
        current_start, reference, lit_frames, frame_count = None, None, None, 0

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
            lit_frames, frame_count = mask.astype(np.uint16), 1
        elif mask_iou(mask, reference) < same_event_iou:
            close(prev_t if prev_t is not None else t)
            current_start, reference = t, mask
            lit_frames, frame_count = mask.astype(np.uint16), 1
        else:
            # follow the highlight sweep so slow drift stays one event
            reference = mask
            lit_frames += mask
            frame_count += 1
        prev_t = t

    if prev_t is not None:
        close(prev_t)
    return events
