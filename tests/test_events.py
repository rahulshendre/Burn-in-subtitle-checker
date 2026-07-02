"""Event detection over synthetic mask streams."""

import numpy as np
import pytest

from subtitle_checker.subtitles.events import (
    chrome_mask,
    detect_events,
    presence_fraction,
)

H, W = 20, 100
FPS = 4.0


def blank() -> np.ndarray:
    return np.zeros((H, W), dtype=bool)


def text_a() -> np.ndarray:
    mask = blank()
    mask[5:15, 10:50] = True
    return mask


def text_b() -> np.ndarray:
    mask = blank()
    mask[5:15, 55:95] = True
    return mask


def stream(masks: list[np.ndarray]) -> list[tuple[float, np.ndarray]]:
    return [(i / FPS, m) for i, m in enumerate(masks)]


def test_single_steady_line_is_one_event() -> None:
    masks = [blank()] * 4 + [text_a()] * 8 + [blank()] * 4
    events = detect_events(stream(masks))
    assert len(events) == 1
    # stabilization costs one sample on the start
    assert events[0].start == pytest.approx(1.25)
    assert events[0].end == pytest.approx(1.0 + 7 / FPS)


def test_text_change_with_no_gap_splits_events() -> None:
    masks = [text_a()] * 8 + [text_b()] * 8
    events = detect_events(stream(masks))
    assert len(events) == 2
    assert events[0].end < events[1].start


def test_karaoke_highlight_sweep_stays_one_event() -> None:
    # the highlight progressively thickens the text region a little
    masks = []
    for i in range(12):
        mask = text_a()
        mask[15:17, 10 : 10 + 3 * i] = True  # growing underline-ish sweep
        masks.append(mask)
    events = detect_events(stream(masks))
    assert len(events) == 1


def test_single_frame_blip_is_dropped() -> None:
    masks = [blank()] * 4 + [text_a()] + [blank()] * 4
    assert detect_events(stream(masks)) == []


def test_tiny_noise_is_not_text() -> None:
    noise = blank()
    noise[0:2, 0:20] = True  # 40 px < MIN_TEXT_PIXELS
    assert detect_events(stream([noise] * 8)) == []


def test_moving_sparkle_is_killed_by_stabilization() -> None:
    # 200 px of bright "sequins" that shift every frame — enough pixels to
    # pass the text minimum, but never stable across two frames
    masks = []
    for i in range(12):
        mask = blank()
        col = (25 * i) % 80  # jumps far enough that frames never overlap
        mask[2:12, col : col + 20] = True
        masks.append(mask)
    assert detect_events(stream(masks)) == []


def test_chrome_pixels_are_ignored() -> None:
    bug = blank()
    bug[0:6, 88:100] = True  # persistent corner bug, 72 px

    masks = []
    for i in range(40):
        mask = bug.copy()
        if 8 <= i < 16:
            mask |= text_a()
        masks.append(mask)

    presence = presence_fraction(m for m in masks)
    chrome = chrome_mask(presence)
    assert chrome[2, 90]  # the bug is chrome
    assert not chrome[10, 30]  # the text region is not

    events = detect_events(stream(masks), chrome=chrome)
    assert len(events) == 1
    assert events[0].start == pytest.approx(2.25)


def test_presence_fraction_requires_frames() -> None:
    with pytest.raises(ValueError, match="no frames"):
        presence_fraction(iter([]))
