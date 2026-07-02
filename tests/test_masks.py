"""Tests for band-frame binarization and mask comparison."""

import numpy as np

from subtitle_checker.subtitles.masks import binarize, mask_iou


def test_binarize_keeps_white_and_yellow_drops_scene() -> None:
    gray = np.array([[255, 226, 120, 40]], dtype=np.uint8)
    assert binarize(gray).tolist() == [[True, True, False, False]]


def test_iou_identical_masks() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    assert mask_iou(mask, mask) == 1.0


def test_iou_disjoint_masks() -> None:
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    a[0, 0] = True
    b[3, 3] = True
    assert mask_iou(a, b) == 0.0


def test_iou_both_empty_counts_as_same() -> None:
    empty = np.zeros((4, 4), dtype=bool)
    assert mask_iou(empty, empty) == 1.0


def test_iou_partial_overlap() -> None:
    a = np.zeros((1, 4), dtype=bool)
    b = np.zeros((1, 4), dtype=bool)
    a[0, :3] = True  # 3 pixels
    b[0, 1:] = True  # 3 pixels, 2 shared
    assert mask_iou(a, b) == 0.5
