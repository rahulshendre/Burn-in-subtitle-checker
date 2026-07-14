"""Tests for band-frame binarization and mask comparison."""

import numpy as np

from subtitle_checker.subtitles.masks import binarize, mask_iou, remove_fat_regions, text_mask


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


def test_fat_filter_keeps_strokes_drops_blobs() -> None:
    mask = np.zeros((40, 100), dtype=bool)
    mask[10:13, 5:45] = True  # 3 px stroke - text-like
    mask[20:38, 50:95] = True  # 18 px solid blob - clothing-like
    cleaned = remove_fat_regions(mask, thickness=3)
    assert cleaned[11, 20]  # stroke survives
    assert not cleaned[29, 70]  # blob core removed
    assert cleaned[10:13, 8:42].all()  # stroke intact end to end


def test_fat_filter_with_no_fat_regions_is_identity() -> None:
    mask = np.zeros((20, 60), dtype=bool)
    mask[8:10, 5:55] = True  # thin stroke only
    assert (remove_fat_regions(mask) == mask).all()


def test_text_mask_composes_threshold_and_fat_filter() -> None:
    gray = np.zeros((40, 100), dtype=np.uint8)
    gray[10:13, 5:45] = 255  # bright stroke
    gray[20:38, 50:95] = 255  # bright blob
    gray[30:32, 5:45] = 100  # dim stroke - under threshold
    cleaned = text_mask(gray)
    assert cleaned[11, 20]
    assert not cleaned[29, 70]
    assert not cleaned[31, 20]
