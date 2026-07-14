"""Binary text masks and their comparison.

Subtitle text on SLS content is high-luminance: white fill, and the karaoke
highlight is yellow - still bright in grayscale. A luminance threshold
therefore isolates the text *shape*, and comparing shapes instead of raw
pixels keeps the word-by-word highlight sweep from looking like a text
change.
"""

from __future__ import annotations

import numpy as np

# White text ≈ 255, yellow highlight ≈ 226 in grayscale; scene content in the
# subtitle band almost never sustains values this high.
DEFAULT_THRESHOLD = 190


def binarize(gray: np.ndarray, threshold: int = DEFAULT_THRESHOLD) -> np.ndarray:
    """Grayscale band frame → boolean text mask."""
    return gray >= threshold


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two masks; 1.0 when both are empty."""
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def _erode(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] &= mask[:-1, :]
    out[:-1, :] &= mask[1:, :]
    out[:, 1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1:]
    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def remove_fat_regions(mask: np.ndarray, thickness: int = 3) -> np.ndarray:
    """Drop bright regions thicker than text strokes.

    Subtitle strokes are 2-3 px at detection scale; bright clothing, skin,
    or sky are solid blobs. Erosion kills anything thinner than
    ``thickness``; dilating the survivors back and subtracting removes the
    fat regions while the strokes stay.
    """
    core = mask
    for _ in range(thickness):
        core = _erode(core)
    if not core.any():
        return mask
    fat = core
    for _ in range(thickness + 1):
        fat = _dilate(fat)
    return mask & ~fat


def text_mask(gray: np.ndarray, threshold: int = DEFAULT_THRESHOLD) -> np.ndarray:
    """Grayscale band → mask of text-stroke-like bright pixels."""
    return remove_fat_regions(binarize(gray, threshold))
