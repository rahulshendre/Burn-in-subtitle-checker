"""Closed-loop evaluation of Stage 3 forced alignment on real audio.

The structural eval can use synthetic regions because structural logic never
looks at audio content. Alignment does — it scores words against real speech —
so the truth has to come from a real clip: run Stage 1, keep the events whose
OCR is confident enough to trust as a transcript, and each is a *correct* pair
(its text matches the audio beneath it). Swap one word on a copy and you have a
*wrong* pair over the identical audio. A working aligner scores correct above
wrong; where the two distributions separate is the TEXT_MISMATCH threshold.

The single-word swap is the hard case on purpose. Most of the line still
matches, so the frame-weighted score dips only a little — if alignment
separates that, the blunter multi-word errors fall out easily. Where it does
not, that gap is exactly what the Sarvam ASR cross-check is there to cover.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from subtitle_checker.artifacts import SubtitleEvent
from subtitle_checker.match.align import ForcedAligner, score_event

# An event's OCR must be at least this confident before its text is trusted as
# ground truth. Garbled OCR (ornate clips) would otherwise pollute the correct
# side and hide the signal — on Pushpa the junk lines score as randomly as
# wrong text, so they must not count as "correct".
MIN_OCR_CONF = 0.5
# Fewer words than this and one swap changes most of the line (not the hard case
# we want to characterise); very short lines also risk the CTC frame floor.
MIN_WORDS = 3


@dataclass
class AlignmentEval:
    """Separation between correct and word-swapped lines on one clip."""

    pairs: int  # events that yielded a correct/swapped score pair
    correct_mean: float
    swapped_mean: float
    threshold: float  # best-separating cut on the alignment score
    recall: float  # swapped lines that fall below the threshold (defects caught)
    precision: float  # flags that are truly swapped, not correct lines


def _swap_one_word(text: str, donors: list[str], rng: random.Random) -> str:
    """Replace one word of the line with a different word from the clip."""
    words = text.split()
    pos = rng.randrange(len(words))
    choices = sorted(d for d in donors if d != words[pos])  # sorted → deterministic
    if choices:
        words[pos] = rng.choice(choices)
    return " ".join(words)


def _best_threshold(correct: list[float], swapped: list[float]) -> tuple[float, float, float]:
    """Cut that best separates correct (high) from swapped (low) scores.

    Returns (threshold, recall, precision) at the cut maximising balanced
    accuracy — the point that best trades catching swaps against sparing
    correct lines.
    """
    if not correct or not swapped:
        return 0.5, 0.0, 0.0
    best = (0.5, -1.0, 0.0, 0.0)  # threshold, balanced accuracy, recall, precision
    for cut in sorted(set(correct + swapped)):
        caught = sum(s < cut for s in swapped)
        false = sum(s < cut for s in correct)
        recall = caught / len(swapped)
        specificity = 1 - false / len(correct)
        precision = caught / (caught + false) if caught + false else 0.0
        balanced = (recall + specificity) / 2
        if balanced > best[1]:
            best = (cut, balanced, recall, precision)
    return best[0], best[2], best[3]


def evaluate_alignment(
    events: list[SubtitleEvent],
    audio: np.ndarray,
    aligner: ForcedAligner,
    min_ocr_conf: float = MIN_OCR_CONF,
    min_words: int = MIN_WORDS,
    seed: int = 0,
) -> AlignmentEval:
    """Score each trusted line correct-vs-word-swapped and measure separation."""
    rng = random.Random(seed)
    donors = sorted({w for e in events for w in e.text.split()})
    correct: list[float] = []
    swapped: list[float] = []
    for event in events:
        if event.confidence < min_ocr_conf or len(event.text.split()) < min_words:
            continue
        clean = score_event(event, audio, aligner).score
        if clean is None:
            continue
        wrong_text = _swap_one_word(event.text, donors, rng)
        wrong = SubtitleEvent(event.start, event.end, wrong_text, event.confidence)
        wrong_score = score_event(wrong, audio, aligner).score
        if wrong_score is None:
            continue
        correct.append(clean)
        swapped.append(wrong_score)
    threshold, recall, precision = _best_threshold(correct, swapped)
    return AlignmentEval(
        pairs=len(correct),
        correct_mean=sum(correct) / len(correct) if correct else 0.0,
        swapped_mean=sum(swapped) / len(swapped) if swapped else 0.0,
        threshold=threshold,
        recall=recall,
        precision=precision,
    )
