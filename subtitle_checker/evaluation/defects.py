"""Synthetic defect planning for the evaluation harness.

Takes a clean subtitle event list (the verified truth) and returns a mutated
copy with controlled, labelled defects. Burning the mutated list back onto
the source video (see burn.py) yields a test video whose subtitle errors are
known exactly, so pipeline flags can be scored as precision/recall
(see score.py) instead of eyeballed.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from subtitle_checker.artifacts import SubtitleEvent, Verdict


class DefectType(str, Enum):
    WORD_SWAP = "word_swap"
    TIMING_SHIFT = "timing_shift"
    DROP_LINE = "drop_line"
    EXTRA_LINE = "extra_line"


EXPECTED_VERDICT = {
    DefectType.WORD_SWAP: Verdict.TEXT_MISMATCH,
    DefectType.TIMING_SHIFT: Verdict.TIMING_DRIFT,
    DefectType.DROP_LINE: Verdict.MISSING_SUBTITLE,
    DefectType.EXTRA_LINE: Verdict.ORPHAN_SUBTITLE,
}

# A shifted line must move far enough that the drift is unambiguous.
MIN_SHIFT_S = 0.8
MAX_SHIFT_S = 1.6

# An extra line needs a real silent gap to sit in. Gaps between SLS subtitle
# events are dialogue-free by construction (subtitles are verbatim).
MIN_GAP_S = 1.5


@dataclass
class Defect:
    """One planted defect: where it is and what the pipeline should say."""

    type: DefectType
    start: float
    end: float
    original_text: str
    mutated_text: str

    def __post_init__(self) -> None:
        self.type = DefectType(self.type)

    @property
    def expected_verdict(self) -> Verdict:
        return EXPECTED_VERDICT[self.type]


def plan_defects(
    events: list[SubtitleEvent], seed: int = 0
) -> tuple[list[SubtitleEvent], list[Defect]]:
    """Plant one defect of each type; return (mutated events, defect labels).

    Victim lines are chosen with a seeded RNG so the same input always yields
    the same test video.
    """
    if len(events) < 4:
        raise ValueError("need at least 4 subtitle events to plant all defect types")

    rng = random.Random(seed)
    mutated = [SubtitleEvent(e.start, e.end, e.text, e.confidence) for e in events]
    swap_i, shift_i, drop_i = rng.sample(range(len(mutated)), 3)

    defects = [
        _swap_word(rng, mutated, swap_i),
        _shift_timing(rng, mutated, shift_i),
    ]
    extra_event, extra_defect = _make_extra_line(rng, truth=events, occupied=mutated)
    defects.append(extra_defect)

    dropped = mutated[drop_i]
    defects.append(
        Defect(
            type=DefectType.DROP_LINE,
            start=dropped.start,
            end=dropped.end,
            original_text=dropped.text,
            mutated_text="",
        )
    )
    del mutated[drop_i]

    mutated.append(extra_event)
    mutated.sort(key=lambda e: e.start)
    return mutated, defects


def save_defects(path: Path, defects: list[Defect]) -> None:
    payload = [asdict(d) for d in defects]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_defects(path: Path) -> list[Defect]:
    return [Defect(**d) for d in json.loads(path.read_text(encoding="utf-8"))]


def _swap_word(rng: random.Random, events: list[SubtitleEvent], index: int) -> Defect:
    """Replace one word of the victim line with a word from elsewhere in the clip."""
    victim = events[index]
    words = victim.text.split()
    pos = rng.randrange(len(words))
    # sorted() keeps the choice deterministic across interpreter runs
    donors = sorted({w for e in events for w in e.text.split() if w != words[pos]})
    if not donors:
        raise ValueError("subtitle text too uniform to plant a word swap")
    words[pos] = rng.choice(donors)
    swapped = " ".join(words)
    events[index] = SubtitleEvent(victim.start, victim.end, swapped, victim.confidence)
    return Defect(
        type=DefectType.WORD_SWAP,
        start=victim.start,
        end=victim.end,
        original_text=victim.text,
        mutated_text=swapped,
    )


def _shift_timing(rng: random.Random, events: list[SubtitleEvent], index: int) -> Defect:
    """Slide the victim line off its audio; the defect span covers both positions."""
    victim = events[index]
    shift = rng.uniform(MIN_SHIFT_S, MAX_SHIFT_S) * rng.choice((-1, 1))
    if victim.start + shift < 0:
        shift = abs(shift)
    moved = SubtitleEvent(victim.start + shift, victim.end + shift, victim.text, victim.confidence)
    events[index] = moved
    return Defect(
        type=DefectType.TIMING_SHIFT,
        start=min(victim.start, moved.start),
        end=max(victim.end, moved.end),
        original_text=victim.text,
        mutated_text=victim.text,
    )


def _make_extra_line(
    rng: random.Random, truth: list[SubtitleEvent], occupied: list[SubtitleEvent]
) -> tuple[SubtitleEvent, Defect]:
    """Build a subtitle line sitting in the largest silent gap.

    Silence is judged from the *truth* timeline — the audio never changes, so
    speech sits wherever truth subtitles sat, even for lines the mutations
    dropped or moved. Collision is judged against the *mutated* timeline so
    the extra line never overlaps a line that was shifted into the gap.
    """
    ordered = sorted(truth, key=lambda e: e.start)
    gaps = [
        (a.end, b.start) for a, b in zip(ordered, ordered[1:]) if b.start - a.end >= MIN_GAP_S
    ]
    free = [g for g in gaps if not any(o.start < g[1] and g[0] < o.end for o in occupied)]
    if not free:
        raise ValueError(f"no silent gap of at least {MIN_GAP_S}s to plant an extra line")
    gap_start, gap_end = max(free, key=lambda g: g[1] - g[0])
    text = rng.choice(ordered).text
    duration = min(2.5, (gap_end - gap_start) * 0.8)
    start = gap_start + ((gap_end - gap_start) - duration) / 2
    event = SubtitleEvent(start=start, end=start + duration, text=text)
    defect = Defect(
        type=DefectType.EXTRA_LINE,
        start=event.start,
        end=event.end,
        original_text="",
        mutated_text=text,
    )
    return event, defect
