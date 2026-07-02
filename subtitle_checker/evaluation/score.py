"""Score pipeline flags against planted defects.

Matching is by span overlap: a flag counts for a defect if their time spans
come within OVERLAP_TOLERANCE_S of touching. Verdict correctness is tracked
separately, so "found the spot but called it the wrong thing" is visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from subtitle_checker.artifacts import CheckResult, Verdict
from subtitle_checker.evaluation.defects import Defect

# Absorbs frame-sampling jitter between a flag's span and the defect's span.
OVERLAP_TOLERANCE_S = 0.25


@dataclass
class TypeScore:
    planted: int = 0
    caught: int = 0
    verdict_correct: int = 0

    @property
    def recall(self) -> float:
        return self.caught / self.planted if self.planted else 0.0


@dataclass
class EvalScore:
    """Precision/recall of pipeline flags against a planted-defect label set."""

    planted: int
    caught: int
    true_flags: int
    false_flags: int
    uncheckable: int
    by_type: dict[str, TypeScore] = field(default_factory=dict)

    @property
    def recall(self) -> float:
        return self.caught / self.planted if self.planted else 0.0

    @property
    def precision(self) -> float:
        flagged = self.true_flags + self.false_flags
        return self.true_flags / flagged if flagged else 0.0


def _overlaps(flag: CheckResult, defect: Defect, tol: float = OVERLAP_TOLERANCE_S) -> bool:
    return flag.start < defect.end + tol and defect.start < flag.end + tol


def score_results(results: list[CheckResult], defects: list[Defect]) -> EvalScore:
    """Match non-OK flags against planted defects by span overlap.

    UNCHECKABLE spans are honest abstentions: counted, but neither true nor
    false flags.
    """
    flags = [r for r in results if r.verdict not in (Verdict.OK, Verdict.UNCHECKABLE)]
    uncheckable = sum(1 for r in results if r.verdict is Verdict.UNCHECKABLE)

    by_type: dict[str, TypeScore] = {}
    caught = 0
    for defect in defects:
        type_score = by_type.setdefault(defect.type.value, TypeScore())
        type_score.planted += 1
        hits = [f for f in flags if _overlaps(f, defect)]
        if hits:
            caught += 1
            type_score.caught += 1
            if any(f.verdict is defect.expected_verdict for f in hits):
                type_score.verdict_correct += 1

    true_flags = sum(1 for f in flags if any(_overlaps(f, d) for d in defects))

    return EvalScore(
        planted=len(defects),
        caught=caught,
        true_flags=true_flags,
        false_flags=len(flags) - true_flags,
        uncheckable=uncheckable,
        by_type=by_type,
    )
