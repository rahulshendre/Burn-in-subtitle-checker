"""Tests for the evaluation harness: defect planning, scoring, ASS rendering."""

from pathlib import Path

import pytest

from subtitle_checker.artifacts import CheckResult, SubtitleEvent, Verdict
from subtitle_checker.evaluation.burn import _ass_timestamp, events_to_ass
from subtitle_checker.evaluation.defects import (
    MIN_GAP_S,
    MIN_SHIFT_S,
    Defect,
    DefectType,
    load_defects,
    plan_defects,
    save_defects,
)
from subtitle_checker.evaluation.score import score_results

LINES = [
    "वो कहाँ गया था",
    "ठीक है भाई",
    "मुझे कुछ नहीं पता",
    "कल फिर आना यहाँ",
    "यह बहुत अच्छा है",
    "अब घर चलते हैं",
]


def make_truth(n: int = 6, gap: float = 2.0, duration: float = 2.5) -> list[SubtitleEvent]:
    events = []
    t = 0.0
    for i in range(n):
        events.append(SubtitleEvent(start=t, end=t + duration, text=LINES[i % len(LINES)]))
        t += duration + gap
    return events


def defect_of(defects: list[Defect], kind: DefectType) -> Defect:
    matches = [d for d in defects if d.type is kind]
    assert len(matches) == 1
    return matches[0]


class TestPlanDefects:
    def test_deterministic_for_same_seed(self) -> None:
        truth = make_truth()
        assert plan_defects(truth, seed=7) == plan_defects(truth, seed=7)

    def test_one_defect_of_each_type_and_event_count_balances(self) -> None:
        truth = make_truth()
        mutated, defects = plan_defects(truth, seed=1)
        assert sorted(d.type.value for d in defects) == sorted(t.value for t in DefectType)
        # one line dropped, one line added
        assert len(mutated) == len(truth)

    def test_word_swap_changes_exactly_one_word(self) -> None:
        _, defects = plan_defects(make_truth(), seed=2)
        swap = defect_of(defects, DefectType.WORD_SWAP)
        before, after = swap.original_text.split(), swap.mutated_text.split()
        assert len(before) == len(after)
        assert sum(1 for a, b in zip(before, after) if a != b) == 1

    def test_timing_shift_preserves_duration_and_moves_line(self) -> None:
        truth = make_truth()
        mutated, defects = plan_defects(truth, seed=3)
        shift = defect_of(defects, DefectType.TIMING_SHIFT)
        # the defect span is the union of old and new position, so the
        # original event touches one edge of it
        original = next(
            e
            for e in truth
            if e.text == shift.original_text and (e.start == shift.start or e.end == shift.end)
        )
        duration = original.end - original.start
        moved = [
            e
            for e in mutated
            if e.text == shift.original_text
            and e.start != original.start
            and e.end - e.start == pytest.approx(duration)
        ]
        assert len(moved) == 1
        assert abs(moved[0].start - original.start) >= MIN_SHIFT_S
        assert shift.start <= moved[0].start and moved[0].end <= shift.end

    def test_dropped_line_is_absent_from_mutated(self) -> None:
        truth = make_truth()
        mutated, defects = plan_defects(truth, seed=4)
        drop = defect_of(defects, DefectType.DROP_LINE)
        assert not any(
            e.start == drop.start and e.text == drop.original_text for e in mutated
        )

    def test_extra_line_sits_inside_a_truth_gap(self) -> None:
        truth = make_truth(gap=MIN_GAP_S + 1.0)
        mutated, defects = plan_defects(truth, seed=5)
        extra = defect_of(defects, DefectType.EXTRA_LINE)
        assert any(e.start == extra.start and e.text == extra.mutated_text for e in mutated)
        ordered = sorted(truth, key=lambda e: e.start)
        in_gap = any(
            a.end <= extra.start and extra.end <= b.start
            for a, b in zip(ordered, ordered[1:])
        )
        assert in_gap

    def test_too_few_events_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 4"):
            plan_defects(make_truth(n=3))

    def test_no_silent_gap_rejected(self) -> None:
        packed = make_truth(gap=0.2)
        with pytest.raises(ValueError, match="silent gap"):
            plan_defects(packed)

    def test_labels_round_trip_through_json(self, tmp_path: Path) -> None:
        _, defects = plan_defects(make_truth(), seed=6)
        path = tmp_path / "defects.json"
        save_defects(path, defects)
        assert load_defects(path) == defects
        assert "वो" in path.read_text(encoding="utf-8") or "है" in path.read_text(encoding="utf-8")


def flag_for(defect: Defect, verdict: Verdict | None = None) -> CheckResult:
    return CheckResult(
        start=defect.start,
        end=defect.end,
        verdict=verdict or defect.expected_verdict,
        reason="planted",
    )


class TestScoreResults:
    def test_perfect_run_scores_one(self) -> None:
        _, defects = plan_defects(make_truth(), seed=8)
        results = [flag_for(d) for d in defects]
        score = score_results(results, defects)
        assert score.recall == 1.0
        assert score.precision == 1.0
        assert all(t.verdict_correct == t.planted for t in score.by_type.values())

    def test_missed_defect_lowers_recall(self) -> None:
        _, defects = plan_defects(make_truth(), seed=9)
        results = [flag_for(d) for d in defects[:-1]]
        score = score_results(results, defects)
        assert score.caught == len(defects) - 1
        assert score.recall == pytest.approx((len(defects) - 1) / len(defects))

    def test_stray_flag_lowers_precision(self) -> None:
        _, defects = plan_defects(make_truth(), seed=10)
        stray = CheckResult(start=500.0, end=502.0, verdict=Verdict.TEXT_MISMATCH, reason="stray")
        results = [flag_for(d) for d in defects] + [stray]
        score = score_results(results, defects)
        assert score.false_flags == 1
        assert score.precision == pytest.approx(len(defects) / (len(defects) + 1))

    def test_wrong_verdict_still_caught_but_tracked(self) -> None:
        _, defects = plan_defects(make_truth(), seed=11)
        drop = defect_of(defects, DefectType.DROP_LINE)
        results = [flag_for(d) for d in defects if d is not drop]
        results.append(flag_for(drop, verdict=Verdict.TEXT_MISMATCH))
        score = score_results(results, defects)
        assert score.recall == 1.0
        assert score.by_type[DefectType.DROP_LINE.value].verdict_correct == 0

    def test_uncheckable_is_not_a_false_flag(self) -> None:
        _, defects = plan_defects(make_truth(), seed=12)
        abstain = CheckResult(start=900.0, end=903.0, verdict=Verdict.UNCHECKABLE, reason="song")
        results = [flag_for(d) for d in defects] + [abstain]
        score = score_results(results, defects)
        assert score.false_flags == 0
        assert score.uncheckable == 1
        assert score.precision == 1.0

    def test_ok_results_are_ignored(self) -> None:
        _, defects = plan_defects(make_truth(), seed=13)
        ok = CheckResult(start=defects[0].start, end=defects[0].end, verdict=Verdict.OK, reason="")
        score = score_results([ok], defects)
        assert score.caught == 0
        assert score.true_flags == 0


class TestAssRendering:
    def test_timestamp_format(self) -> None:
        assert _ass_timestamp(0.0) == "0:00:00.00"
        assert _ass_timestamp(1.5) == "0:00:01.50"
        assert _ass_timestamp(3661.25) == "1:01:01.25"
        assert _ass_timestamp(-2.0) == "0:00:00.00"

    def test_script_contains_all_events_in_order(self) -> None:
        events = [
            SubtitleEvent(start=5.0, end=7.0, text="ठीक है भाई"),
            SubtitleEvent(start=1.0, end=3.0, text="वो कहाँ गया था"),
        ]
        script = events_to_ass(events)
        dialogue = [line for line in script.splitlines() if line.startswith("Dialogue:")]
        assert len(dialogue) == 2
        assert "वो कहाँ गया था" in dialogue[0]
        assert "ठीक है भाई" in dialogue[1]

    def test_control_characters_neutralised(self) -> None:
        script = events_to_ass([SubtitleEvent(0.0, 1.0, "brace {test} back\\slash")])
        assert "{test}" not in script
        assert "(test)" in script
        assert "back" in script and "\\slash" not in script
