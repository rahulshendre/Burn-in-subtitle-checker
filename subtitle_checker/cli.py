"""Command-line entrypoint for the burn-in subtitle checker."""

from __future__ import annotations

import argparse
import sys

from subtitle_checker import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subtitle-checker",
        description="Flag mismatches between audio dialogue and burned-in subtitles.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    check = subparsers.add_parser("check", help="Run the full pipeline on a video file")
    check.add_argument("--video", required=True, help="Path to the input video")
    check.add_argument("--lang", default="hi", help="ISO language code (hi, kn, mr)")
    check.add_argument("--out", default="out", help="Directory for artifacts and the report")

    ev = subparsers.add_parser(
        "eval-detection",
        help="Burn known lines onto a clean clip and measure what Stage 1 detects back",
    )
    ev.add_argument("--clean-clip", required=True, help="Video with no burned-in subtitles")
    ev.add_argument("--lang", default="hi", help="ISO language code for OCR")
    ev.add_argument("--out", default="out", help="Directory for the burned clip")

    es = subparsers.add_parser(
        "eval-structural",
        help="Plant drop/extra defects on truth lines and score the structural flags",
    )
    es.add_argument("--seed", type=int, default=0, help="RNG seed for defect planting")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        return _run_check(args)
    if args.command == "eval-detection":
        return _run_eval_detection(args)
    if args.command == "eval-structural":
        return _run_eval_structural(args)

    parser.print_help()
    return 0


def _run_check(args: argparse.Namespace) -> int:
    from pathlib import Path

    video = Path(args.video)
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2

    from subtitle_checker.artifacts import save_artifact
    from subtitle_checker.subtitles.ocr import EasyOcrEngine
    from subtitle_checker.subtitles.reconstruct import reconstruct_subtitles

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = reconstruct_subtitles(video, engine=EasyOcrEngine([args.lang]))
    artifact = out_dir / f"{video.stem}_subtitle_events.json"
    save_artifact(artifact, "subtitle_events", events)

    readable = sum(1 for e in events if e.text.strip())
    print(f"{len(events)} subtitle events ({readable} with text) -> {artifact}")
    for event in events:
        text = event.text.strip() or "<unreadable>"
        print(f"  {event.start:7.2f}-{event.end:7.2f}  [{event.confidence:.2f}]  {text}")
    return 0


def _run_eval_detection(args: argparse.Namespace) -> int:
    from pathlib import Path

    clip = Path(args.clean_clip)
    if not clip.exists():
        print(f"video not found: {clip}", file=sys.stderr)
        return 2

    from subtitle_checker.evaluation.detection import evaluate_detection
    from subtitle_checker.subtitles.ocr import EasyOcrEngine

    report = evaluate_detection(clip, Path(args.out), engine=EasyOcrEngine([args.lang]))

    for m in report.matches:
        print(
            f"MATCH {m.truth.start:6.2f}-{m.truth.end:6.2f} -> "
            f"{m.detected.start:6.2f}-{m.detected.end:6.2f}  sim={m.similarity:.2f}"
        )
        print(f"      truth: {m.truth.text}")
        print(f"      ocr:   {m.detected.text}")
    for tr in report.missed:
        print(f"MISS  {tr.start:6.2f}-{tr.end:6.2f}  {tr.text}")
    for d in report.strays:
        print(f"STRAY {d.start:6.2f}-{d.end:6.2f}  [{d.confidence:.2f}] {d.text!r}")

    print(f"\nrecall: {len(report.matches)}/{report.truth_count}")
    print(f"mean text similarity: {report.mean_similarity:.3f}")
    print(f"mean |start error|: {report.mean_start_error:.2f}s")
    print(f"mean |end error|:   {report.mean_end_error:.2f}s")
    return 0


def _run_eval_structural(args: argparse.Namespace) -> int:
    from subtitle_checker.evaluation.detection import DEFAULT_TRUTH_LINES, make_truth
    from subtitle_checker.evaluation.structural_eval import evaluate_structural

    truth = make_truth(DEFAULT_TRUTH_LINES)
    score, results = evaluate_structural(truth, seed=args.seed)

    for r in results:
        text = r.subtitle_text or "<no subtitle>"
        print(f"{r.verdict.value:17} {r.start:6.2f}-{r.end:6.2f}  {text}")
    print(f"\nrecall:    {score.caught}/{score.planted}")
    print(f"precision: {score.precision:.2f}  ({score.false_flags} false flags)")
    for name, ts in sorted(score.by_type.items()):
        print(f"  {name:12} caught {ts.caught}/{ts.planted}, verdict ok {ts.verdict_correct}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
