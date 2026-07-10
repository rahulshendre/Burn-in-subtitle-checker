"""Command-line entrypoint for the burn-in subtitle checker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from subtitle_checker import __version__

# uroman (used by the forced aligner) wants ISO 639-3; the CLI speaks 639-1.
_UROMAN_LANG = {"hi": "hin", "kn": "kan", "mr": "mar"}
# Sarvam wants BCP-47 codes; the CLI speaks 639-1.
_SARVAM_LANG = {"hi": "hi-IN", "kn": "kn-IN", "mr": "mr-IN"}


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
    check.add_argument(
        "--asr",
        action="store_true",
        help="Also run the Sarvam ASR cross-check for word-level errors (needs SARVAM_API_KEY)",
    )

    rep = subparsers.add_parser(
        "report", help="Render a saved check into a self-contained HTML report"
    )
    rep.add_argument(
        "--results",
        required=True,
        help="A check_results.json file, or the out/<stem>/ directory holding it",
    )
    rep.add_argument("--video", required=True, help="Source video for frame + audio snippets")
    rep.add_argument("--out", help="Output HTML path (default: beside the results)")

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

    ea = subparsers.add_parser(
        "eval-alignment",
        help="Swap words on a clip's real lines and measure alignment separation",
    )
    ea.add_argument("--video", required=True, help="A real clip with burned-in subtitles")
    ea.add_argument("--lang", default="hi", help="ISO language code (hi, kn, mr)")
    ea.add_argument("--min-ocr-conf", type=float, default=0.5, help="Trust OCR text above this")
    ea.add_argument("--seed", type=int, default=0, help="RNG seed for the word swap")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        return _run_check(args)
    if args.command == "report":
        return _run_report(args)
    if args.command == "eval-detection":
        return _run_eval_detection(args)
    if args.command == "eval-structural":
        return _run_eval_structural(args)
    if args.command == "eval-alignment":
        return _run_eval_alignment(args)

    parser.print_help()
    return 0


def _run_check(args: argparse.Namespace) -> int:
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

    _run_audio_checks(video, events, out_dir, args.lang, args.asr)
    return 0


def _run_report(args: argparse.Namespace) -> int:
    video = Path(args.video)
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2

    results_path = _resolve_results(Path(args.results), video)
    if results_path is None:
        print(f"no check_results found at: {args.results}", file=sys.stderr)
        return 2

    from subtitle_checker.artifacts import load_artifact
    from subtitle_checker.report.evidence import write_report

    kind, results = load_artifact(results_path)
    if kind != "check_results":
        print(f"not a check_results artifact: {results_path}", file=sys.stderr)
        return 2

    out = Path(args.out) if args.out else results_path.parent / f"{video.stem}_report.html"
    skipped = _load_skipped(results_path, video, results)
    write_report(video, results, out, title=f"Subtitle check — {video.stem}", skipped=skipped)
    print(f"report -> {out}  ({len(results)} row(s))")
    return 0


def _load_skipped(results_path: Path, video: Path, results: list) -> list | None:
    """Rebuild the skipped-lines list from sibling artifacts when they exist."""
    from subtitle_checker.artifacts import load_artifact
    from subtitle_checker.match.asr import skipped_lines

    events_path = results_path.parent / f"{video.stem}_subtitle_events.json"
    if not events_path.exists():
        return None
    _, events = load_artifact(events_path)
    regions_path = results_path.parent / f"{video.stem}_audio_regions.json"
    regions = load_artifact(regions_path)[1] if regions_path.exists() else None
    return skipped_lines(events, results, regions)


def _resolve_results(path: Path, video: Path) -> Path | None:
    """Accept a check_results.json file directly, or a directory holding one."""
    if path.is_file():
        return path
    if path.is_dir():
        named = path / f"{video.stem}_check_results.json"
        if named.exists():
            return named
        found = sorted(path.glob("*_check_results.json"))
        if found:
            return found[0]
    return None


def _run_audio_checks(video: Path, events: list, out_dir: Path, lang: str, run_asr: bool) -> None:
    """Stage 2 + 3: label the audio, raise flags, transcribe lines, write the report."""
    from subtitle_checker.artifacts import save_artifact
    from subtitle_checker.audio.regions import label_regions
    from subtitle_checker.ingest.audio_track import extract_audio
    from subtitle_checker.match.structural import check_structural

    try:
        from subtitle_checker.audio.vad import SileroVad
        vad = SileroVad()
        audio = extract_audio(video)
        regions = label_regions(audio, vad)
    except ImportError:
        print("audio stage skipped — install the extra with: pip install '.[audio]'")
        return

    save_artifact(out_dir / f"{video.stem}_audio_regions.json", "audio_regions", regions)
    flags = check_structural(events, regions)
    flags += _alignment_flags(events, audio, regions, lang)
    results = flags
    if run_asr:
        results = _merge_results(flags, _asr_ledger(events, audio, regions, lang))
    results.sort(key=lambda r: r.start)
    save_artifact(out_dir / f"{video.stem}_check_results.json", "check_results", results)

    from subtitle_checker.match.asr import skipped_lines

    _print_flags(results)
    _write_report(video, results, out_dir, skipped_lines(events, results, regions))


def _alignment_flags(events: list, audio, regions: list, lang: str) -> list:
    """Stage 3: score each speech-covered line's text against the audio."""
    from subtitle_checker.match.align import MmsAligner, score_events
    from subtitle_checker.match.verdicts import check_alignment

    try:
        aligner = MmsAligner(lang=_UROMAN_LANG.get(lang, "hin"))
        scores = score_events(events, audio, aligner)
    except ImportError:
        print("alignment stage skipped — install the extra with: pip install '.[align]'")
        return []
    return check_alignment(scores, regions)


def _asr_ledger(events: list, audio, regions: list, lang: str) -> list:
    """Stage 3 secondary: transcribe each trusted line for flags + report ledger."""
    import os

    if not os.environ.get("SARVAM_API_KEY"):
        print("ASR cross-check skipped — set SARVAM_API_KEY to enable it")
        return []
    from subtitle_checker.match.asr import SarvamAsr, transcribe_lines

    try:
        engine = SarvamAsr(lang=_SARVAM_LANG.get(lang, "hi-IN"))
        return transcribe_lines(events, audio, regions, engine)
    except ImportError:
        print("ASR cross-check skipped — install the extra with: pip install '.[asr]'")
        return []


def _merge_results(flags: list, ledger: list) -> list:
    """Combine flags with the ASR ledger into the saved check_results.

    An ASR row carries heard-vs-written evidence a bare alignment flag lacks, so
    an ASR TEXT_MISMATCH supersedes an alignment flag on the same line. Otherwise
    the remaining flags win their span, and the leftover ledger rows (the OK
    heard-vs-written scan) fill in the lines nothing flagged.
    """
    from subtitle_checker.artifacts import Verdict

    asr_mismatch = {(r.start, r.end) for r in ledger if r.verdict is Verdict.TEXT_MISMATCH}
    flags = [f for f in flags if (f.start, f.end) not in asr_mismatch]
    kept = {(f.start, f.end) for f in flags}
    return flags + [r for r in ledger if (r.start, r.end) not in kept]


def _print_flags(results: list) -> None:
    from subtitle_checker.artifacts import Verdict

    flags = [r for r in results if r.verdict is not Verdict.OK]
    heard = len(results) - len(flags)
    tail = f", {heard} line(s) transcribed:" if heard else ":"
    print(f"\n{len(flags)} flag(s)" + tail)
    for f in flags:
        text = f.subtitle_text.strip() or "<no subtitle>"
        print(f"  {f.verdict.value:17} {f.start:7.2f}-{f.end:7.2f}  {text}")


def _write_report(video: Path, results: list, out_dir: Path, skipped: list | None = None) -> None:
    from subtitle_checker.report.evidence import write_report

    path = out_dir / f"{video.stem}_report.html"
    write_report(video, results, path, title=f"Subtitle check — {video.stem}", skipped=skipped)
    print(f"report -> {path}")


def _run_eval_detection(args: argparse.Namespace) -> int:
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


def _run_eval_alignment(args: argparse.Namespace) -> int:
    video = Path(args.video)
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2

    from subtitle_checker.evaluation.alignment_eval import evaluate_alignment
    from subtitle_checker.ingest.audio_track import extract_audio
    from subtitle_checker.match.align import MmsAligner
    from subtitle_checker.subtitles.ocr import EasyOcrEngine
    from subtitle_checker.subtitles.reconstruct import reconstruct_subtitles

    events = reconstruct_subtitles(video, engine=EasyOcrEngine([args.lang]))
    audio = extract_audio(video)
    aligner = MmsAligner(lang=_UROMAN_LANG.get(args.lang, "hin"))
    result = evaluate_alignment(
        events, audio, aligner, min_ocr_conf=args.min_ocr_conf, seed=args.seed
    )

    print(f"\n{result.pairs} trusted line(s) scored correct vs word-swapped")
    print(f"correct mean: {result.correct_mean:.3f}   swapped mean: {result.swapped_mean:.3f}")
    print(f"best threshold: {result.threshold:.3f}")
    print(f"recall: {result.recall:.2f}   precision: {result.precision:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
