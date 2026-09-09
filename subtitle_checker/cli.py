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
    check.add_argument(
        "--ocr",
        choices=["easyocr", "sarvam-vision"],
        default="easyocr",
        help="OCR engine: easyocr (on-device default) or sarvam-vision "
        "(cloud quality engine, needs SARVAM_API_KEY)",
    )

    cs = subparsers.add_parser(
        "check-script",
        help="Check audio against an authored subtitle file (SRT/VTT) - ASR only, no OCR",
    )
    cs.add_argument("--video", required=True, help="Path to the input video (audio source)")
    cs.add_argument("--script", required=True, help="Authored subtitle file (SRT or VTT)")
    cs.add_argument("--lang", default="hi", help="ISO language code (hi, kn, mr)")
    cs.add_argument("--out", default="out", help="Directory for artifacts and the report")

    leg = subparsers.add_parser(
        "legibility",
        help="Grade a video's subtitle legibility (Stage 1 only, no audio needed)",
    )
    leg.add_argument("--video", required=True, help="Path to the input video")
    leg.add_argument("--lang", default="hi", help="ISO language code for OCR (hi, kn, mr)")
    leg.add_argument("--out", default="out", help="Directory for the events artifact")
    leg.add_argument(
        "--ocr",
        choices=["easyocr", "sarvam-vision"],
        default="easyocr",
        help="OCR engine used to read the worst lines (easyocr is local + free)",
    )
    leg.add_argument(
        "--worst", type=int, default=None, help="How many least legible lines to list"
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

    ui = subparsers.add_parser(
        "ui", help="Serve the pre-built reports in a minimal local web UI"
    )
    ui.add_argument(
        "paths",
        nargs="*",
        default=["out"],
        help="Report files or directories to list (default: out)",
    )
    ui.add_argument("--port", type=int, default=8000, help="Local port to serve on")
    ui.add_argument("--no-open", action="store_true", help="Do not open a browser window")
    ui.add_argument(
        "--label",
        action="append",
        help="Friendly label for each path, in order (repeatable)",
    )
    ui.add_argument(
        "--demo",
        action="store_true",
        help="Demo layout: a video picker plus a transcript (SRT) upload, then Check",
    )
    ui.add_argument(
        "--srt",
        action="append",
        help="Transcript (SRT) filename shown per video, in order (repeatable)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        return _run_check(args)
    if args.command == "check-script":
        return _run_script_check(args)
    if args.command == "legibility":
        return _run_legibility(args)
    if args.command == "report":
        return _run_report(args)
    if args.command == "eval-detection":
        return _run_eval_detection(args)
    if args.command == "eval-structural":
        return _run_eval_structural(args)
    if args.command == "eval-alignment":
        return _run_eval_alignment(args)
    if args.command == "ui":
        return _run_ui(args)

    parser.print_help()
    return 0


def _run_ui(args: argparse.Namespace) -> int:
    from subtitle_checker.report.webui import (
        attach_transcripts,
        discover_reports,
        relabel,
        serve,
    )

    entries = relabel(discover_reports([Path(p) for p in args.paths]), args.label or [])
    if args.demo:
        entries = attach_transcripts(entries, args.srt or [])
    if not entries:
        print(f"no reports found in: {', '.join(args.paths)}", file=sys.stderr)
        print("generate one first with: subtitle-checker check --video <clip>", file=sys.stderr)
    serve(entries, port=args.port, open_browser=not args.no_open, demo=args.demo)
    return 0


def _require_video(path_str: str) -> Path | None:
    """Return the video path if it exists, else print an error and return None."""
    video = Path(path_str)
    if video.exists():
        return video
    print(f"video not found: {video}", file=sys.stderr)
    return None


def _report_title(video: Path) -> str:
    return f"Subtitle check - {video.stem}"


def _sibling_artifact(results_path: Path, video: Path, name: str) -> Path:
    """Path to a stage artifact saved next to the check results."""
    return results_path.parent / f"{video.stem}_{name}.json"


def _build_ocr(name: str, lang: str):
    """Build the OCR engine named on the command line."""
    if name == "sarvam-vision":
        from subtitle_checker.subtitles.ocr import SarvamVisionOcr

        return SarvamVisionOcr(lang=_SARVAM_LANG.get(lang, "hi-IN"))
    from subtitle_checker.subtitles.ocr import EasyOcrEngine

    return EasyOcrEngine([lang])


def _run_check(args: argparse.Namespace) -> int:
    video = _require_video(args.video)
    if video is None:
        return 2

    from subtitle_checker.artifacts import save_artifact
    from subtitle_checker.subtitles.reconstruct import reconstruct_subtitles

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = reconstruct_subtitles(video, engine=_build_ocr(args.ocr, args.lang))
    artifact = out_dir / f"{video.stem}_subtitle_events.json"
    save_artifact(artifact, "subtitle_events", events)

    readable = sum(1 for e in events if e.text.strip())
    print(f"{len(events)} subtitle events ({readable} with text) -> {artifact}")
    for event in events:
        text = event.text.strip() or "<unreadable>"
        print(f"  {event.start:7.2f}-{event.end:7.2f}  [{event.confidence:.2f}]  {text}")

    _print_legibility(events)
    _print_compliance(events)
    _run_audio_checks(video, events, out_dir, args.lang, args.asr)
    return 0


def _run_script_check(args: argparse.Namespace) -> int:
    """Audio-vs-script pipeline: compare the audio to an authored SRT/VTT file.

    No OCR and no legibility - the reference text is the authored subtitle file,
    fully trusted, so this checks only whether the spoken audio matches what the
    script says (the ASR cross-check) plus the structural gaps (a scripted line
    with no speech under it, or speech with no scripted line).
    """
    video = _require_video(args.video)
    if video is None:
        return 2
    script = Path(args.script)
    if not script.exists():
        print(f"script not found: {script}", file=sys.stderr)
        return 2

    from subtitle_checker.artifacts import save_artifact
    from subtitle_checker.subtitles.srt import parse_script

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = parse_script(script)
    if not events:
        print(f"no subtitle cues parsed from: {script}", file=sys.stderr)
        return 2
    artifact = out_dir / f"{video.stem}_subtitle_events.json"
    save_artifact(artifact, "subtitle_events", events)
    print(f"{len(events)} script cue(s) -> {artifact}")
    for event in events:
        print(f"  {event.start:7.2f}-{event.end:7.2f}  {event.text}")

    _run_script_audio_checks(video, events, out_dir, args.lang)
    return 0


def _run_script_audio_checks(video: Path, events: list, out_dir: Path, lang: str) -> None:
    """Whole-transcript audio-vs-script check: transcribe the full audio once,
    align it to the script word-for-word, write the per-cue heard-vs-script report.

    No per-cue windowing and no VAD - the global alignment re-attributes heard
    words to the cue they belong to, so a sentence split across short cues or a
    small intro timing offset no longer manufactures false mismatches.
    """
    import os

    if not os.environ.get("SARVAM_API_KEY"):
        print(
            "ASR skipped - set SARVAM_API_KEY to run the audio-vs-script check",
            file=sys.stderr,
        )
        return

    from subtitle_checker.artifacts import save_artifact
    from subtitle_checker.ingest.audio_track import extract_audio
    from subtitle_checker.match.asr import SarvamAsr
    from subtitle_checker.match.script_align import align_script, transcribe_full

    audio = extract_audio(video)
    engine = SarvamAsr(lang=_SARVAM_LANG.get(lang, "hi-IN"))
    heard = transcribe_full(audio, engine)
    results = align_script(events, heard)
    save_artifact(out_dir / f"{video.stem}_check_results.json", "check_results", results)

    _print_flags(results)
    _write_report(
        video, results, out_dir,
        title=f"Audio vs SRT - {video.stem}",
        source_label="SRT",
    )


def _run_legibility(args: argparse.Namespace) -> int:
    video = _require_video(args.video)
    if video is None:
        return 2

    from subtitle_checker.artifacts import save_artifact
    from subtitle_checker.subtitles.reconstruct import reconstruct_subtitles

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = reconstruct_subtitles(video, engine=_build_ocr(args.ocr, args.lang))
    save_artifact(out_dir / f"{video.stem}_subtitle_events.json", "subtitle_events", events)
    if _print_legibility(events, args.worst) is None:
        print("no readable subtitle lines to grade")
    return 0


def _print_legibility(events: list, worst_n: int | None = None):
    """Print the whole-video legibility grade and its least legible lines."""
    from subtitle_checker.subtitles.legibility import (
        BAND_CLEAR,
        DEFAULT_WORST_N,
        video_legibility,
    )

    grade = video_legibility(events, worst_n or DEFAULT_WORST_N)
    if grade is None:
        return None
    print(f"\nlegibility: {grade.score:.0f}/100 over {grade.line_count} line(s)")
    for band in grade.bands:
        mins, secs = divmod(int(round(band.seconds)), 60)
        clock = f"{mins}m{secs:02d}s"
        print(f"  {band.label:5s} ({band.share * 100:3.0f}%)  {clock}")
    low = [line for line in grade.worst if line.score < BAND_CLEAR]
    if low:
        print("least legible lines:")
        for line in low:
            text = line.text.strip() or "<unreadable>"
            print(f"  {line.score:5.0f}  {line.start:7.2f}-{line.end:7.2f}  {text}")
    _print_recommendations(events)
    return grade


def _print_recommendations(events: list) -> None:
    """Print the actionable fixes for below-Clear lines, when any apply."""
    from subtitle_checker.subtitles.recommend import advice_summary, legibility_advice

    advice = legibility_advice(events)
    if not advice:
        return
    print(f"how to improve legibility: {advice_summary(advice)}")
    for a in advice:
        text = a.text.strip() or "<unreadable>"
        print(f"  {a.start:7.2f}-{a.end:7.2f}  {text}")
        for tip in a.tips:
            print(f"      - {tip}")


def _print_compliance(events: list):
    """Print how many subtitle lines follow the checkable guideline caps."""
    from subtitle_checker.subtitles.compliance import (
        MAX_CHARS_ON_SCREEN,
        check_compliance,
    )

    comp = check_compliance(events)
    if comp is None:
        return None
    print(
        f"\nguideline compliance: {comp.compliant}/{comp.graded} lines "
        f"({comp.share * 100:.0f}%)"
    )
    print(f"  <={MAX_CHARS_ON_SCREEN} chars on screen: {comp.chars_pass}/{comp.chars_measured}")
    for v in sorted(comp.violations, key=lambda v: v.start):
        text = v.text or "<unreadable>"
        print(f"  {v.start:7.2f}-{v.end:7.2f}  {'; '.join(v.failures())}  {text}")
    return comp


def _run_report(args: argparse.Namespace) -> int:
    video = _require_video(args.video)
    if video is None:
        return 2

    results_path = _resolve_results(Path(args.results), video)
    if results_path is None:
        print(f"no check_results found at: {args.results}", file=sys.stderr)
        return 2

    from subtitle_checker.artifacts import load_artifact
    from subtitle_checker.report.evidence import write_mistakes, write_report

    kind, results = load_artifact(results_path)
    if kind != "check_results":
        print(f"not a check_results artifact: {results_path}", file=sys.stderr)
        return 2

    out = Path(args.out) if args.out else results_path.parent / f"{video.stem}_report.html"
    skipped = _load_skipped(results_path, video, results)
    legibility = _load_legibility(results_path, video)
    compliance = _load_compliance(results_path, video)
    recommendations = _load_recommendations(results_path, video)
    write_report(
        video, results, out,
        title=_report_title(video), skipped=skipped,
        legibility=legibility, compliance=compliance,
        recommendations=recommendations,
    )
    print(f"report -> {out}  ({len(results)} row(s))")

    mistakes = out.parent / f"{video.stem}_mistakes.html"
    write_mistakes(
        video, results, mistakes, title=f"Subtitle mistakes - {video.stem}"
    )
    print(f"mistakes -> {mistakes}")
    return 0


def _load_legibility(results_path: Path, video: Path) -> object | None:
    """Grade legibility from the sibling events artifact when it exists."""
    from subtitle_checker.artifacts import load_artifact
    from subtitle_checker.subtitles.legibility import video_legibility

    events_path = _sibling_artifact(results_path, video, "subtitle_events")
    if not events_path.exists():
        return None
    return video_legibility(load_artifact(events_path)[1])


def _load_recommendations(results_path: Path, video: Path) -> list | None:
    """Rebuild legibility recommendations from the sibling events artifact."""
    from subtitle_checker.artifacts import load_artifact
    from subtitle_checker.subtitles.recommend import legibility_advice

    events_path = _sibling_artifact(results_path, video, "subtitle_events")
    if not events_path.exists():
        return None
    return legibility_advice(load_artifact(events_path)[1])


def _load_compliance(results_path: Path, video: Path) -> object | None:
    """Grade guideline compliance from the sibling events artifact when it exists."""
    from subtitle_checker.artifacts import load_artifact
    from subtitle_checker.subtitles.compliance import check_compliance

    events_path = _sibling_artifact(results_path, video, "subtitle_events")
    if not events_path.exists():
        return None
    return check_compliance(load_artifact(events_path)[1])


def _load_skipped(results_path: Path, video: Path, results: list) -> list | None:
    """Rebuild the skipped-lines list from sibling artifacts when they exist."""
    from subtitle_checker.artifacts import load_artifact
    from subtitle_checker.match.asr import skipped_lines

    events_path = _sibling_artifact(results_path, video, "subtitle_events")
    if not events_path.exists():
        return None
    _, events = load_artifact(events_path)
    regions_path = _sibling_artifact(results_path, video, "audio_regions")
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
        print("audio stage skipped - install the extra with: pip install '.[audio]'")
        return

    save_artifact(out_dir / f"{video.stem}_audio_regions.json", "audio_regions", regions)
    flags = check_structural(events, regions)
    flags += _alignment_flags(events, audio, regions, lang)
    results = flags
    if run_asr:
        results = _merge_results(flags, _asr_ledger(events, audio, regions, lang))
        results = _fill_missing_text(results, audio, lang)
    results.sort(key=lambda r: r.start)
    save_artifact(out_dir / f"{video.stem}_check_results.json", "check_results", results)

    from subtitle_checker.match.asr import skipped_lines
    from subtitle_checker.subtitles.compliance import check_compliance
    from subtitle_checker.subtitles.legibility import video_legibility
    from subtitle_checker.subtitles.recommend import legibility_advice

    _print_flags(results)
    _write_report(
        video, results, out_dir,
        skipped=skipped_lines(events, results, regions),
        legibility=video_legibility(events),
        compliance=check_compliance(events),
        recommendations=legibility_advice(events),
    )


def _alignment_flags(events: list, audio, regions: list, lang: str) -> list:
    """Stage 3: score each speech-covered line's text against the audio."""
    from subtitle_checker.match.align import MmsAligner, score_events
    from subtitle_checker.match.verdicts import check_alignment

    try:
        aligner = MmsAligner(lang=_UROMAN_LANG.get(lang, "hin"))
        scores = score_events(events, audio, aligner)
    except ImportError:
        print("alignment stage skipped - install the extra with: pip install '.[align]'")
        return []
    return check_alignment(scores, regions)


def _asr_ledger(events: list, audio, regions: list, lang: str) -> list:
    """Stage 3 secondary: transcribe each trusted line for flags + report ledger."""
    import os

    if not os.environ.get("SARVAM_API_KEY"):
        print("ASR cross-check skipped - set SARVAM_API_KEY to enable it")
        return []
    from subtitle_checker.match.asr import SarvamAsr, transcribe_lines

    try:
        engine = SarvamAsr(lang=_SARVAM_LANG.get(lang, "hi-IN"))
        return transcribe_lines(events, audio, regions, engine)
    except ImportError:
        print("ASR cross-check skipped - install the extra with: pip install '.[asr]'")
        return []


def _fill_missing_text(results: list, audio, lang: str) -> list:
    """Transcribe the audio under each MISSING_SUBTITLE span for a best-guess.

    A missing-subtitle flag knows speech is there but never transcribed it, so the
    report can only say "not transcribed". This fills in what the audio says, as an
    unverified suggestion the editor can confirm. Gated on the same Sarvam key as
    the ASR cross-check; other verdicts pass through untouched.
    """
    import os

    if not os.environ.get("SARVAM_API_KEY"):
        return results
    from subtitle_checker.match.asr import SarvamAsr, transcribe_missing

    try:
        engine = SarvamAsr(lang=_SARVAM_LANG.get(lang, "hi-IN"))
        return transcribe_missing(results, audio, engine)
    except ImportError:
        return results


def _merge_results(flags: list, ledger: list) -> list:
    """Combine flags with the ASR ledger into the saved check_results.

    The ASR ledger is the word-level authority for any line it transcribed: it
    heard the audio and carries heard-vs-written evidence, so its verdict wins
    over an alignment flag on the same span. An ASR OK clears a stale alignment
    mismatch - alignment mis-scores a correct line under a heavy background score
    or a clipped detection span, and if the words were truly absent the ASR could
    not have heard them match - and an ASR TEXT_MISMATCH replaces the bare flag
    with the evidence. Structural flags (MISSING / ORPHAN / UNCHECKABLE) sit on
    gap spans the ledger never covers, so they are left untouched.
    """
    ledger_spans = {(r.start, r.end) for r in ledger}
    flags = [f for f in flags if (f.start, f.end) not in ledger_spans]
    return flags + ledger


def _print_flags(results: list) -> None:
    from subtitle_checker.artifacts import Verdict

    flags = [r for r in results if r.verdict is not Verdict.OK]
    heard = len(results) - len(flags)
    tail = f", {heard} line(s) transcribed:" if heard else ":"
    print(f"\n{len(flags)} flag(s)" + tail)
    for f in flags:
        text = f.subtitle_text.strip() or "<no subtitle>"
        print(f"  {f.verdict.value:17} {f.start:7.2f}-{f.end:7.2f}  {text}")


def _write_report(
    video: Path,
    results: list,
    out_dir: Path,
    skipped: list | None = None,
    legibility: object | None = None,
    compliance: object | None = None,
    recommendations: list | None = None,
    title: str | None = None,
    source_label: str = "OCR",
) -> None:
    from subtitle_checker.report.evidence import write_mistakes, write_report

    path = out_dir / f"{video.stem}_report.html"
    write_report(
        video, results, path,
        title=title or _report_title(video), skipped=skipped,
        legibility=legibility, compliance=compliance,
        recommendations=recommendations, source_label=source_label,
    )
    print(f"report -> {path}")

    mistakes_path = out_dir / f"{video.stem}_mistakes.html"
    write_mistakes(
        video, results, mistakes_path,
        title=f"Subtitle mistakes - {video.stem}", source_label=source_label,
    )
    print(f"mistakes -> {mistakes_path}")


def _run_eval_detection(args: argparse.Namespace) -> int:
    clip = _require_video(args.clean_clip)
    if clip is None:
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
    video = _require_video(args.video)
    if video is None:
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
