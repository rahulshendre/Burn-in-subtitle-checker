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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        return _run_check(args)

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


if __name__ == "__main__":
    raise SystemExit(main())
