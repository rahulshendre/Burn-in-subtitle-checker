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
        print(
            "Pipeline stages are landing in separate PRs; nothing to run yet.",
            file=sys.stderr,
        )
        return 2

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
