"""Entry point of the packaged app (Mac and Windows).

A double-clicked app has no terminal, so the bundled ffmpeg/ffprobe are put on
PATH here and all output goes to a log file in the app's folder, where a tester
can find it when something goes wrong.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    bundle = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    os.environ["PATH"] = str(bundle / "bin") + os.pathsep + os.environ.get("PATH", "")

    from subtitle_checker.report.app import default_home, serve_app

    home = default_home()
    home.mkdir(parents=True, exist_ok=True)
    log = open(home / "app.log", "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    sys.stdout = sys.stderr = log
    serve_app(home)


if __name__ == "__main__":
    main()
