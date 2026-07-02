"""One-call injector: clean clip + truth subtitles → labelled defective test video."""

from __future__ import annotations

import subprocess
from pathlib import Path

from subtitle_checker.artifacts import SubtitleEvent, save_artifact
from subtitle_checker.evaluation.burn import DEFAULT_FONT, burn_subtitles
from subtitle_checker.evaluation.defects import MAX_SHIFT_S, plan_defects, save_defects


def _video_duration(video: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(proc.stdout.strip())


def make_test_video(
    video: Path,
    truth_events: list[SubtitleEvent],
    out_dir: Path,
    seed: int = 0,
    font: str = DEFAULT_FONT,
) -> tuple[Path, Path]:
    """Burn a defective subtitle track onto ``video``; return (video, labels) paths.

    ``truth_events`` must be the verified-correct subtitles for the clip.
    Three sibling files land in ``out_dir`` (suffixed with the seed): the
    defective video, the defect labels the scorer reads, and the mutated
    subtitle track that was burned (for debugging the burn itself).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video).stem

    # A defect planted past the end of the video silently never renders and
    # would be scored as "missed" — corrupting the numbers, not testing them.
    duration = _video_duration(Path(video))
    last_end = max(e.end for e in truth_events)
    if last_end + MAX_SHIFT_S > duration:
        raise ValueError(
            f"truth events end at {last_end:.2f}s but video is {duration:.2f}s; "
            f"events must stop at least {MAX_SHIFT_S}s before the end"
        )

    mutated, defects = plan_defects(truth_events, seed=seed)
    out_video = burn_subtitles(video, mutated, out_dir / f"{stem}_defective_seed{seed}.mp4", font)

    labels_path = out_dir / f"{stem}_defects_seed{seed}.json"
    save_defects(labels_path, defects)
    save_artifact(out_dir / f"{stem}_burned_subs_seed{seed}.json", "subtitle_events", mutated)
    return out_video, labels_path
