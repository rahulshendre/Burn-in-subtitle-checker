"""Sample the subtitle band of a video as grayscale frames.

Frames stream straight out of an ffmpeg rawvideo pipe — nothing lands on
disk. The detection pass reads a downscaled band; OCR re-extracts single
frames at native resolution.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The subtitle band: bottom 30% of the frame.
DEFAULT_BAND_TOP = 0.70
DEFAULT_FPS = 4.0
DEFAULT_OUT_WIDTH = 640


@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float


def probe(video: Path) -> VideoInfo:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    stream = data["streams"][0]
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        duration=float(data["format"]["duration"]),
    )


def band_crop(height: int, band_top: float) -> tuple[int, int]:
    """(crop height, y offset) for the subtitle band, even-sized for codecs."""
    crop_h = int(height * (1.0 - band_top))
    crop_h -= crop_h % 2
    return crop_h, height - crop_h


def iter_band_frames(
    video: Path,
    fps: float = DEFAULT_FPS,
    band_top: float = DEFAULT_BAND_TOP,
    out_width: int = DEFAULT_OUT_WIDTH,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp, grayscale band) at ``fps`` for the whole video."""
    info = probe(video)
    crop_h, y = band_crop(info.height, band_top)
    out_h = int(crop_h * out_width / info.width)
    out_h -= out_h % 2
    vf = f"crop=iw:{crop_h}:0:{y},fps={fps},scale={out_width}:{out_h}"
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video),
        "-vf",
        vf,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    frame_bytes = out_width * out_h
    index = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(out_h, out_width)
            yield index / fps, frame
            index += 1
    finally:
        proc.stdout.close()
        proc.wait()


def extract_band_frame(
    video: Path, t: float, band_top: float = DEFAULT_BAND_TOP
) -> np.ndarray:
    """Native-resolution grayscale band at time ``t`` (for OCR)."""
    info = probe(video)
    crop_h, y = band_crop(info.height, band_top)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{t:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-vf",
        f"crop=iw:{crop_h}:0:{y}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    if len(out) < crop_h * info.width:
        raise ValueError(f"no frame at {t:.3f}s in {video}")
    return np.frombuffer(out[: crop_h * info.width], dtype=np.uint8).reshape(crop_h, info.width)
