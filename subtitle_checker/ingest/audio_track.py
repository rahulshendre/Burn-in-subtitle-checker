"""Extract a video's audio track as mono 16 kHz float32 samples.

Samples stream straight out of an ffmpeg pipe — nothing lands on disk.
16 kHz mono is the native format for VAD and every ASR/alignment backend
downstream, so the conversion happens exactly once, here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


def extract_audio(video: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode the first audio stream to mono float32 in [-1, 1]."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    # frombuffer is a read-only view of the pipe bytes; copy so downstream
    # consumers (torch tensors, in-place ops) get a writable, owned array
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()
