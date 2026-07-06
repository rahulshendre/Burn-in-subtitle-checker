"""Round-trip a synthetic tone through ffmpeg extraction."""

import subprocess
from pathlib import Path

import numpy as np

from subtitle_checker.ingest.audio_track import SAMPLE_RATE, extract_audio


def _make_tone_clip(path: Path, duration: float = 2.0) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-ar",
            "48000",
            str(path),
        ],
        check=True,
    )


def test_extract_audio_resamples_to_mono_16k(tmp_path: Path) -> None:
    clip = tmp_path / "tone.wav"
    _make_tone_clip(clip, duration=2.0)
    samples = extract_audio(clip)
    assert samples.dtype == np.float32
    # 2 s at 16 kHz, allow codec padding at the edges
    assert abs(len(samples) - 2 * SAMPLE_RATE) < SAMPLE_RATE * 0.1
    # a full-scale sine should keep real amplitude after resampling
    assert np.abs(samples).max() > 0.1


def test_extract_audio_missing_stream_raises(tmp_path: Path) -> None:
    clip = tmp_path / "silent.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=1",
            str(clip),
        ],
        check=True,
    )
    try:
        extract_audio(clip)
    except subprocess.CalledProcessError:
        return
    raise AssertionError("expected CalledProcessError for a video with no audio stream")
