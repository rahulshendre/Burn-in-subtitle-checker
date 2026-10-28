# Burn-in Subtitle Checker

Flags moments where the spoken audio and the burned-in subtitles of a video do
not match, so a reviewer can jump straight to the flagged spots instead of
watching the whole file.

Built for PlanetRead's Same Language Subtitling (SLS) content under C4GT DMP
2026, issue [#3](https://github.com/PlanetRead/Burn-in-subtitle-checker/issues/3).
Works on Hindi and Marathi videos today.

## The idea

Reading burned-in subtitle text with OCR is reliable; transcribing this noisy
Indian-language audio from scratch is not. So the tool reads the subtitles
first and treats that text as the known signal, then checks the audio against
it instead of transcribing blind.

| Stage | Package | Output |
|-------|---------|--------|
| 1. Read the subtitles | `subtitle_checker/subtitles/` | subtitle events with on/off times and text |
| 2. Label the audio | `subtitle_checker/audio/` | speech / music / silence regions |
| 3. Match audio to text | `subtitle_checker/match/` | a verdict and reason for each line |
| 4. Report | `subtitle_checker/report/` | a self-contained HTML report for editors |

Each stage reads and writes a JSON artifact (`subtitle_checker/artifacts.py`),
so a stage can be cached, re-run, or swapped without touching the others.
`subtitle_checker/evaluation/` holds the harness that scores the pipeline
against planted errors.

The reports are written for editors, not engineers: plain labels ("Subtitle on
screen", "What the audio says", "Speech with no subtitle"), one Confidence figure
per flag, and counts that add up. Alongside the mismatch check, the full report
grades how legible the subtitles are (by contrast against the picture), suggests
fixes for the weakest lines and checks the subtitles against PlanetRead's
subtitle guidelines. A second page lists only the mistakes, each with a
suggested correction.

## Ways to run it

- **Desktop app (Mac and Windows).** A double-click app that opens a page in
  the browser. Paste the Sarvam API key once (it is checked with Sarvam when you
  save it, and the "Change API key" button replaces it later), pick a video and
  the language, run the check, then click "Open report". Failures are explained
  in plain words, and the full error goes to `app.log` in the app's
  `Subtitle Checker` folder. The video stays on the computer; only cropped
  subtitle images and short audio clips go to Sarvam.
  - Mac (Apple Silicon): `packaging/macos/build.sh` builds `Subtitle Checker.app`.
  - Windows (64-bit): the `windows` GitHub Actions workflow builds
    `Subtitle-Checker-Setup.exe`, installs it on a Windows runner, checks that
    the installed app starts and serves the current page, and attaches the
    installer to the run. `packaging/windows/build.ps1` does the same build by hand.
  - The packaged app uses port 8000; set `SUBTITLE_CHECKER_PORT` to use another.
- **Command line.**

  ```bash
  export SARVAM_API_KEY=...
  subtitle-checker check --video clip.mp4 --lang hi --ocr sarvam-vision --asr
  subtitle-checker app   # the same local app, started from a terminal
  ```

## Development

```bash
pip install -e ".[dev,audio,asr,vision]"
ruff check .
pytest
```

Extras: `audio` (voice activity detection on onnxruntime), `asr` and `vision`
(Sarvam speech and OCR), `ocr` (on-device EasyOCR, pulls in torch), `align`
(forced alignment, pulls in torchaudio). The Sarvam key is read from the
`SARVAM_API_KEY` environment variable only.

## Status

Hindi and Marathi work end to end. 394 tests pass on macOS and on Windows (CI).

This branch is the tip of a stack of reviewable branches. The branch-by-branch
map is in the README on the fork's `main`.
