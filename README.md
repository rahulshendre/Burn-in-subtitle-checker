# Burn-in Subtitle Checker

Flags moments where the spoken audio and the burned-in subtitles of a video do
not match, so a reviewer can jump straight to the flagged spots instead of
watching the whole file.

Built for PlanetRead and for content with Same Language Subtitling (SLS) content under C4GT DMP
2026, issue [#3](https://github.com/PlanetRead/Burn-in-subtitle-checker/issues/3).

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

Each stage reads and writes a JSON artifact, so a stage can be cached, re-run,
or swapped without touching the others. A separate evaluation harness scores the
pipeline against planted errors.

## Where the code is

The work is built stage by stage, each on its own reviewable branch. The
branches stack, so the tip branch `feat/align-rescue` contains the full current
pipeline.

| Branch | What it adds |
|--------|--------------|
| `feat/package-skeleton` | package layout, JSON artifacts, CLI, CI |
| `feat/eval-harness` | error injector and scoring harness (the ground truth) |
| `feat/subtitle-events` | Stage 1: subtitle detection and OCR |
| `feat/audio-regions` | Stage 2: speech / music / silence and structural checks |
| `feat/forced-alignment` | Stage 3: forced alignment and Sarvam ASR cross-check |
| `feat/report` | Stage 4: self-contained HTML editor report |
| `feat/ocr-junk-filter` | drop OCR junk boxes from logo and chrome leakage |
| `feat/region-chrome` | drop animated channel logos via region presence |
| `feat/coverage-metric` | measure how many lines Stage 3 actually verifies |
| `feat/align-rescue` | verify low-confidence lines that align strongly (current tip) |

## Development

```bash
git checkout feat/align-rescue
pip install -e ".[dev]"
ruff check .
pytest
```

Hindi first; other languages follow once Hindi works end to end.
