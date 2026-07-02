# Burn-in Subtitle Checker

A lightweight tool that flags mismatches between the audio dialogue and the
burned-in subtitles of a video file, so QA reviewers only need to check the
flagged moments instead of scrubbing the whole video.

Built for PlanetRead's Same Language Subtitling (SLS) content under
C4GT DMP 2026 — see issue
[#3](https://github.com/PlanetRead/Burn-in-subtitle-checker/issues/3).

## How it works

The subtitle track is reconstructed first (it is the reliable signal — burned-in
text is clean and consistent), then the audio is checked *against* it:

| Stage | Module | Output |
|---|---|---|
| 1. Subtitle track reconstruction | `subtitle_checker/subtitles/` | subtitle events with exact on/off times |
| 2. Audio labelling | `subtitle_checker/audio/` | speech / music / song regions |
| 3. Matching | `subtitle_checker/match/` | verdict per span, with reason |
| 4. Report | `subtitle_checker/report/` | HTML report for editors |

Every stage reads and writes a JSON artifact (see
`subtitle_checker/artifacts.py`), so stages can be cached, re-run, or swapped
independently. `subtitle_checker/evaluation/` holds the accuracy harness the
pipeline is scored against.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

Status: under active development. Pipeline stages are landing as separate,
reviewable PRs.
