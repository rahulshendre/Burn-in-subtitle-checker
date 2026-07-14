# Burn-in Subtitle Checker

Flags moments where the spoken audio and the burned-in subtitles of a video do
not match, so a reviewer can jump straight to the flagged spots instead of
watching the whole file.

Built for PlanetRead's Same Language Subtitling (SLS) content under C4GT DMP
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

Each stage reads and writes a JSON artifact (`subtitle_checker/artifacts.py`),
so a stage can be cached, re-run, or swapped without touching the others.
`subtitle_checker/evaluation/` holds the harness that scores the pipeline
against planted errors.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

Hindi first; other languages follow once Hindi works end to end.

Status: under active development. Pipeline stages are landing as separate,
reviewable PRs.
