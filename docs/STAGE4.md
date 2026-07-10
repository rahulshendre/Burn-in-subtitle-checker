# Stage 4 — Editor Report

**What it does:** turns the check results from Stages 1–3 into a single
self-contained HTML file an editor opens in a browser — no server, no install,
no external assets. Every flag becomes a card with the evidence needed to judge
it in seconds; every checked line becomes a heard-vs-written row to scan.

This is not cosmetics bolted onto the pipeline. It *is* the product. The one
thing the tool does reliably — put the written subtitle, the audio the editor
can play, and what an Indic ASR heard side by side — is a human-judgment aid,
and Stage 4 is its delivery. It also answers the mentor's explicit ask: a
minimal, non-Streamlit UI that "just shows the tool", deployable as an on-device
file.

## Why a static HTML file

The report embeds its own evidence — frames as base64 PNG, audio as base64 MP3 —
so the whole thing is one portable file. It can be dropped in a shared Drive,
emailed, or opened offline on any machine with a browser. That matches the
deployment constraint (on-device, no Streamlit server) and sidesteps the whole
class of "the images didn't load" problems a linked-asset report has.

```
check_results + video  →  report.html  (frames + audio embedded, opens anywhere)
```

## Anatomy of the report

Two sections, in the order an editor works:

1. **Flags** — one card per problem, worst-first. Each card carries:
   - the **subtitle frame** (a thumbnail cut at the event midpoint),
   - the **written** subtitle beside what the ASR **heard**,
   - a **3-second audio snippet** to play and verify,
   - the **verdict** badge, the **reason**, and the score where one applies.

2. **Matching lines — heard vs written** — a scan table of every line that
   *passed* the automatic check. Each row shows the frame thumbnail, the written
   text, the heard transcript, and an audio button. This is where the editor
   catches the subtle single-word errors the pipeline deliberately does **not**
   auto-flag (they sit below the OCR↔ASR noise floor — see [STAGE3.md](STAGE3.md)).

Cards are grouped and ordered by severity:

```
TEXT_MISMATCH → MISSING_SUBTITLE → ORPHAN_SUBTITLE → TIMING_DRIFT → UNCHECKABLE
```

`OK` never becomes a card — it flows into the heard-vs-written table instead.

## Where the heard-vs-written rows come from

The flags Stages 2–3 raise are, by design, only the confident problems. To fill
the scan table the ASR path uses `transcribe_lines` (in `match/asr.py`), which
returns one `CheckResult` per trusted, speech-covered line — `OK` when the heard
words match, `TEXT_MISMATCH` when they diverge grossly — always carrying
`heard_text`. `check_asr` is just the gross-mismatch subset of it, so the flag
logic is unchanged; the OK rows are the new material the report needs.

`check` merges the two: an ASR row that carries heard-vs-written **supersedes** a
bare alignment flag on the same line (`_merge_results`), so the card shows what
was heard rather than an evidence-free "does not match". This is a first step
toward the signal fusion [STAGE3.md](STAGE3.md) notes as future work.

## The Evidence protocol — why the renderer is pure

The renderer (`report/html.py`) never calls ffmpeg. It asks an `Evidence`
provider for each frame and audio clip:

```python
class Evidence(Protocol):
    def frame_png(self, t: float) -> bytes | None: ...
    def audio_clip(self, start: float, end: float) -> tuple[bytes, str] | None: ...
```

The ffmpeg-backed implementation (`report/evidence.py`) cuts a scaled PNG frame
and a small MP3 span straight to stdout — no temp files. A failed or empty
ffmpeg run returns `None` and the renderer draws a placeholder, so a report
still generates even if one snippet cannot be cut.

Keeping media behind the protocol means the renderer is unit-tested with a fake
that returns opaque bytes — no ffmpeg in the HTML tests — exactly the pattern
Silero (VAD), EasyOCR, and Sarvam already follow.

## Module map

| File | Job |
|------|-----|
| `report/html.py` | pure renderer: `render_report(results, evidence, *, title)` → HTML string; `Evidence` Protocol |
| `report/evidence.py` | `FfmpegEvidence` (frame + MP3 to stdout) and `write_report` (render + write file) |
| `match/asr.py` | `transcribe_lines` — the per-line OK/mismatch ledger with `heard_text` |
| `cli.py` | `report` subcommand, auto-generation in `check`, `_merge_results` |

No new dependency: frames and audio come from ffmpeg (already required),
embedding uses stdlib `base64`.

## How to run it

The report is generated automatically at the end of a check:

```
subtitle-checker check --video path/to/video.mp4 --lang hi --asr
# → out/<stem>_report.html
```

Or render one from a previous run's saved artifacts, without re-running the
pipeline:

```
subtitle-checker report --results out/<stem>/ --video path/to/video.mp4
```

`--results` accepts either the `check_results.json` file or the directory that
holds it; `--out` overrides the default output path.

## What the demo shows

**Real clip (Mann Atisunder, heavy background score)** — the pipeline runs
end-to-end and raises **0 flags**; the report is a 5-line heard-vs-written
ledger. Sarvam recovers the dialogue the score buries (proper nouns garble —
`मित्तल` heard as `पृथ्वी` — but content words come through, and no correct
line is falsely flagged). This is the honest editor surface on real footage.

**Injected-error clip** — five known lines are burned onto Mann's real audio
with one line replaced by an unrelated sentence. The Stage 3 ASR cross-check
catches it: written `फिर हम बाज़ार से सब्ज़ी ले आएंगे।` against heard
`हमारे परिवार की बेइज्जती होगी।` (the true audio), a `TEXT_MISMATCH` card, while
the four correct lines pass into the ledger. Because Abinash dropped the
six-parameter/error videos, injection is the only way to demonstrate a catch —
the report machinery is identical to what a real error would trigger.

## Known limitations

- **The ledger covers trusted speech-covered lines only.** Lines over music or
  silence, or with low-confidence OCR, are not transcribed (the same gates as
  the ASR check) and so do not appear in the heard-vs-written table.
- **Long clips make large files.** Every card and ledger row embeds a frame and
  an audio clip. For a short clip this is a portable ~150 KB–750 KB file; a full
  episode with hundreds of lines would be many megabytes. The report is built
  for review of flagged/checked lines, not as a bulk data dump.
- **The video must be present** when the report is generated — the frames and
  audio are cut from it live. The HTML is self-contained *after* generation.

## Extending

- **Different look:** the renderer emits one HTML string from `CheckResult`s;
  restyle by editing the inline CSS, or swap `render_report` for another
  formatter (PDF, a static-site page) reusing the same `Evidence` provider.
- **Richer evidence:** implement `Evidence` differently — e.g. a waveform image
  per clip, or a longer audio window — without touching the renderer.
- **Per-event fusion:** `_merge_results` already lets ASR evidence win a span;
  combining alignment score *and* ASR ratio into a single confidence is the
  next step, and the `CheckResult` contract already carries both.
