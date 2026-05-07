# Demo Submission - Rahul Shendre

I've been working with PlanetRead since June 2025, first through C4GT DMP 2025, then continuing as a software development intern. Most of my work has been around subtitle tooling for BIRD's Same Language Subtitling pipeline. The most relevant pieces:

- [Click & Align](https://github.com/PlanetRead/subtitle-tool-for-Adobe-Premier/tree/click_and_align_subtitle_tool) and [Slide & Align](https://github.com/PlanetRead/subtitle-tool-for-Adobe-Premier/tree/slide_and_align_subtitle_tool) - subtitle automation plugins for Adobe Premiere Pro, live on Adobe Exchange, supporting all 22 Indian languages
- [BookBox](https://github.com/rahulshendre/BookBox) - the app that streams BookBox AniBooks (PlanetRead/BIRD content) - migrated to React Native, live on Play Store, App Store, and Amazon app store

The subtitle tools I built only handle alignment, where the editor places the subtitles. Whether the text itself is correct has always been a manual check.
This tool closes that gap.

---

## Modules covered

All three.

**Module 1 - Audio transcription**
Whisper `small` model, `temperature=0` for deterministic output. Results cached to `segments.json` so subsequent runs skip transcription entirely.

**Module 2 - Subtitle extraction**
OpenCV grabs a frame at each segment's midpoint. Bottom 20% is cropped - where burned-in subtitles sit in the test videos. Binary threshold applied before Tesseract OCR (`lang='hin'`) to clean up Devanagari on compressed frames.

**Module 3 - Mismatch detection + report**
RapidFuzz `token_set_ratio` on normalized text (punctuation stripped, whitespace collapsed). Segments below threshold flagged as REVIEW. HTML report is generated with timestamps, ASR text, OCR text, cropped frame thumbnails, and scores.

---

## Demo video

[![Demo](https://img.youtube.com/vi/G06-LdzV9PU/hqdefault.jpg)](https://youtu.be/G06-LdzV9PU)

[![YouTube](https://img.shields.io/badge/YouTube-Play-FF0000?style=flat-square&logo=youtube&logoColor=white)](https://youtu.be/G06-LdzV9PU)

---

## Code

**Repo:** https://github.com/rahulshendre/subtitle_mismatch_mvp

| Branch | Language | Notes |
|--------|----------|-------|
| `main` | Hindi | Primary - strongest results |
| `marathi` | Marathi | Devanagari script, same OCR pipeline |
| `kannada` | Kannada | Tesseract reads the script correctly, although Whisper ASR accuracy is limited - documented in [ANALYSIS.md](https://github.com/rahulshendre/subtitle_mismatch_mvp/blob/main/ANALYSIS.md).|

The pipeline is language-agnostic, changing language in Whisper and lang in Tesseract is all it takes. Hindi gives the strongest results because of Whisper's training data distribution across Indian languages.

---

## Test content

BookBox AniBook - *Rani Goes to School* (Hindi, Marathi, Kannada versions).

Hindi run results:
- Total segments: 44
- OK: 42
- REVIEW: 2 - both false positives, explained in [ANALYSIS.md](https://github.com/rahulshendre/subtitle_mismatch_mvp/blob/main/ANALYSIS.md)

---

## Known limitations

Full breakdown in [ANALYSIS.md](https://github.com/rahulshendre/subtitle_mismatch_mvp/blob/main/ANALYSIS.md).

1. **Intro frame false positives** - title cards with no subtitle return empty OCR, score 0.00, flagged as REVIEW. Pipeline can't distinguish a blank OCR result from a genuinely subtitle-free frame.
2. **Single-frame midpoint sampling** - one frame per segment misses subtitle transitions mid-segment. Multi-frame sampling with best-OCR selection is the fix.
3. **Silence gaps** - Whisper hallucinates on silent segments. Confidence score filtering is the planned approach, the same problem came up in the DMP 2025 Adobe plugin work as well and we couldn't fully solve it there due to ExtendScript limitations.