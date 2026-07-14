"""Stage 3 - matching subtitle events against the audio.

Input: ``subtitle_events`` + ``audio_regions`` artifacts (and the audio track).
Output: ``check_results`` artifact - one verdict per span, with a reason.

Three independent signals, cheapest first:

1. Structural checks - speech with no subtitle, subtitle with no speech,
   timing offset between the two. No ASR involved.
2. Forced alignment - score the subtitle text directly against its audio
   window (verification, not open transcription).
3. ASR cross-check - transcribe and fuzzy-compare where the ASR backend is
   confident.

Signals fuse into a single verdict per span; spans nothing can judge
honestly are marked UNCHECKABLE rather than guessed at.
"""
