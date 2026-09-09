"""Whole-transcript word alignment for the audio-vs-script pipeline.

The per-cue windowed ASR check (asr.py) works for burned-in subtitles, where each
detected event is a self-contained line. Authored broadcast scripts are different:
one sentence is split across several 1-2 second cues for reading pace, and the
audio often lags the cue timing by a beat at an intro. Transcribing each tiny cue
window then catches the *neighbouring* fragment and reports a mismatch on text
that is actually correct.

So this matcher does not chop the audio at cue edges. It transcribes the whole
audio once, aligns the full script word-stream against the full heard word-stream
with an order-preserving diff, then re-attributes the heard words back to the cue
each script word belongs to. A word that is genuinely wrong, missing, or extra
still surfaces as a diff (order is preserved); a correct word that merely straddled
a cue boundary or was spoken a beat late now lands on its own cue. Spelling drift
between ASR and the script (समधन vs समदन) is absorbed by comparing each cue's text
to its aligned heard text with the same fuzzy ratio the rest of the pipeline uses,
not by demanding exact tokens.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np
from rapidfuzz.fuzz import token_set_ratio

from subtitle_checker.artifacts import CheckResult, SubtitleEvent, Verdict
from subtitle_checker.match.asr import SAMPLE_RATE, AsrEngine

# A cue whose aligned heard text scores below this is a gross mismatch. After the
# global align removes boundary bleed, correct cues cluster high (85-100) and a
# genuinely wrong word drops well below, so the cut separates them cleanly.
MATCH_RATIO = 65.0
# One- or two-word cues (है?, चलो।) are function words the ASR folds into a
# neighbour; too little to judge alone, so they are shown but never flagged.
MIN_WORDS = 2
# Transcribe the audio in chunks this long. Long enough to give the ASR sentence
# context, short enough to stay inside Sarvam's sync short-audio endpoint.
CHUNK_S = 20.0

# Danda and common punctuation are dropped before matching; matras and nukta are
# kept - a wrong matra is exactly the kind of error the tool must catch.
_PUNCT = re.compile(r"[।॥,.!?\"'()\[\]{}:;–—-]")


def _norm(word: str) -> str:
    """Normalise a token for matching: strip punctuation, lowercase Latin."""
    return _PUNCT.sub("", word).strip().lower()


def transcribe_full(
    audio: np.ndarray,
    engine: AsrEngine,
    sample_rate: int = SAMPLE_RATE,
    chunk_s: float = CHUNK_S,
) -> list[str]:
    """Transcribe the whole audio in fixed chunks -> one ordered word stream.

    Chunking keeps each request inside the short-audio endpoint; the words are
    concatenated in order, which is all the positional alignment needs (it never
    relies on per-word timestamps).
    """
    words: list[str] = []
    step = max(1, int(chunk_s * sample_rate))
    for start in range(0, len(audio), step):
        window = audio[start : start + step]
        if not window.size:
            continue
        heard = engine.transcribe(window)
        if heard:
            words.extend(heard.split())
    return words


def _heard_per_word(
    script_norm: list[str], heard_words: list[str]
) -> list[list[str]]:
    """For each script word, the heard words the global align maps onto it.

    Equal blocks map one-to-one. A replace block spreads its heard words across
    its script words by position. Deleted script words get nothing (no audio
    matched them). Inserted heard words - dialogue with no script - attach to the
    preceding script word so they are not lost from the report.
    """
    heard_norm = [_norm(w) for w in heard_words]
    matcher = SequenceMatcher(a=script_norm, b=heard_norm, autojunk=False)
    mapped: list[list[str]] = [[] for _ in script_norm]
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapped[i1 + k].append(heard_words[j1 + k])
        elif tag == "replace":
            span = i2 - i1
            chunk = heard_words[j1:j2]
            for idx, k in enumerate(range(i1, i2)):
                lo = (idx * len(chunk)) // span
                hi = ((idx + 1) * len(chunk)) // span
                mapped[k].extend(chunk[lo:hi])
        elif tag == "insert" and i1 > 0:
            mapped[i1 - 1].extend(heard_words[j1:j2])
    return mapped


def align_script(
    events: list[SubtitleEvent],
    heard_words: list[str],
    min_ratio: float = MATCH_RATIO,
    min_words: int = MIN_WORDS,
) -> list[CheckResult]:
    """One CheckResult per cue, its heard text pulled from the global alignment.

    The score is the audio match alone (the script is authored, fully trusted).
    A cue scoring below ``min_ratio`` is a TEXT_MISMATCH; a cue too short to judge
    stays OK, still shown with its heard words for an editor to eyeball.
    """
    script_words: list[str] = []
    owner: list[int] = []
    for ci, event in enumerate(events):
        for word in event.text.split():
            script_words.append(word)
            owner.append(ci)

    mapped = _heard_per_word([_norm(w) for w in script_words], heard_words)

    results: list[CheckResult] = []
    for ci, event in enumerate(events):
        heard = [w for k, o in enumerate(owner) if o == ci for w in mapped[k]]
        heard_text = " ".join(heard)
        n_words = sum(1 for o in owner if o == ci)
        ratio = token_set_ratio(event.text, heard_text) if heard_text else 0.0

        if n_words < min_words:
            # A one- or two-word cue is folded into a neighbour by the ASR - too
            # little to score, so it is shown but carries no match number.
            results.append(
                CheckResult(
                    start=event.start,
                    end=event.end,
                    verdict=Verdict.OK,
                    reason="line too short to check word-for-word",
                    subtitle_text=event.text,
                    heard_text=heard_text,
                )
            )
            continue

        if ratio >= min_ratio:
            verdict = Verdict.OK
            reason = f"heard words match the SRT (match {ratio:.0f}%)"
        else:
            verdict = Verdict.TEXT_MISMATCH
            reason = f"heard words differ from the SRT (match {ratio:.0f}%)"

        results.append(
            CheckResult(
                start=event.start,
                end=event.end,
                verdict=verdict,
                reason=reason,
                subtitle_text=event.text,
                heard_text=heard_text,
                score=ratio / 100.0,
                ocr_confidence=None,
                combined_score=round(ratio, 1),
            )
        )
    return results
