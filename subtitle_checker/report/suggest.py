"""Turn a flagged mismatch into a suggested correction - honestly.

Management asked the tool to do more than flag a wrong line: it should also say
what the line *should* say, so a production editor can act without re-scrubbing
the audio themselves. The correction we can offer is what the ASR heard under
that subtitle - the transcript already captured on every flagged line
(match.asr / match.script_align store it as ``heard_text``).

The catch is that ASR is the weak side of this whole tool (see PLAN_V2): OCR of
burned-in text is reliable, but the transcript degrades on songs and heavy
background score and can garble Devanagari. Printing a garbled transcript as
"the correct text" would be worse than saying nothing - an editor might trust it
and make the subtitle worse. This is the same lesson as the legibility over-claim
(a bug is worse than an abstention).

So a suggestion is offered only when the heard text is worth standing behind, and
otherwise the tool says plainly that it could not determine the correct text. The
gate here is deliberately conservative and mirrors the trust gates the flag layers
already apply upstream; this module never manufactures a suggestion the pipeline
did not already have evidence for.
"""

from __future__ import annotations

from dataclasses import dataclass

from subtitle_checker.artifacts import CheckResult, Verdict

# A transcript needs at least this many words before it reads as a real line
# rather than a stray syllable the ASR latched onto. One or two heard words under
# a mismatch is as likely noise as a correction, so it is not offered as one.
MIN_HEARD_WORDS = 2


@dataclass(frozen=True)
class Suggestion:
    """What the tool proposes for one flagged line.

    ``text`` is the suggested wording when we can stand behind it, else "".
    ``confident`` says whether a usable suggestion was found - when False, ``note``
    explains why the tool declined to guess. ``heading`` is the label the card
    shows above the text, so a corrected line and a filled-in missing line read
    differently ("Suggested correction" vs "Audio says (best guess)").
    """

    text: str
    confident: bool
    note: str
    heading: str = "Suggested correction"


# Shown when the tool will not guess a correction. Honest by design: an editor
# reads it as "check this yourself", not as an empty or broken field.
_NO_SUGGESTION = "Could not determine the correct text - check the audio for this line."

# The correction for a wrong line is the heard text, offered as the fix. A missing
# line has no wrong text to correct - the heard text is what *should be captioned*,
# offered as an unverified best guess (ASR is the weak side, see the module note).
_CORRECTION_NOTE = "Suggested from what the audio says here."
_MISSING_NOTE = "Unverified transcription of the audio - confirm before use."
_MISSING_HEADING = "Audio says (best guess)"


def suggest_correction(r: CheckResult) -> Suggestion:
    """Propose the wording for one flagged line, or decline to.

    Two flags carry a suggestion. A genuine TEXT_MISMATCH whose transcript has
    enough heard words gets the heard text as a correction. A MISSING_SUBTITLE
    that ASR has since transcribed (match.asr.transcribe_missing) gets the heard
    text as a best-guess of what the caption should say, labelled as unverified.
    Every other case - an orphan or uncheckable span, or a flag whose transcript
    is empty or too short - gets an honest "could not determine".
    """
    heard = r.heard_text.strip()
    if len(heard.split()) < MIN_HEARD_WORDS:
        # No transcript, or too little of one to read as the intended line.
        return Suggestion("", False, _NO_SUGGESTION)
    if r.verdict is Verdict.TEXT_MISMATCH:
        return Suggestion(heard, True, _CORRECTION_NOTE)
    if r.verdict is Verdict.MISSING_SUBTITLE:
        return Suggestion(heard, True, _MISSING_NOTE, heading=_MISSING_HEADING)
    # Orphan / uncheckable: nothing was mis-transcribed to correct.
    return Suggestion("", False, _NO_SUGGESTION)
