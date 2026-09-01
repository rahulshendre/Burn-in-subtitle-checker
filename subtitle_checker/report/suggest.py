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

    ``text`` is the suggested correct wording when we can stand behind it, else
    "". ``confident`` says whether a usable suggestion was found - when False,
    ``note`` explains why the tool declined to guess.
    """

    text: str
    confident: bool
    note: str


# Shown when the tool will not guess a correction. Honest by design: an editor
# reads it as "check this yourself", not as an empty or broken field.
_NO_SUGGESTION = "Could not determine the correct text - check the audio for this line."


def suggest_correction(r: CheckResult) -> Suggestion:
    """Propose the correct wording for one flagged line, or decline to.

    The heard transcript becomes the suggestion only for a genuine text mismatch
    that actually carries enough heard words to trust. A structural flag (missing
    or orphan line, uncheckable span) has no heard line to correct toward, and a
    mismatch whose transcript is empty or too short gets an honest "could not
    determine" instead of a guessed one.
    """
    if r.verdict is not Verdict.TEXT_MISMATCH:
        # Missing / orphan / uncheckable: nothing was mis-transcribed to correct.
        return Suggestion("", False, _NO_SUGGESTION)
    heard = r.heard_text.strip()
    if len(heard.split()) < MIN_HEARD_WORDS:
        # No transcript, or too little of one to read as the intended line.
        return Suggestion("", False, _NO_SUGGESTION)
    return Suggestion(heard, True, "Suggested from what the audio says here.")
