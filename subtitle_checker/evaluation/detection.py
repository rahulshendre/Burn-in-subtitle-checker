"""Closed-loop evaluation of Stage 1 detection on real footage.

Burn known truth lines onto a clean (subtitle-free) clip with the harness
burner, run the Stage 1 detector on the result, and measure how much of the
truth came back: recall, timing error, and OCR text similarity. Run it on a
segment of real broadcast footage — logos, tickers and all — and the numbers
say how the detector behaves in production conditions, not on synthetics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from subtitle_checker.artifacts import SubtitleEvent
from subtitle_checker.evaluation.burn import burn_subtitles
from subtitle_checker.subtitles.ocr import OcrEngine

# Hindi truth lines used when the caller supplies none.
DEFAULT_TRUTH_LINES = [
    "हम सब से नज़रे कैसे मिला पायेंगे?",
    "आप उस पर इल्ज़ाम लगा रही है।",
    "मैं खुद चलकर उनके पास जाऊंगा।",
    "ये उम्मीद नहीं थी आपसे।",
    "अब कोई कुछ भी कहे, फर्क नहीं पड़ता।",
    "एक मां को और क्या चाहिए।",
    "दो ही जनों को सबसे अधिक पीड़ा होती है।",
    "बस, एक उपकार करें।",
]

# Real serial subtitles render above the bottom ticker zone.
BURN_MARGIN_V = 110
# A detected event must cover at least this much of a truth line to match.
MIN_OVERLAP_S = 0.5


def make_truth(
    lines: list[str],
    start: float = 4.0,
    line_s: float = 3.0,
    gap_s: float = 2.5,
) -> list[SubtitleEvent]:
    """Lay the truth lines out on a timeline, one line then a gap."""
    events = []
    t = start
    for line in lines:
        events.append(SubtitleEvent(start=t, end=t + line_s, text=line))
        t += line_s + gap_s
    return events


@dataclass
class Match:
    truth: SubtitleEvent
    detected: SubtitleEvent
    similarity: float


@dataclass
class DetectionReport:
    matches: list[Match] = field(default_factory=list)
    missed: list[SubtitleEvent] = field(default_factory=list)
    strays: list[SubtitleEvent] = field(default_factory=list)

    @property
    def truth_count(self) -> int:
        return len(self.matches) + len(self.missed)

    @property
    def recall(self) -> float:
        return len(self.matches) / self.truth_count if self.truth_count else 0.0

    @property
    def mean_similarity(self) -> float:
        sims = [m.similarity for m in self.matches]
        return sum(sims) / len(sims) if sims else 0.0

    @property
    def mean_start_error(self) -> float:
        errs = [abs(m.detected.start - m.truth.start) for m in self.matches]
        return sum(errs) / len(errs) if errs else 0.0

    @property
    def mean_end_error(self) -> float:
        errs = [abs(m.detected.end - m.truth.end) for m in self.matches]
        return sum(errs) / len(errs) if errs else 0.0


def _overlap(a: SubtitleEvent, b: SubtitleEvent) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def match_detection(
    truth: list[SubtitleEvent],
    detected: list[SubtitleEvent],
    min_overlap_s: float = MIN_OVERLAP_S,
) -> DetectionReport:
    """Pair each truth line with its best-overlapping detection."""
    report = DetectionReport()
    used: set[int] = set()
    for tr in truth:
        best, best_ov = None, min_overlap_s
        for i, d in enumerate(detected):
            if i in used:
                continue
            ov = _overlap(tr, d)
            if ov > best_ov:
                best, best_ov = i, ov
        if best is None:
            report.missed.append(tr)
            continue
        used.add(best)
        d = detected[best]
        sim = SequenceMatcher(None, tr.text, d.text).ratio()
        report.matches.append(Match(truth=tr, detected=d, similarity=sim))
    report.strays = [d for i, d in enumerate(detected) if i not in used]
    return report


def evaluate_detection(
    clean_clip: Path,
    out_dir: Path,
    lines: list[str] | None = None,
    engine: OcrEngine | None = None,
) -> DetectionReport:
    """Burn truth lines onto ``clean_clip``, detect them back, and compare."""
    from subtitle_checker.subtitles.reconstruct import reconstruct_subtitles

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = make_truth(lines or DEFAULT_TRUTH_LINES)

    burned = out_dir / f"{Path(clean_clip).stem}_burned.mp4"
    if not burned.exists():
        burn_subtitles(clean_clip, truth, burned, margin_v=BURN_MARGIN_V)

    detected = reconstruct_subtitles(burned, engine=engine)
    return match_detection(truth, detected)
