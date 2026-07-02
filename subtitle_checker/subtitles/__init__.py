"""Stage 1 — subtitle track reconstruction.

Input: video file.
Output: ``subtitle_events`` artifact — one entry per burned-in subtitle line
with exact on/off timestamps.

Approach: sample the subtitle band at a few fps, binarize to a text mask, and
diff consecutive masks to find text-change events. Diffing masks (text shape)
rather than raw pixels keeps SLS karaoke word-highlighting from registering
as new events. Each unique event is OCR'd once. Persistent text that survives
scene cuts (logos, disclaimers, watermarks) is dropped before output.
"""
