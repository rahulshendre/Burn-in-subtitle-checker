"""Stage 2 — audio labelling and routing.

Input: audio track from ingest.
Output: ``audio_regions`` artifact — speech / music / song / silence spans.

Voice activity detection plus a coarse music/speech classifier. Downstream
matching uses these labels to route: clean dialogue gets the full check,
song regions get alignment-only treatment with honest confidence limits.
Optional source separation (vocals stem) plugs in here.
"""
