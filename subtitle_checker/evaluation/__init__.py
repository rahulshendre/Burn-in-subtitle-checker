"""Evaluation harness - the pipeline is scored, not vibed.

Two parts:

1. A synthetic mismatch injector: takes a clean clip, re-burns its subtitles
   with controlled defects (swapped words, shifted timing, dropped lines) to
   produce labelled test videos with known answers.
2. A scorer: runs the pipeline over the labelled set and reports precision
   and recall per signal, so every component choice and threshold is a
   number, not a demo.
"""
