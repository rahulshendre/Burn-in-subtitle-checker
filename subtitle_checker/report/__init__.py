"""Stage 4 - editor-facing report.

Input: ``check_results`` artifact (plus frames/audio for evidence snippets).
Output: an HTML report for non-technical QA reviewers.

Each flagged row carries the evidence needed to judge it in seconds: a frame
thumbnail, a short audio snippet, both texts, and the verdict reason. Sorted
by severity, grouped by verdict type.
"""
