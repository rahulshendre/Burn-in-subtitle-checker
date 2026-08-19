"""Tests for the minimal report web UI (pure parts, no socket)."""

from __future__ import annotations

from subtitle_checker.report.webui import (
    ReportEntry,
    attach_transcripts,
    discover_reports,
    relabel,
    render_demo_index,
    render_index,
)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<html>report</html>", encoding="utf-8")


def test_discover_finds_reports_and_ignores_others(tmp_path):
    _touch(tmp_path / "mann" / "Mann_Atisunder_report.html")
    _touch(tmp_path / "gatha" / "Gatha_report.html")
    _touch(tmp_path / "mann" / "notes.txt")
    _touch(tmp_path / "mann" / "Mann_Atisunder_subtitle_events.json")

    entries = discover_reports([tmp_path])

    assert [e.label for e in entries] == ["Gatha", "Mann Atisunder"]
    assert [e.id for e in entries] == ["0", "1"]
    assert all(e.path.name.endswith("_report.html") for e in entries)


def test_discover_accepts_a_direct_file_and_dedupes(tmp_path):
    report = tmp_path / "d" / "Clip_report.html"
    _touch(report)

    entries = discover_reports([report, tmp_path])

    assert len(entries) == 1
    assert entries[0].label == "Clip"


def test_render_index_lists_entries_with_a_viewer():
    entries = [
        ReportEntry(id="0", label="Mann Atisunder", path=None),
        ReportEntry(id="1", label="Gatha", path=None),
    ]

    page = render_index(entries, title="Subtitle Checker")

    assert "Subtitle Checker" in page
    assert "Mann Atisunder" in page
    assert '<select id="pick">' in page
    assert '<iframe id="view"' in page
    assert "/report/" in page


def test_render_index_empty_state():
    page = render_index([])

    assert "No reports found" in page
    assert "<iframe" not in page


def test_attach_transcripts_sets_names_and_defaults():
    entries = [
        ReportEntry(id="0", label="Anupamaa wedding", path=None),
        ReportEntry(id="1", label="Mann", path=None),
    ]

    out = attach_transcripts(entries, ["without-ocr.srt"])

    assert out[0].transcript == "without-ocr.srt"
    assert out[1].transcript == "Mann.srt"  # unmatched defaults to <label>.srt


def test_render_demo_index_has_video_and_transcript_controls():
    entries = [
        ReportEntry(id="0", label="Anupamaa wedding", path=None, transcript="wedding.srt"),
    ]

    page = render_demo_index(entries, title="Subtitle Checker")

    assert '<select id="video">' in page  # video picker
    assert 'type="file"' in page and 'id="srt"' in page  # transcript upload
    assert "wedding.srt" in page  # the transcript name shows
    assert '<iframe id="view"' in page
    assert 'data-srt="wedding.srt"' in page


def test_render_demo_index_empty_state():
    page = render_demo_index([])

    assert "No reports found" in page
    assert "<iframe" not in page


def test_relabel_overrides_in_order_and_keeps_the_rest():
    entries = [
        ReportEntry(id="0", label="raw one", path=None),
        ReportEntry(id="1", label="raw two", path=None),
    ]

    out = relabel(entries, ["Mann - clean (0 flags)"])

    assert out[0].label == "Mann - clean (0 flags)"
    assert out[1].label == "raw two"
    assert out[0].id == "0" and out[0].path is None
