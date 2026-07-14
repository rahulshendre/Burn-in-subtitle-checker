"""Tests for the minimal report web UI (pure parts, no socket)."""

from __future__ import annotations

from subtitle_checker.report.webui import (
    ReportEntry,
    discover_reports,
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
