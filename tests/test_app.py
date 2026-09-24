"""Tests for the local app (pure parts and the job runner, no socket)."""

from __future__ import annotations

import os

from subtitle_checker.report import app
from subtitle_checker.report.app import (
    Job,
    load_key,
    past_runs,
    render_app,
    resolve_inside,
    safe_name,
    save_key,
)


def test_safe_name_strips_paths_and_odd_characters():
    assert safe_name("../../etc/passwd") == "passwd"
    assert safe_name("C:\\\\clips\\\\Mann ep 3.mp4") == "Mann ep 3.mp4"
    assert safe_name("a<b>|c.mp4") == "a_b__c.mp4"
    assert safe_name("") == "upload"


def test_resolve_inside_refuses_paths_outside_home(tmp_path):
    home = tmp_path / "home"
    (home / "runs").mkdir(parents=True)
    report = home / "runs" / "x_report.html"
    report.write_text("r", encoding="utf-8")
    (tmp_path / "secret.html").write_text("s", encoding="utf-8")

    assert resolve_inside(home, "runs/x_report.html") == report.resolve()
    assert resolve_inside(home, "../secret.html") is None
    assert resolve_inside(home, "runs/missing.html") is None


def test_key_saved_locally_and_loaded(tmp_path, monkeypatch):
    monkeypatch.delenv(app.KEY_ENV, raising=False)
    save_key(tmp_path, "sk-test")
    assert os.environ[app.KEY_ENV] == "sk-test"
    assert oct((tmp_path / "config.json").stat().st_mode)[-3:] == "600"

    monkeypatch.delenv(app.KEY_ENV)
    load_key(tmp_path)
    assert os.environ[app.KEY_ENV] == "sk-test"


def test_load_key_keeps_an_existing_env_key(tmp_path, monkeypatch):
    save_key(tmp_path, "saved")
    monkeypatch.setenv(app.KEY_ENV, "from-env")
    load_key(tmp_path)
    assert os.environ[app.KEY_ENV] == "from-env"


def test_past_runs_lists_reports_with_mistakes_pages(tmp_path):
    run = tmp_path / "runs" / "Clip_20260924-120000"
    run.mkdir(parents=True)
    (run / "Clip_report.html").write_text("r", encoding="utf-8")
    (run / "Clip_mistakes.html").write_text("m", encoding="utf-8")

    runs = past_runs(tmp_path)

    assert len(runs) == 1
    assert runs[0]["label"] == "Clip"
    assert runs[0]["report"] == "runs/Clip_20260924-120000/Clip_report.html"
    assert runs[0]["mistakes"].endswith("Clip_mistakes.html")


def test_render_app_asks_for_key_until_one_is_saved():
    no_key = render_app([], has_key=False)
    assert 'id="keycard"' in no_key
    assert 'id="run" disabled' in no_key
    assert "No checks yet" in no_key

    with_key = render_app([], has_key=True)
    assert 'id="keycard"' not in with_key
    assert 'id="run">' in with_key
    assert "Marathi" in with_key


def test_render_app_lists_past_runs():
    runs = [{"label": "Mann", "report": "runs/a/Mann_report.html",
             "mistakes": "runs/a/Mann_mistakes.html", "when": 0.0}]
    page = render_app(runs, has_key=True)
    assert "/file/runs/a/Mann_report.html" in page
    assert ">mistakes<" in page


def test_run_check_records_report_and_cleans_uploads(tmp_path, monkeypatch):
    from subtitle_checker import cli

    video = tmp_path / "uploads" / "Clip.mp4"
    video.parent.mkdir()
    video.write_bytes(b"v")

    def fake_check(args):
        print("12 subtitle events (12 with text)")
        out = tmp_path / args.out
        out.mkdir(parents=True, exist_ok=True)
        (out / "Clip_report.html").write_text("r", encoding="utf-8")
        print("0 flag(s):")
        return 0

    monkeypatch.setattr(cli, "_run_check", fake_check)
    job = Job(state="running")
    app.run_check(job, tmp_path, video, None, "mr")

    assert job.state == "done"
    assert job.report.startswith("runs/Clip_") and job.report.endswith("Clip_report.html")
    assert any("subtitle events" in line for line in job.log)
    assert not video.exists()


def test_run_check_reports_a_pipeline_failure(tmp_path, monkeypatch):
    from subtitle_checker import cli

    video = tmp_path / "Clip.mp4"
    video.write_bytes(b"v")

    def boom(args):
        raise RuntimeError("Sarvam said 401")

    monkeypatch.setattr(cli, "_run_check", boom)
    job = Job(state="running")
    app.run_check(job, tmp_path, video, None, "hi")

    assert job.state == "error"
    assert "401" in job.error


def test_progress_fills_within_a_step_but_never_past_it():
    from subtitle_checker.report.app import progress

    start, _ = progress(0, 0.0, 40.0)
    mid, left_mid = progress(0, 25.0, 40.0)
    stuck, _ = progress(0, 500.0, 40.0)

    assert start == 10
    assert 10 < mid < 65
    assert stuck < 65  # waits for the pipeline before entering the next step
    assert left_mid > 0


def test_progress_later_steps_sit_further_along():
    from subtitle_checker.report.app import progress

    audio, _ = progress(1, 0.0, 40.0)
    report, left = progress(2, 1.0, 40.0)
    assert audio == 65
    assert 85 <= report < 100
    assert left >= 5


def test_job_json_reports_percent_and_advances_stages(tmp_path):
    job = Job(state="running", started=1.0, stage_started=1.0, duration=40.0)
    job.advance(1)
    job.advance(0)  # never moves backwards
    data = job.as_json()
    assert data["stage"] == 1
    assert data["stages"][1] == "Checking the audio"
    assert 65 <= data["percent"] < 85

    job.state = "done"
    assert job.as_json()["percent"] == 100


def test_render_app_has_a_quit_button():
    page = render_app([], has_key=True)
    assert 'id="quit"' in page
    assert "/quit" in page
