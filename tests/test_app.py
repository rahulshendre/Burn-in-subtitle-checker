"""Tests for the local app (pure parts and the job runner, no socket)."""

from __future__ import annotations

import os
import sys

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
    if sys.platform != "win32":  # Windows has no owner-only mode bits
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
    assert 'id="keycard">' in no_key  # shown, not hidden
    assert 'id="changekey"' not in no_key
    assert 'id="run" disabled' in no_key
    assert "No checks yet" in no_key

    with_key = render_app([], has_key=True)
    assert 'id="keycard" hidden' in with_key
    assert 'id="run">' in with_key
    assert "Marathi" in with_key


def test_a_saved_key_can_always_be_changed_from_the_page():
    page = render_app([], has_key=True)
    assert ">Change API key<" in page
    assert 'id="keymsg"' in page  # where a rejected key is explained


def _serve(tmp_path):
    import threading

    server = app._Server(("127.0.0.1", 0), app._make_handler(tmp_path, Job(), threading.Lock()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _post_key(server, key):
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        f"http://127.0.0.1:{server.server_address[1]}/key", data=key.encode(), method="POST"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode()


def test_a_key_sarvam_rejects_is_not_saved(tmp_path, monkeypatch):
    monkeypatch.delenv(app.KEY_ENV, raising=False)
    monkeypatch.setattr(app, "check_key", lambda key: "Sarvam did not accept this key.")
    server = _serve(tmp_path)
    try:
        status, body = _post_key(server, "typo")
    finally:
        server.shutdown()
    assert status == 400
    assert body == "Sarvam did not accept this key."
    assert not (tmp_path / "config.json").exists()
    assert app.KEY_ENV not in os.environ


def test_a_key_sarvam_accepts_is_saved(tmp_path, monkeypatch):
    monkeypatch.delenv(app.KEY_ENV, raising=False)
    monkeypatch.setattr(app, "check_key", lambda key: None)
    server = _serve(tmp_path)
    try:
        status, _ = _post_key(server, "  good-key  ")
    finally:
        server.shutdown()
    assert status == 200
    assert os.environ[app.KEY_ENV] == "good-key"
    assert (tmp_path / "config.json").exists()
    monkeypatch.delenv(app.KEY_ENV)


def test_an_empty_key_is_refused_without_calling_sarvam(tmp_path, monkeypatch):
    def never(key):
        raise AssertionError("must not call Sarvam for an empty key")

    monkeypatch.setattr(app, "check_key", never)
    server = _serve(tmp_path)
    try:
        status, body = _post_key(server, "   ")
    finally:
        server.shutdown()
    assert status == 400
    assert "Paste the key" in body


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


def test_run_check_tells_the_user_when_sarvam_rejects_the_key(tmp_path, monkeypatch):
    from subtitle_checker import cli
    from subtitle_checker.report.keycheck import RUN_REJECTED

    video = tmp_path / "Clip.mp4"
    video.write_bytes(b"v")

    class Rejected(Exception):
        status_code = 403  # what the Sarvam SDK raises for a bad key

    def boom(args):
        raise Rejected("headers: {}, status_code: 403, body: invalid_api_key_error")

    monkeypatch.setattr(cli, "_run_check", boom)
    job = Job(state="running")
    app.run_check(job, tmp_path, video, None, "hi")

    assert job.state == "error"
    assert job.error == RUN_REJECTED


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


def test_finished_check_shows_an_open_report_button_not_a_popup():
    # A tab opened minutes after the click is blocked by the browser's pop-up
    # blocker, and a reload would wipe the message, so the page offers a link.
    page = render_app([], has_key=True)
    assert "window.open" not in page
    assert "},1500)" not in page  # the old reload-after-1.5s that cleared the message
    assert 'class="openreport"' in page
    assert ">Open report<" in page


def test_render_app_has_a_quit_button():
    page = render_app([], has_key=True)
    assert 'id="quit"' in page
    assert "/quit" in page


def test_server_refuses_port_sharing_on_windows():
    import sys

    assert app._Server.allow_reuse_address == (sys.platform != "win32")
