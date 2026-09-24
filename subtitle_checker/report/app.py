"""Local app: pick a video, run the check, open the report - all in the browser.

The viewer in webui.py only shows reports that already exist. This is the tool
people in the org install and use themselves: the page uploads a video to a
server running on their own machine, the pipeline runs there, and the finished
report joins the list of past runs. Only the Sarvam calls leave the machine
(cropped subtitle bands for OCR, short audio windows for ASR); the video and
every report stay in the app's folder.

Stdlib only (http.server, threads), same as the viewer, so it packages into a
single on-device executable. One check runs at a time - a laptop cannot do
more, and the Sarvam rate limit would not allow it anyway. The page shell is a
pure function (render_app) so it is unit-tested without a socket.
"""

from __future__ import annotations

import argparse
import contextlib
import html
import io
import json
import os
import re
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote

from subtitle_checker.report.webui import _STYLE, discover_reports

LANGUAGES = {"hi": "Hindi", "mr": "Marathi"}
KEY_ENV = "SARVAM_API_KEY"
_CONFIG = "config.json"
_CHUNK = 1 << 20

# The pipeline steps shown on the page, with where each sits on the progress bar
# (start, end percent; the upload fills 0-10 in the browser) and how long it
# usually takes per second of video. The timings come from a real full-Sarvam
# run (40 s Marathi clip: subtitles 49 s, audio 10 s, report 10 s), so the bar
# is an estimate that moves at the pace the pipeline really does.
STAGES = (
    ("Reading the subtitles", 10, 65, 1.25, 0.0),
    ("Checking the audio", 65, 85, 0.2, 5.0),
    ("Building the report", 85, 100, 0.2, 5.0),
)
# The pipeline reports progress by printing; these markers advance the stage, so
# the CLI stays the single code path with no progress plumbing.
_STAGE_MARKERS = (
    ("subtitle events", 1),
    ("script cue(s)", 1),
    ("flag(s)", 2),
)
_CAP = 0.95  # a step never shows full until the pipeline says it has finished


def _expected(stage: int, duration: float) -> float:
    _, _, _, per_second, fixed = STAGES[stage]
    return max(per_second * duration + fixed, 5.0)


def progress(stage: int, in_stage: float, duration: float) -> tuple[int, int]:
    """Estimated (percent, seconds left) for a job in `stage` for `in_stage` s.

    Within a step the bar fills at the expected pace but stops short of the next
    step until the pipeline actually moves on, so it never claims more than done.
    """
    _, start, end, _, _ = STAGES[stage]
    expected = _expected(stage, duration)
    frac = min(in_stage / expected, _CAP)
    percent = int(start + (end - start) * frac)
    left = max(expected - in_stage, 5.0) + sum(
        _expected(i, duration) for i in range(stage + 1, len(STAGES))
    )
    return percent, int(left)


def video_duration(video: Path) -> float:
    """Length of the video in seconds, or 0 when it cannot be read."""
    from subtitle_checker.subtitles.sampler import probe

    try:
        return probe(video).duration
    except Exception:  # an unreadable file fails later with a clearer message
        return 0.0


def default_home() -> Path:
    """Folder that holds uploads, finished runs and the saved key."""
    return Path.home() / "Subtitle Checker"


def load_key(home: Path) -> None:
    """Put a saved Sarvam key into the environment unless one is already set."""
    if os.environ.get(KEY_ENV):
        return
    try:
        key = json.loads((home / _CONFIG).read_text(encoding="utf-8")).get("sarvam_key", "")
    except (OSError, ValueError):
        return
    if key:
        os.environ[KEY_ENV] = key


def save_key(home: Path, key: str) -> None:
    """Save the key on this machine only (owner-readable) and use it right away."""
    home.mkdir(parents=True, exist_ok=True)
    path = home / _CONFIG
    path.write_text(json.dumps({"sarvam_key": key}), encoding="utf-8")
    path.chmod(0o600)
    os.environ[KEY_ENV] = key


def safe_name(name: str) -> str:
    """Reduce an uploaded filename to a plain, safe file name."""
    base = Path(name.replace("\\", "/")).name
    base = re.sub(r"[^\w.\- ]", "_", base).strip(" .")
    return base or "upload"


def resolve_inside(home: Path, rel: str) -> Path | None:
    """Map a URL path back to a file inside home; refuse anything outside it."""
    path = (home / unquote(rel)).resolve()
    if path.is_relative_to(home.resolve()) and path.is_file():
        return path
    return None


@dataclass
class Job:
    """The one check that is running (or last ran)."""

    state: str = "idle"  # idle | running | done | error
    stage: int = 0  # index into STAGES
    video: str = ""
    report: str = ""  # path relative to home, set when done
    error: str = ""
    started: float = 0.0
    stage_started: float = 0.0
    duration: float = 0.0  # video length, drives the time estimate
    log: list[str] = field(default_factory=list)

    def advance(self, stage: int) -> None:
        if stage > self.stage:
            self.stage, self.stage_started = stage, time.time()

    def as_json(self) -> dict:
        now = time.time()
        if self.state == "done":
            percent, left = 100, 0
        elif self.state == "running":
            percent, left = progress(self.stage, now - self.stage_started, self.duration)
        else:
            percent, left = 0, 0
        return {
            "state": self.state,
            "stage": self.stage,
            "stages": [s[0] for s in STAGES],
            "percent": percent,
            "left": left,
            "video": self.video,
            "report": self.report,
            "error": self.error,
            "elapsed": int(now - self.started) if self.started else 0,
            "log": self.log[-8:],
        }


class _StageWriter(io.TextIOBase):
    """Capture the pipeline's printed progress into the job."""

    def __init__(self, job: Job) -> None:
        self._job = job
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._job.log.append(line.rstrip())
            for marker, stage in _STAGE_MARKERS:
                if marker in line:
                    self._job.advance(stage)
        return len(text)


def run_check(job: Job, home: Path, video: Path, srt: Path | None, lang: str) -> None:
    """Run one check on the full Sarvam stack and record the outcome in job."""
    from subtitle_checker import cli

    out_dir = home / "runs" / f"{video.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    args = argparse.Namespace(
        video=str(video), lang=lang, out=str(out_dir), asr=True, ocr="sarvam-vision",
        script=str(srt) if srt else None,
        # The ASR check overrides every alignment flag (0 of 48 survived on five
        # real clips), so the app skips the 1.2 GB alignment model entirely.
        no_align=True,
    )
    writer = _StageWriter(job)
    try:
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            code = cli._run_script_check(args) if srt else cli._run_check(args)
        report = out_dir / f"{video.stem}_report.html"
        if code == 0 and report.exists():
            job.report = report.relative_to(home).as_posix()
            job.state = "done"
        else:
            job.state = "error"
            job.error = job.log[-1] if job.log else "the check did not produce a report"
    except Exception as exc:  # surface any pipeline failure on the page
        job.state, job.error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        # The report embeds the frames and audio it needs; the uploaded copies
        # would only fill the disk.
        for upload in (video, srt):
            if upload is not None:
                upload.unlink(missing_ok=True)


def past_runs(home: Path) -> list[dict]:
    """Finished reports under home, newest first, with their mistakes pages."""
    entries = discover_reports([home / "runs"]) if (home / "runs").is_dir() else []
    runs = []
    for e in entries:
        mistakes = e.path.with_name(e.path.name.replace("_report.html", "_mistakes.html"))
        runs.append({
            "label": e.label,
            "report": e.path.relative_to(home).as_posix(),
            "mistakes": mistakes.relative_to(home).as_posix() if mistakes.exists() else "",
            "when": e.path.stat().st_mtime,
        })
    runs.sort(key=lambda r: r["when"], reverse=True)
    return runs


def _fmt_when(ts: float) -> str:
    return time.strftime("%d %b %Y, %H:%M", time.localtime(ts))


def render_app(runs: list[dict], *, has_key: bool, title: str = "Subtitle Checker") -> str:
    """Render the app page: key setup, new check form, progress, past runs."""
    lang_opts = "".join(f'<option value="{k}">{v}</option>' for k, v in LANGUAGES.items())
    if runs:
        rows = "".join(
            "<li>"
            f'<a href="/file/{quote(r["report"])}" target="_blank">{html.escape(r["label"])}</a>'
            + (
                f' <a class="sub" href="/file/{quote(r["mistakes"])}" target="_blank">mistakes</a>'
                if r["mistakes"] else ""
            )
            + f' <span class="when">{_fmt_when(r["when"])}</span></li>'
            for r in runs
        )
        history = f'<ul class="runs">{rows}</ul>'
    else:
        history = '<p class="muted">No checks yet. Your finished reports will appear here.</p>'

    key_card = (
        '<section class="card" id="keycard">'
        "<h2>Sarvam API key</h2>"
        '<p class="muted">Needed once. It is saved on this computer only.</p>'
        '<div class="row"><input id="key" type="password" placeholder="Paste the key">'
        '<button id="savekey">Save</button></div></section>'
    )
    body = (
        f'<div class="bar"><div class="brand">{html.escape(title)}</div>'
        '<button id="quit" class="quit">Quit app</button></div>'
        '<main class="wrap">'
        + ("" if has_key else key_card)
        + '<section class="card"><h2>New check</h2>'
        '<label class="lbl" for="video">Video</label>'
        '<input id="video" type="file" accept="video/*">'
        '<label class="lbl" for="srt">Subtitle file (optional)</label>'
        '<input id="srt" type="file" accept=".srt,.vtt">'
        '<p class="muted small">Add an SRT to check the audio against it instead of the '
        "subtitles burned into the video.</p>"
        '<label class="lbl" for="lang">Language</label>'
        f'<select id="lang">{lang_opts}</select>'
        f'<div class="row"><button id="run"{"" if has_key else " disabled"}>Run check</button></div>'
        '<div id="status" class="status" hidden></div>'
        "</section>"
        f'<section class="card"><h2>Past checks</h2>{history}</section>'
        "</main>"
        f"<script>{_SCRIPT}</script>"
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title>{_STYLE}{_APP_STYLE}</head>"
        f"<body>{body}</body></html>"
    )


_APP_STYLE = """<style>
  .wrap { max-width:44rem; margin:1.5rem auto; padding:0 1rem; display:grid; gap:1rem; }
  .card { border:1px solid #dde; border-radius:8px; padding:1rem 1.2rem; background:#fff; }
  .card h2 { margin:0 0 .6rem; font-size:1.05rem; }
  .lbl { display:block; margin:.7rem 0 .25rem; font-size:.8rem; font-weight:600; }
  .row { display:flex; gap:.6rem; margin-top:.9rem; }
  input[type=password] { flex:1; padding:.35rem .5rem; border:1px solid #ccd; border-radius:5px; }
  select { min-width:10rem; }
  button:disabled { background:#aab; cursor:not-allowed; }
  .muted { color:#667; } .small { font-size:.8rem; margin:.3rem 0 0; }
  .status { margin-top:1rem; padding:.7rem .9rem; border-radius:6px; background:#f3f7f4; }
  .status.error { background:#fdeeee; color:#8a1f1f; }
  .status pre { margin:.5rem 0 0; font-size:.75rem; white-space:pre-wrap; color:#556; }
  .track { height:.8rem; border-radius:.4rem; background:#dfe8e2; overflow:hidden; }
  .fill { height:100%; background:#2a6; transition:width .8s ease; }
  .pct { margin:.45rem 0 .3rem; font-size:.85rem; color:#445; }
  .steps { list-style:none; padding:0; margin:.4rem 0 0; font-size:.9rem; }
  .steps li { padding:.12rem 0; } .steps span { display:inline-block; width:1.3rem; }
  .steps .done { color:#1f6f43; } .steps .now { font-weight:600; } .steps .todo { color:#99a; }
  details { margin-top:.5rem; font-size:.8rem; color:#667; }
  .runs { list-style:none; padding:0; margin:0; }
  .runs li { padding:.45rem 0; border-bottom:1px solid #eef; }
  .runs a { color:#1f6f43; font-weight:600; } .runs a.sub { font-weight:400; font-size:.85rem; }
  .when { float:right; color:#889; font-size:.8rem; }
  .quit { background:transparent; border:1px solid rgba(255,255,255,.6); font-weight:400; }
</style>"""

_SCRIPT = """
(function(){
var $=function(i){return document.getElementById(i);};
var box=$('status');
var STEPS=['Uploading the video','Reading the subtitles','Checking the audio','Building the report'];
function esc(t){return t.replace(/[&<>]/g,function(c){
  return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c];});}
function mins(s){return s<60?'under a minute':'about '+Math.ceil(s/60)+' min';}
function bar(pct,step,note,log){box.hidden=false;box.className='status';
  var items=STEPS.map(function(n,i){var c=i<step?'done':(i===step?'now':'todo');
    var m=i<step?'&#10003;':(i===step?'&#9654;':'&middot;');
    return '<li class="'+c+'"><span>'+m+'</span>'+n+'</li>';}).join('');
  box.innerHTML='<div class="track"><div class="fill" style="width:'+pct+'%"></div></div>'
    +'<div class="pct"><b>'+pct+'%</b> '+note+'</div><ol class="steps">'+items+'</ol>'
    +(log?'<details><summary>Details</summary><pre>'+esc(log)+'</pre></details>':'');}
function fail(msg,log){box.hidden=false;box.className='status error';
  box.innerHTML=esc(msg)+(log?'<pre>'+esc(log)+'</pre>':'');$('run').disabled=false;}
if($('savekey')){$('savekey').onclick=function(){
  var k=$('key').value.trim(); if(!k){return;}
  fetch('/key',{method:'POST',body:k}).then(function(){location.reload();});};}
function upload(file,onpct){return new Promise(function(ok,no){
  var x=new XMLHttpRequest(); x.open('POST','/upload?name='+encodeURIComponent(file.name));
  x.upload.onprogress=function(e){if(e.lengthComputable){onpct(e.loaded/e.total);}};
  x.onload=function(){x.status===200?ok(x.responseText):no(new Error('Upload failed.'));};
  x.onerror=function(){no(new Error('Upload failed.'));}; x.send(file);});}
function poll(){fetch('/job').then(function(r){return r.json();}).then(function(j){
  if(j.state==='running'){bar(j.percent,j.stage+1,'&middot; '+mins(j.left)+' left (estimate)'
    +' &middot; keep this page open',j.log.join('\\n'));setTimeout(poll,1000);}
  else if(j.state==='done'){bar(100,STEPS.length,'Done. Opening the report...');
    window.open('/file/'+encodeURI(j.report),'_blank');setTimeout(function(){location.reload();},1500);}
  else if(j.state==='error'){fail('The check failed: '+j.error,j.log.join('\\n'));}});}
$('run').onclick=function(){
  var v=$('video').files[0], s=$('srt').files[0];
  if(!v){fail('Pick a video first.');return;}
  $('run').disabled=true; bar(0,0,'&middot; copying the video into the app');
  upload(v,function(f){bar(Math.round(f*10),0,'&middot; copying the video into the app');})
  .then(function(vn){return (s?upload(s,function(){}):Promise.resolve('')).then(function(sn){
    return fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({video:vn,srt:sn,lang:$('lang').value})});});})
  .then(function(r){if(!r.ok){return r.text().then(function(t){throw new Error(t);});}
    poll();}).catch(function(e){fail(e.message);});};
fetch('/job').then(function(r){return r.json();}).then(function(j){
  if(j.state==='running'){$('run').disabled=true;poll();}});
$('quit').onclick=function(){
  fetch('/job').then(function(r){return r.json();}).then(function(j){
    if(j.state==='running'&&!confirm('A check is running. Quit anyway?')){return;}
    fetch('/quit',{method:'POST'}).then(function(){document.body.innerHTML=
      '<p style="padding:2rem">Subtitle Checker has closed. You can close this tab.</p>';});});};
})();
"""


def _make_handler(home: Path, job: Job, lock: threading.Lock) -> type[BaseHTTPRequestHandler]:
    uploads = home / "uploads"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if self.path in ("/", "/index.html"):
                page = render_app(past_runs(home), has_key=bool(os.environ.get(KEY_ENV)))
                self._send(200, page.encode("utf-8"))
            elif self.path == "/job":
                self._send(200, json.dumps(job.as_json()).encode("utf-8"), "application/json")
            elif self.path.startswith("/file/"):
                path = resolve_inside(home, self.path[len("/file/"):])
                if path is not None and path.suffix == ".html":
                    self._send(200, path.read_bytes())
                else:
                    self._send(404, b"not found")
            else:
                self._send(404, b"not found")

        def do_POST(self) -> None:  # noqa: N802 (http.server API)
            length = int(self.headers.get("Content-Length") or 0)
            if self.path.startswith("/upload?name="):
                name = safe_name(unquote(self.path.split("=", 1)[1]))
                uploads.mkdir(parents=True, exist_ok=True)
                with (uploads / name).open("wb") as fh:
                    remaining = length
                    while remaining > 0:
                        chunk = self.rfile.read(min(_CHUNK, remaining))
                        if not chunk:
                            break
                        fh.write(chunk)
                        remaining -= len(chunk)
                self._send(200, name.encode("utf-8"), "text/plain")
            elif self.path == "/key":
                key = self.rfile.read(length).decode("utf-8").strip()
                if key:
                    save_key(home, key)
                self._send(200, b"ok", "text/plain")
            elif self.path == "/quit":
                self._send(200, b"bye", "text/plain")
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            elif self.path == "/run":
                self._start(json.loads(self.rfile.read(length) or b"{}"))
            else:
                self._send(404, b"not found")

        def _start(self, req: dict) -> None:
            video = uploads / safe_name(req.get("video", ""))
            srt = uploads / safe_name(req["srt"]) if req.get("srt") else None
            lang = req.get("lang") if req.get("lang") in LANGUAGES else "hi"
            if not os.environ.get(KEY_ENV):
                self._send(400, b"Add the Sarvam API key first.", "text/plain")
                return
            if not video.is_file():
                self._send(400, b"The video did not upload. Try again.", "text/plain")
                return
            with lock:
                if job.state == "running":
                    self._send(409, b"A check is already running.", "text/plain")
                    return
                now = time.time()
                job.__init__(state="running", video=video.name, started=now,
                             stage_started=now, duration=video_duration(video))
            threading.Thread(
                target=run_check, args=(job, home, video, srt, lang), daemon=True
            ).start()
            self._send(200, b"started", "text/plain")

        def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:  # keep the console clean
            pass

    return Handler


def serve_app(
    home: Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
) -> None:
    """Run the app on this machine until interrupted."""
    home = home or default_home()
    home.mkdir(parents=True, exist_ok=True)
    load_key(home)
    handler = _make_handler(home, Job(), threading.Lock())
    url = f"http://{host}:{port}/"
    try:
        httpd = ThreadingHTTPServer((host, port), handler)
    except OSError:
        # Opened a second time while already running: show the running app.
        print(f"Subtitle Checker is already running at {url}")
        if open_browser:
            webbrowser.open(url)
        return
    print(f"Subtitle Checker running at {url}  (files in {home})")
    print("Stop it with the Quit app button on the page (or Ctrl+C here).")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
