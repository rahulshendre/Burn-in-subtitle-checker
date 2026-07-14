"""Minimal local web UI to show the tool.

This is the "minimal, not Streamlit, on-device" viewer the mentor asked for. It
does not run the pipeline; it serves the self-contained HTML reports that the
report stage already produced (base64 frames and audio embedded), behind a small
picker. Loading a pre-built report is instant and cannot fail mid-demo, which is
the whole point when presenting live.

Stdlib only (http.server), so there is no framework to install and the whole
thing packages cleanly into an on-device executable later. The picker shell is a
pure function (render_index) so it can be unit-tested without a socket.
"""

from __future__ import annotations

import html
import re
import threading
import webbrowser
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote


@dataclass(frozen=True)
class ReportEntry:
    """One pre-built report the UI can show."""

    id: str
    label: str
    path: Path


def _clean_label(stem: str) -> str:
    """Turn a report filename stem into a human label."""
    name = re.sub(r"_report$", "", stem)
    name = name.replace("_", " ").strip()
    return re.sub(r"\s+", " ", name) or stem


def discover_reports(roots: list[Path]) -> list[ReportEntry]:
    """Collect *_report.html files from the given files or directories.

    Directories are searched recursively. Duplicate paths are dropped, order is
    stable, and each entry gets a positional id used in its URL.
    """
    seen: set[Path] = set()
    paths: list[Path] = []
    for root in roots:
        if root.is_file() and root.name.endswith("_report.html"):
            found = [root]
        elif root.is_dir():
            found = sorted(root.rglob("*_report.html"))
        else:
            found = []
        for p in found:
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                paths.append(p)
    return [
        ReportEntry(id=str(i), label=_clean_label(p.stem), path=p)
        for i, p in enumerate(paths)
    ]


def relabel(entries: list[ReportEntry], labels: list[str]) -> list[ReportEntry]:
    """Override entry labels positionally; unmatched entries keep their filename label."""
    return [
        replace(e, label=labels[i].strip())
        if i < len(labels) and labels[i].strip()
        else e
        for i, e in enumerate(entries)
    ]


def render_index(entries: list[ReportEntry], *, title: str = "Subtitle Checker") -> str:
    """Render the picker shell that loads a report into an iframe."""
    if entries:
        options = "".join(
            f'<option value="{e.id}">{html.escape(e.label)}</option>' for e in entries
        )
        controls = (
            f'<select id="pick">{options}</select>'
            '<button id="go">Check</button>'
        )
        body = (
            f'<div class="bar"><div class="brand">{html.escape(title)}</div>'
            f"<div class=\"controls\">{controls}</div></div>"
            '<iframe id="view" title="report"></iframe>'
            "<script>"
            "var pick=document.getElementById('pick');"
            "var view=document.getElementById('view');"
            "function show(){view.src='/report/'+encodeURIComponent(pick.value);}"
            "document.getElementById('go').onclick=show;"
            "pick.onchange=show;show();"
            "</script>"
        )
    else:
        body = (
            f'<div class="bar"><div class="brand">{html.escape(title)}</div></div>'
            '<p class="empty">No reports found. Generate one with '
            "<code>subtitle-checker check</code> first.</p>"
        )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title>{_STYLE}</head><body>{body}</body></html>"
    )


_STYLE = """<style>
  * { box-sizing:border-box; }
  body { font-family:-apple-system,"Segoe UI",Roboto,sans-serif; margin:0;
         color:#222; }
  .bar { display:flex; align-items:center; justify-content:space-between;
         gap:1rem; padding:.7rem 1.2rem; background:#1f6f43; color:#fff;
         box-shadow:0 1px 4px rgba(0,0,0,.2); }
  .brand { font-weight:600; font-size:1.1rem; }
  .controls { display:flex; gap:.5rem; }
  select { font-size:.95rem; padding:.35rem .5rem; border-radius:5px;
           border:1px solid #cfe; min-width:16rem; }
  button { font-size:.95rem; padding:.35rem 1rem; border:0; border-radius:5px;
           background:#2a6; color:#fff; font-weight:600; cursor:pointer; }
  button:hover { background:#238; background:#1f6f43; }
  iframe { width:100%; height:calc(100vh - 52px); border:0; background:#fff; }
  .empty { color:#777; padding:2rem 1.2rem; }
</style>"""


def _make_handler(
    index_html: str, id_to_path: dict[str, Path]
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if self.path in ("/", "/index.html"):
                self._send(200, index_html.encode("utf-8"))
                return
            if self.path.startswith("/report/"):
                rid = unquote(self.path[len("/report/") :])
                path = id_to_path.get(rid)
                if path is not None and path.exists():
                    self._send(200, path.read_bytes())
                    return
            self._send(404, b"not found")

        def _send(self, code: int, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:  # keep the console clean
            pass

    return Handler


def serve(
    entries: list[ReportEntry],
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    title: str = "Subtitle Checker",
) -> None:
    """Serve the picker and reports on a local port until interrupted."""
    index_html = render_index(entries, title=title)
    handler = _make_handler(index_html, {e.id: e.path for e in entries})
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"{title} running at {url}  ({len(entries)} report(s))")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
