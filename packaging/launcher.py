"""Entry point of the packaged app (Mac and Windows).

A double-clicked app has no terminal, so the bundled ffmpeg/ffprobe are put on
PATH here and all output goes to a log file in the app's folder, where a tester
can find it when something goes wrong.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _hide_child_consoles() -> None:
    """On Windows, keep every ffmpeg/ffprobe child from flashing a console.

    The app itself has no console (PyInstaller --noconsole), so each child
    process would otherwise open its own terminal window for a moment.
    """
    if sys.platform != "win32":
        return
    real_popen = subprocess.Popen

    class _QuietPopen(real_popen):
        def __init__(self, *args, **kwargs):
            kwargs["creationflags"] = kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
            super().__init__(*args, **kwargs)

    subprocess.Popen = _QuietPopen


def _serve_on_mac(home: Path, port: int) -> bool:
    """Serve while the main thread runs a Cocoa loop, so the Dock's Quit and Cmd-Q work.

    A plain Python process never reads the quit request macOS sends from the Dock
    or Cmd-Q, so it timed out and the app stayed open. Returns False when the Mac
    libraries are missing, so the caller can serve the old way.
    """
    try:
        from AppKit import (
            NSApplication,
            NSApplicationActivationPolicyRegular,
            NSMenu,
            NSMenuItem,
            NSTerminateNow,
        )
        from Foundation import NSObject
        from PyObjCTools import AppHelper
    except ImportError:
        return False
    import threading

    from subtitle_checker.report.app import clear_leftovers, serve_app

    running: dict = {}

    def stop() -> None:
        server = running.get("server")
        if server is not None:
            server.shutdown()
            server.server_close()
        clear_leftovers(home)

    class Delegate(NSObject):
        def applicationShouldTerminate_(self, sender):  # noqa: N802 (Cocoa name)
            stop()
            return NSTerminateNow

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    delegate = Delegate.alloc().init()
    app.setDelegate_(delegate)
    menubar, top, menu = NSMenu.alloc().init(), NSMenuItem.alloc().init(), NSMenu.alloc().init()
    menu.addItem_(NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        "Quit Subtitle Checker", "terminate:", "q"))
    top.setSubmenu_(menu)
    menubar.addItem_(top)
    app.setMainMenu_(menubar)

    def serve() -> None:
        try:
            serve_app(home, port=port, ready=lambda server: running.update(server=server))
        finally:  # the page's Quit button ends the server; then the app ends too
            AppHelper.callAfter(app.terminate_, None)

    threading.Thread(target=serve, daemon=True).start()
    app.run()
    return True


def main() -> None:
    bundle = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    _hide_child_consoles()
    os.environ["PATH"] = str(bundle / "bin") + os.pathsep + os.environ.get("PATH", "")

    from subtitle_checker.report.app import default_home, serve_app

    home = default_home()
    home.mkdir(parents=True, exist_ok=True)
    log = open(home / "app.log", "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    sys.stdout = sys.stderr = log
    # Port 8000 unless another program already uses it; then the next free port is
    # taken. Set SUBTITLE_CHECKER_PORT to start the search somewhere else.
    port = os.environ.get("SUBTITLE_CHECKER_PORT", "")
    port_number = int(port) if port.isdigit() else 8000
    if sys.platform == "darwin" and _serve_on_mac(home, port_number):
        return
    serve_app(home, port=port_number)


if __name__ == "__main__":
    main()
