"""Serve the live dashboard as a web page, for viewing on a phone.

    .venv\\Scripts\\python.exe scripts\\serve_dashboard.py            # Tailscale address
    .venv\\Scripts\\python.exe scripts\\serve_dashboard.py --local    # this laptop only

Uses textual-serve: every browser that connects gets its OWN copy of
output/live_dashboard.py, rendered in the page -- same data, same tabs, same
keys. The dashboard is read-only (it cannot place, cancel or change an order),
so a viewer can only look.

WHERE IT LISTENS -- deliberately narrow, because the page has NO login and shows
the balance and every bet:
- default: the laptop's Tailscale IP only (100.x.y.z). Reachable from your own
  devices signed in to your tailnet -- from anywhere -- and from nothing else:
  not the café Wi-Fi, not the internet.
- --local: 127.0.0.1, this laptop only (for testing).
- It refuses to start on 0.0.0.0 / a LAN address unless --host is given
  explicitly, and never opens a public link. Do not port-forward this.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_PORT = 8765


def tailscale_ip() -> str | None:
    """This machine's Tailscale IPv4, or None if Tailscale isn't installed/up."""
    exe = shutil.which("tailscale") or r"C:\Program Files\Tailscale\tailscale.exe"
    if not Path(exe).exists() and not shutil.which("tailscale"):
        return None
    try:
        out = subprocess.run([exe, "ip", "-4"], capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    ip = (out.stdout or "").strip().splitlines()
    return ip[0].strip() if out.returncode == 0 and ip and ip[0].startswith("100.") else None


QUIT_GRACE_SECONDS = 5


def _end_sessions_reliably() -> None:
    """Make a closed browser tab actually end its dashboard.

    textual-serve starts each session through a shell and, on disconnect, sends
    the app a "quit" and waits for it to exit. On Windows that session is three
    processes deep (cmd -> .venv launcher -> anaconda worker) and was observed
    NOT to exit -- the server then waits forever and every phone visit would
    leave a dashboard running, polling Kalshi. So: send the normal quit, give it
    a few seconds, then end that session's whole process tree by its own PID.
    Only the session's tree is touched -- never the desktop dashboard or the loop.
    """
    import asyncio
    from textual_serve import app_service

    original_stop = app_service.AppService.stop

    async def stop(self) -> None:
        # textual-serve calls stop() twice on a disconnect; only act once.
        if getattr(self, "_stakey_stopped", False):
            return
        self._stakey_stopped = True
        proc = getattr(self, "_process", None)
        try:
            await asyncio.wait_for(original_stop(self), timeout=QUIT_GRACE_SECONDS)
        except (asyncio.TimeoutError, Exception):
            pass
        finally:
            if proc is not None and proc.returncode is None:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True)
                print(f"ended dashboard session (pid {proc.pid}) after disconnect")
            self._task = None

    app_service.AppService.stop = stop


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Serve the Stakey dashboard to a browser.")
    ap.add_argument("--local", action="store_true", help="listen on 127.0.0.1 only")
    ap.add_argument("--host", help="explicit address to listen on (you own the exposure)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args(argv)

    if args.host:
        host = args.host
    elif args.local:
        host = "127.0.0.1"
    else:
        host = tailscale_ip()
        if host is None:
            print("Tailscale isn't installed or connected on this laptop, so there is no "
                  "private address to serve on.\n"
                  "  - Install Tailscale and sign in, then run this again, or\n"
                  "  - use --local to test on this laptop only.", file=sys.stderr)
            return 2

    # Same fixes as scripts/run_dashboard.bat: repo root as cwd (the dashboard
    # resolves output/, logs/ and .venv relative to it) and colour on. Each
    # session inherits this environment.
    os.chdir(ROOT)
    os.environ.pop("NO_COLOR", None)
    os.environ["PYTHONIOENCODING"] = "utf-8"

    _end_sessions_reliably()
    from textual_serve.server import Server
    # Our copy of textual-serve's page, plus swipe-to-scroll for phones
    # (see the note at the bottom of scripts/dashboard_web/app_index.html).
    server = Server(f'"{PYTHON}" -m output.live_dashboard', host=host, port=args.port,
                    title="Stakey", templates_path=ROOT / "scripts" / "dashboard_web")
    print(f"Stakey dashboard: http://{host}:{args.port}  (Ctrl+C to stop)")
    server.serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
