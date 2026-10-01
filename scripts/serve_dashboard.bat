@echo off
rem Serve the live dashboard to your phone's browser, over Tailscale only.
rem READ-ONLY, like the desktop dashboard: it cannot place or change an order.
rem Listens on this laptop's Tailscale address, never the LAN or the internet --
rem see scripts\serve_dashboard.py. Open the printed http://100.x.y.z:8765
rem address on your phone (with the Tailscale app connected).
rem   serve_dashboard.bat --local    test on this laptop only (127.0.0.1)
cd /d "%~dp0.."
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set NO_COLOR=
.venv\Scripts\python.exe scripts\serve_dashboard.py %*
