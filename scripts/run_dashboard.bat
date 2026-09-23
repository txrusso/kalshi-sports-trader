@echo off
rem Live terminal dashboard for the trading loop -- READ-ONLY VIEWER.
rem Safe to start, stop and restart at any time: it never places an order and
rem never writes to the ledgers, so it cannot affect a running loop. See
rem output/live_dashboard.py's module docstring for why this is a separate process.
cd /d "%~dp0.."
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

rem Clear NO_COLOR for this window only. Textual honours it and renders the
rem whole dashboard in greys, which loses the green/red profit signal that is
rem the point of the thing. It is cleared here rather than in the dashboard
rem because it is almost never a real preference in this context: it is
rem inherited from whatever spawned the window (an agent/CI session sets it for
rem its own subprocesses, and Start-Process passes the environment straight
rem through), and this launcher exists specifically to open the colour
rem dashboard. Running `python -m output.live_dashboard` directly still honours it.
set NO_COLOR=

.venv\Scripts\python.exe -m output.live_dashboard %*
