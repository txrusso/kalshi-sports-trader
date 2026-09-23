@echo off
rem Live terminal dashboard for the trading loop -- READ-ONLY VIEWER.
rem Safe to start, stop and restart at any time: it never places an order and
rem never writes to the ledgers, so it cannot affect a running loop. See
rem run_dashboard.py's module docstring for why this is a separate process.
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe run_dashboard.py %*
