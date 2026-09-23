@echo off
rem Nightly snapshot compression (KalshiSnapshotCompress, 23:30).
rem
rem Runs as its own scheduled task rather than from run_paper_loop.bat's exit
rem block, for two reasons: it should happen whether or not the loop exited
rem cleanly, and adding it to that bat would have meant editing a file cmd.exe
rem was mid-way through executing -- which reads batch files incrementally by
rem byte offset and spawned duplicate loops twice on 2026-09-23.
rem
rem 23:30 is after the 23:00 stop, but the timing barely matters: today's file
rem is never compressed (the loop appends to it), so this only ever touches
rem days that are already closed.
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\compress_snapshots.py >> logs\snapshot_compress.log 2>&1
