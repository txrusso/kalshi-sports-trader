@echo off
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
echo ===== paper loop started %DATE% %TIME% =====>> logs\paper_loop.log
powershell -NoProfile -Command "& '.venv\Scripts\python.exe' -u cli.py loop --interval 600 --live --in-game 2>&1 | Tee-Object -FilePath 'logs\paper_loop.log' -Append"
