@echo off
cd /d "C:\Users\txrus\Portfolio\kalshi-agent"
set PYTHONUNBUFFERED=1
echo ===== settle %DATE% %TIME% =====>> logs\settle.log
".venv\Scripts\python.exe" -u cli.py settle --notify >> logs\settle.log 2>&1
