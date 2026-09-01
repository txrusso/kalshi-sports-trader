@echo off
cd /d "C:\Users\txrus\Portfolio\kalshi-agent"
set PYTHONUNBUFFERED=1
echo ===== paper loop started %DATE% %TIME% =====>> logs\paper_loop.log
".venv\Scripts\python.exe" -u cli.py loop --interval 1800 --paper >> logs\paper_loop.log 2>&1
