@echo off
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

rem Rotate the log once it gets big (>5 MB). This is also the clean seam for the
rem 2026-09-22 encoding fix below: the pre-fix file is half UTF-16 and half
rem UTF-8, which made it a binary blob to grep/tail and is a large part of why
rem the notification problem went undiagnosed for days. Rotating starts a
rem consistently-encoded file without throwing the history away. Done in
rem PowerShell because batch date/time formatting is locale-dependent and
rem unreliable for building a filename.
rem The try/catch matters: if another process still holds the log open (an old
rem loop that hasn't fully exited), Move-Item fails and would otherwise dump a
rem PowerShell error into the log. Rotation is housekeeping -- skipping it must
rem never be louder than, or get in the way of, actually starting the loop.
powershell -NoProfile -Command "$f='logs\paper_loop.log'; try { if ((Test-Path $f) -and ((Get-Item $f).Length -gt 5MB)) { Move-Item $f ('logs\paper_loop_' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log') -Force -ErrorAction Stop } } catch { }"

echo ===== paper loop started %DATE% %TIME% =====>> logs\paper_loop.log

rem Open the live dashboard (run_dashboard.py) in its own window. It is a
rem READ-ONLY viewer in a SEPARATE process -- it cannot place an order and
rem cannot affect this loop. It is launched separately rather than wrapped
rem around the loop because the loop's stdout is piped into the log below,
rem and a TUI cannot render into a pipe.
rem Guarded two ways: skipped if a dashboard is already up (so a loop
rem restart doesn't stack windows), and wrapped in try/catch so a viewer
rem that fails to open can never stop the loop from starting.
powershell -NoProfile -Command "try { $up = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*run_dashboard.py*' }; if (-not $up) { $here = (Get-Location).Path; if (Get-Command wt.exe -ErrorAction SilentlyContinue) { Start-Process wt.exe -ArgumentList '-d', $here, 'cmd', '/k', 'run_dashboard.bat' } else { Start-Process cmd.exe -ArgumentList '/k','run_dashboard.bat' -WorkingDirectory $here } } } catch { }"

rem NOT `Tee-Object -FilePath`: in Windows PowerShell 5.1 that writes UTF-16LE,
rem so the log became unreadable to grep, tail, and every other text tool.
rem ForEach-Object keeps the console echo (useful when this is run by hand)
rem while pinning the file to UTF-8, matching what cmd's own `>>` above writes.
powershell -NoProfile -Command "& '.venv\Scripts\python.exe' -u cli.py loop --interval 600 --live --in-game 2>&1 | ForEach-Object { Write-Host $_; Add-Content -Path 'logs\paper_loop.log' -Value $_ -Encoding UTF8 }"
