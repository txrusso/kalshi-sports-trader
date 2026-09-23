@echo off
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

rem Rotate the log once it gets big (>5 MB). The REAL rotation now happens on
rem loop EXIT, at the bottom of this file -- this one is only the fallback for
rem a session that was hard-killed before it could get there (`schtasks /End`
rem takes the whole tree down, bat included).
rem Rotating at start could not be relied on, and in practice almost never
rem fired: start is exactly the moment a previous loop is most likely to still
rem be holding the file, and Move-Item then fails. The log sat at 8 MB against
rem a 5 MB threshold for days because of this, and the same lock silently ate
rem the `paper loop started` marker below (cmd's >> cannot open a file
rem PowerShell's Add-Content has open), which is a large part of why a
rem duplicate loop went unnoticed on 2026-09-23.
rem Done in PowerShell because batch date/time formatting is locale-dependent
rem and unreliable for building a filename. The try/catch still matters: if
rem something does hold the log, Move-Item fails and would otherwise dump a
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
powershell -NoProfile -Command "try { $up = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*run_dashboard.py*' }; if (-not $up) { $here = (Get-Location).Path; $bat = Join-Path $here 'run_dashboard.bat'; if (Get-Command wt.exe -ErrorAction SilentlyContinue) { Start-Process wt.exe -ArgumentList '-d', $here, 'cmd', '/k', $bat } else { Start-Process cmd.exe -ArgumentList '/k', $bat -WorkingDirectory $here } } } catch { }"

rem NOT `Tee-Object -FilePath`: in Windows PowerShell 5.1 that writes UTF-16LE,
rem so the log became unreadable to grep, tail, and every other text tool.
rem ForEach-Object keeps the console echo (useful when this is run by hand)
rem while pinning the file to UTF-8, matching what cmd's own `>>` above writes.
powershell -NoProfile -Command "& '.venv\Scripts\python.exe' -u cli.py loop --interval 600 --live --in-game 2>&1 | ForEach-Object { Write-Host $_; Add-Content -Path 'logs\paper_loop.log' -Value $_ -Encoding UTF8 }"

rem ---------------------------------------------------------------------------
rem The loop has exited, so NOTHING holds logs\paper_loop.log any more -- this is
rem the one moment rotation can actually succeed, which is why it lives here and
rem not only at the top. Reached whenever the python tree is stopped but this bat
rem survives, which is what the documented restart does (`taskkill /PID <launcher>
rem /T /F` kills python and lets this pipeline drain). A `schtasks /End` kills the
rem bat too and skips this; the fallback rotate at the top then catches it on the
rem next start, when the file is finally free.
echo ===== paper loop exited %DATE% %TIME% =====>> logs\paper_loop.log
powershell -NoProfile -Command "$f='logs\paper_loop.log'; try { if ((Test-Path $f) -and ((Get-Item $f).Length -gt 5MB)) { Move-Item $f ('logs\paper_loop_' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log') -Force -ErrorAction Stop } } catch { }"

