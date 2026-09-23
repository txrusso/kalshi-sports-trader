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

rem The loop runs under a supervisor (run_loop_supervisor.ps1) rather than being
rem launched directly. Two reasons, both learned the hard way on 2026-09-23:
rem   1. AUTO-RESTART. If python dies -- crash, OOM, stray kill -- the supervisor
rem      brings it straight back, instead of the machine sitting there all day
rem      with no loop and nothing saying so.
rem   2. THE DAILY STOP ACTUALLY HAPPENS. Task Scheduler cannot stop this loop:
rem      its ExecutionTimeLimit and `schtasks /End` terminate the task's OWN
rem      process, not the descendant python tree (observed twice -- /End reported
rem      SUCCESS while both python PIDs kept scanning). So the 15h limit ended the
rem      task instance while the loop ran on forever, and the next morning's 10:00
rem      run started a SECOND live loop on top of the survivor. The KalshiLoopStop
rem      task now kills python at 23:00 and the supervisor declines to restart it
rem      past that, so this bat reaches its exit block and the task instance
rem      finally completes -- which is what makes tomorrow start clean.
rem The supervisor owns the logging pipeline that used to live on this line (NOT
rem `Tee-Object -FilePath`: in Windows PowerShell 5.1 that writes UTF-16LE and
rem makes the log unreadable to grep, tail and every other text tool).
powershell -NoProfile -ExecutionPolicy Bypass -File run_loop_supervisor.ps1

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

