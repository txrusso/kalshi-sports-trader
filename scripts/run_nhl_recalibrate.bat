@echo off
rem Nightly NHL totals recalibration. Runs at 09:00, BEFORE the 10:00 loop
rem start, for two reasons: every one of the previous night's games is final
rem by then (west-coast puck drops end well after the 23:00 settle), and the
rem loop reads the calibration at start, so a promoted value is live for the
rem whole trading day rather than waiting for the following one.
rem
rem --apply promotes a setting only when it beats the live one on BOTH
rem walk-forward splits; most nights this correctly changes nothing.
cd /d "%~dp0.."
set PYTHONUNBUFFERED=1
echo ===== nhl recalibrate %DATE% %TIME% =====>> logs\nhl_recalibrate.log
".venv\Scripts\python.exe" -u -m backtest.nhl_recalibrate --apply >> logs\nhl_recalibrate.log 2>&1
