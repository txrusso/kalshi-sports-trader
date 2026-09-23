' Launch Stakey's trading loop with no visible console window.
'
' The scheduled task KalshiPaperLoop points at this instead of at
' run_paper_loop.bat directly, so the loop runs unattended and the dashboard
' (output/live_dashboard.py, opened by the bat) is the only window on screen. Nothing
' about the loop itself changes: same bat, same working directory, same
' logging to logs\paper_loop.log.
'
' Hidden means the console WINDOW is not shown -- the process still has a
' console, so the bat's `Write-Host` echo into the logging pipeline still
' works exactly as before.
'
' bWaitOnReturn = True is the important part, and is not cosmetic: wscript
' blocks here until the bat exits, so the scheduled-task INSTANCE stays alive
' for the whole life of the loop. That is what keeps the task's
' ExecutionTimeLimit (15h) meaningful and what lets its
' MultipleInstancesPolicy=IgnoreNew actually block a second loop. Launching
' fire-and-forget (False) would let the task instance exit immediately while
' the loop kept running, recreating exactly the orphaned-tree situation that
' stacked two live loops on 2026-09-23.

Option Explicit

Dim shell, here, exitCode
Set shell = CreateObject("WScript.Shell")

here = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
shell.CurrentDirectory = here

' 0 = hidden window, True = wait for it to finish
exitCode = shell.Run("""" & here & "run_paper_loop.bat""", 0, True)

WScript.Quit exitCode
