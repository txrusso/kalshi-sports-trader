' Launch the phone dashboard server (serve_dashboard.bat) with no visible window.
'
' Run by the KalshiDashboardServe scheduled task (daily 10:00, and at logon).
' It is a READ-ONLY viewer served over Tailscale only -- see
' scripts\serve_dashboard.py -- and is deliberately a separate task from the
' trading loop, so a problem here can never affect trading.
'
' Output is appended to logs\dashboard_serve.log.
'
' bWaitOnReturn = True keeps the task instance alive for as long as the server
' runs, so the task's MultipleInstancesPolicy=IgnoreNew stops a second server
' from starting (which would fail anyway: the port is taken), and a crash
' (non-zero exit) is seen by the task's restart-on-failure setting.

Option Explicit

Dim shell, here, root, exitCode
Set shell = CreateObject("WScript.Shell")

here = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
root = Left(here, InStrRev(Left(here, Len(here) - 1), "\"))
shell.CurrentDirectory = root

' 0 = hidden window, True = wait for it to finish
exitCode = shell.Run("cmd /c """"" & here & "serve_dashboard.bat"" >> """ & root & "logs\dashboard_serve.log"" 2>&1""", 0, True)

WScript.Quit exitCode
