<#
Supervisor for Stakey's trading loop: restarts it if it dies, and owns its
daily lifetime so Task Scheduler does not have to.

WHY THIS EXISTS
---------------
Task Scheduler cannot stop this loop. Its ExecutionTimeLimit (and `schtasks
/End`) terminate the task's OWN process, not the descendant python tree --
observed twice on 2026-09-23, where `/End` reported SUCCESS while both python
PIDs kept running and scanning. So the "15h limit" quietly ended the task
instance while the loop carried on forever, and the next morning's 10:00 run
started a SECOND live loop on top of the survivor. Two loops both submitting
real orders is how 2026-09-23 began.

The fix is to stop relying on Task Scheduler for the stop:
  * this supervisor computes its own stop instant, once, at launch;
  * the KalshiLoopStop task kills the python tree at that time;
  * the supervisor sees python exit at/after the stop and declines to restart,
    so the bat's exit block runs (log marker + rotation) and the task instance
    finally completes on its own -- which is what lets the next day start clean.

Anything else that kills python -- a crash, an OOM, a stray taskkill, a
transient API blowup -- is BEFORE the stop, so it gets restarted instead.

The python tree is deliberately killed by a separate task rather than from
here: this process is blocked on the loop's output pipeline (that pipeline is
what writes the log), so it cannot watch the clock at the same time.

TESTABILITY
-----------
Every knob is a parameter so the restart/stop logic can be exercised offline
against a dummy command instead of the real loop. Defaults are production.
#>
[CmdletBinding()]
param(
    # The loop itself. Defaults match what the scheduled task has always run.
    [string]   $Exe = '.venv\Scripts\python.exe',
    [string[]] $LoopArgs = @('-u', 'cli.py', 'loop', '--interval', '600', '--live', '--in-game'),

    # Local hour at which the loop is done for the day. The KalshiLoopStop task
    # kills python at this time; this is what stops it being restarted.
    [int] $StopHour = 23,

    # Explicit stop instant, overriding $StopHour. Exists so the "python exited
    # at/after the stop" branch can be tested without waiting for 23:00 -- the
    # computed stop is always in the future, so it can only be crossed DURING a
    # run. Also handy for a one-off "run until X tonight".
    [datetime] $StopAt,

    # A crash-looping start would otherwise spin forever. Well above any
    # plausible real restart count for one day.
    [int] $MaxRestarts = 50,
    [int] $BackoffSeconds = 30,

    [string] $Log = 'logs\paper_loop.log'
)

$ErrorActionPreference = 'Continue'

# The bat already cds here, but the loop runs `cli.py` from the repo root and
# resolves .venv relatively, so make that guarantee explicit rather than
# inherited -- this script is only ever valid when run from the repo.
# It lives in scripts/, so the repo root is its parent.
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)

function Write-Marker([string] $Message) {
    # Markers go to the same log as the loop's own output so the restart story
    # is readable in one place, in time order, next to the cycle lines.
    $line = "===== supervisor $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message ====="
    Write-Host $line
    try { Add-Content -Path $Log -Value $line -Encoding UTF8 } catch { }
}

# Computed ONCE, from launch time, so a manual start after the stop hour runs
# through to the NEXT day's stop instead of exiting immediately.
if ($PSBoundParameters.ContainsKey('StopAt')) {
    $stopAt = $StopAt
} else {
    $stopAt = (Get-Date).Date.AddHours($StopHour)
    if ((Get-Date) -ge $stopAt) { $stopAt = $stopAt.AddDays(1) }
}
Write-Marker "start; stop at $($stopAt.ToString('yyyy-MM-dd HH:mm'))"

$restarts = 0
while ($true) {
    # The pipeline is the logging path: stdout+stderr are echoed to the (hidden)
    # console and appended as UTF-8. NOT Tee-Object -- in Windows PowerShell 5.1
    # that writes UTF-16LE and makes the log unreadable to every text tool.
    & $Exe @LoopArgs 2>&1 | ForEach-Object {
        Write-Host $_
        try { Add-Content -Path $Log -Value $_ -Encoding UTF8 } catch { }
    }
    $code = $LASTEXITCODE

    if ((Get-Date) -ge $stopAt) {
        Write-Marker "loop exited (code $code) at/after the daily stop -- not restarting"
        break
    }

    if (($restarts + 1) -gt $MaxRestarts) {
        Write-Marker "loop exited (code $code); restart cap $MaxRestarts reached -- giving up"
        break
    }
    $restarts++

    Write-Marker ("loop exited (code $code) BEFORE the daily stop -- " +
                  "restart $restarts/$MaxRestarts in ${BackoffSeconds}s")
    Start-Sleep -Seconds $BackoffSeconds
}

Write-Marker "supervisor done after $restarts restart(s)"
