<#
Stop Stakey's trading loop by killing its python tree.

Run daily at 23:00 by the KalshiLoopStop scheduled task, and usable by hand as
the safe way to stop the loop (e.g. to pick up code changes).

WHAT IT DELIBERATELY DOES NOT KILL
----------------------------------
Only the python tree. The supervisor (run_loop_supervisor.ps1), the bat and
wscript are left alive ON PURPOSE:
  * the supervisor sees python exit, checks the clock, and declines to restart
    once the daily stop has passed;
  * the bat then runs its exit block -- the `paper loop exited` marker and the
    log rotation, which is the ONE moment nothing holds the log open;
  * wscript returns, so the scheduled-task INSTANCE completes, which is what
    stops the next morning's run stacking a second loop on a survivor.
Killing the whole tree (which is what `schtasks /End` does) skips all three.

Killing from the .venv LAUNCHER with /T is the safe direction: it takes the
anaconda worker child down with it. Killing the worker alone would leave the
launcher orphaned. See CLAUDE.md for why the loop is always two processes.
#>
[CmdletBinding()]
param(
    [string] $Log = 'logs\paper_loop.log',
    [switch] $WhatIfOnly,

    # Kill the supervisor too, so the loop stays down.
    #
    # Without this, running the script BEFORE the daily stop does not stop the
    # loop at all -- it bounces it: the supervisor sees python exit before
    # 23:00 and restarts it after its backoff (verified live, 11:39:09 kill ->
    # 11:39:40 restart). That is exactly right for the 23:00 task and for
    # picking up a code change, and exactly wrong if you actually want the loop
    # down during the day.
    [switch] $NoRestart
)

$ErrorActionPreference = 'Continue'

# The scheduled task runs this with no working directory, so relative paths
# would resolve against whatever cwd it inherits. Anchor on the repo instead.
# It lives in scripts/, so the repo root is its parent.
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
if (-not [IO.Path]::IsPathRooted($Log)) { $Log = Join-Path $repo $Log }

function Write-Marker([string] $Message) {
    $line = "===== stop_loop $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message ====="
    Write-Host $line
    try { Add-Content -Path $Log -Value $line -Encoding UTF8 } catch { }
}

function Get-LoopProcesses {
    @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*cli.py*loop*' })
}

$all = Get-LoopProcesses
if ($all.Count -eq 0) {
    Write-Marker 'no loop running -- nothing to stop'
    exit 0
}

# Prefer the .venv launcher: /T takes its anaconda worker child with it.
$launchers = @($all | Where-Object { $_.CommandLine -like '*\.venv\Scripts\python.exe*' })
$targets = if ($launchers.Count -gt 0) { $launchers } else { $all }

if ($WhatIfOnly) {
    $extra = if ($NoRestart) { ' + the supervisor' } else { '' }
    Write-Marker "would kill: $(($targets.ProcessId) -join ', ')$extra"
    exit 0
}

foreach ($p in $targets) {
    Write-Marker "killing loop pid $($p.ProcessId)"
    taskkill /PID $p.ProcessId /T /F 2>$null | Out-Null
}

Start-Sleep -Seconds 5

# A worker can outlive its launcher if the tree was already broken; sweep it.
$left = Get-LoopProcesses
if ($left.Count -gt 0) {
    foreach ($p in $left) {
        Write-Marker "sweeping leftover loop pid $($p.ProcessId)"
        taskkill /PID $p.ProcessId /T /F 2>$null | Out-Null
    }
    Start-Sleep -Seconds 3
    $left = Get-LoopProcesses
}

if ($left.Count -eq 0) {
    if ($NoRestart) {
        # Kill the supervisor, not the whole tree: the bat then runs its exit
        # block (marker + log rotation) and wscript returns, so the scheduled
        # task INSTANCE completes. Killing wscript instead would skip all of
        # that, which is the `schtasks /End` mistake this whole design avoids.
        # Python is already down, so there is no window for a restart.
        $sup = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -like '*run_loop_supervisor*' })
        foreach ($s in $sup) {
            Write-Marker "killing supervisor pid $($s.ProcessId) (-NoRestart)"
            taskkill /PID $s.ProcessId /T /F 2>$null | Out-Null
        }
        Write-Marker 'loop stopped and will NOT be restarted'
    } else {
        Write-Marker 'loop stopped (supervisor will restart it before the daily stop)'
    }
    exit 0
}

Write-Marker "FAILED -- still running: $(($left.ProcessId) -join ', ')"
exit 1
