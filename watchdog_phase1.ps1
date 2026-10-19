# Detached launcher and watchdog for the Phase 1 run (run_phase1.sh).
#   powershell -NoProfile -ExecutionPolicy Bypass -File watchdog_phase1.ps1 -Python <path to python.exe>
# Starts the driver as its own process (Git bash by explicit path: plain "bash" can resolve to the WSL launcher), checks it
# every -TickSec seconds, writes a timestamped HEARTBEAT line to results/phase1/heartbeat.log every -IntervalSec seconds, and
# writes an "ALERT" line the moment the driver exits before finishing. The run is resumable, so the driver is restarted (at most
# -MaxRestarts times). "ALERT STALLED" is written when no replicate has been added for three heartbeat intervals.
param(
    [Parameter(Mandatory = $true)][string]$Python,
    [int]$IntervalSec = 300,
    [int]$TickSec = 30,
    [int]$MaxRestarts = 5,
    [string]$Bash = "C:\Program Files\Git\bin\bash.exe",
    [string]$DriverLog = "..\phase1_driver.log"
)
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo
$hb = Join-Path $repo "results\phase1\heartbeat.log"
$env:PYTHON = $Python

function Stamp { (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss") }
function Rows {
    $parts = @(); $total = 0
    foreach ($f in Get-ChildItem (Join-Path $repo "results\phase1") -Filter "diag_*.csv" -ErrorAction SilentlyContinue | Sort-Object Name) {
        $n = @(Get-Content $f.FullName).Count - 1
        $parts += ("{0}={1}" -f $f.BaseName.Substring(5), $n); $total += $n
    }
    return @{ text = ($parts -join " "); total = $total }
}
function Start-Driver {
    return Start-Process -FilePath $Bash -ArgumentList "run_phase1.sh" -WorkingDirectory $repo -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $repo $DriverLog) -RedirectStandardError (Join-Path $repo ($DriverLog + ".err"))
}
function Is-Done { (Test-Path (Join-Path $repo $DriverLog)) -and (Select-String -Path (Join-Path $repo $DriverLog) -Pattern "PHASE1_P1_P4_DONE" -Quiet) }

$restarts = 0
$drv = Start-Driver
Add-Content $hb ("{0} START watchdog pid={1} driver_pid={2} heartbeat={3}s tick={4}s" -f (Stamp), $PID, $drv.Id, $IntervalSec, $TickSec)
$lastBeat = Get-Date; $last = -1; $still = 0
while ($true) {
    Start-Sleep -Seconds $TickSec
    if ($drv.HasExited) {
        $r = Rows
        if (Is-Done) { Add-Content $hb ("{0} DONE driver finished (PHASE1_P1_P4_DONE in log); total rows {1}" -f (Stamp), $r.total); break }
        Add-Content $hb ("{0} ALERT driver exited before completion (exit code {1}); rows {2}" -f (Stamp), $drv.ExitCode, $r.text)
        if ($restarts -ge $MaxRestarts) { Add-Content $hb ("{0} ALERT giving up after {1} restarts" -f (Stamp), $restarts); break }
        $restarts++
        $drv = Start-Driver
        Add-Content $hb ("{0} ALERT restarted driver (restart {1}/{2}) pid={3}" -f (Stamp), $restarts, $MaxRestarts, $drv.Id)
        $still = 0
    }
    if (((Get-Date) - $lastBeat).TotalSeconds -ge $IntervalSec) {
        $lastBeat = Get-Date
        $r = Rows
        $py = @(Get-Process -Name python -ErrorAction SilentlyContinue).Count
        Add-Content $hb ("{0} HEARTBEAT driver_alive={1} driver_pid={2} python_procs={3} rows: {4} total={5}" -f (Stamp), (-not $drv.HasExited), $drv.Id, $py, $r.text, $r.total)
        if ($r.total -eq $last) { $still++ } else { $still = 0 }
        if ($still -ge 3) { Add-Content $hb ("{0} ALERT STALLED no new replicate for {1} heartbeats (total {2})" -f (Stamp), $still, $r.total) }
        $last = $r.total
    }
}
