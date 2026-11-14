<#
.SYNOPSIS
  Register (or remove) a Windows Task Scheduler entry that runs the Breeze 9:00 AM desk job.

.DESCRIPTION
  Every day at 09:00 local time the task runs `web_server.py --daily-run`: settle finished
  games first, then refresh stats and scan NBA + NFL (game lines and player props). It
  uses the same lock file as the web desk, so it never overlaps a run the desk started.
  The mode (live or mock) follows the desk's toggle unless -Mock or -Live is given.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\register_breeze_9am.ps1
  powershell -ExecutionPolicy Bypass -File scripts\register_breeze_9am.ps1 -Mock
  powershell -ExecutionPolicy Bypass -File scripts\register_breeze_9am.ps1 -Unregister
#>
param(
  [string]$TaskName = "Breeze Bets 9AM Desk Run",
  [string]$At = "09:00",
  [switch]$Mock,
  [switch]$Live,
  [switch]$Unregister
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

if ($Unregister) {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
  Write-Host "Removed scheduled task '$TaskName'."
  return
}
if ($Mock -and $Live) { throw "Pass -Mock or -Live, not both." }

$Python = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = (Get-Command python).Source }

$ArgList = "`"$(Join-Path $Root 'web_server.py')`" --daily-run"
if ($Mock) { $ArgList += " --mock" }
if ($Live) { $ArgList += " --live" }

$Action = New-ScheduledTaskAction -Execute $Python -Argument $ArgList -WorkingDirectory $Root
$Trigger = New-ScheduledTaskTrigger -Daily -At $At
# StartWhenAvailable: if the PC was asleep at 9:00, run as soon as it wakes.
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
  -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
  -Description "Breeze Bets: settle finished games, refresh stats, scan NBA + NFL." -Force | Out-Null

Write-Host "Registered '$TaskName': daily at $At -> $Python $ArgList"
Write-Host "Run it now with: Start-ScheduledTask -TaskName '$TaskName'"
