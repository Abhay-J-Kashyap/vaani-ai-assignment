# Real drain at 18:05 local (author approved, 2026-11-12): after the 20% reserve is fully released at 18:00.
# Waits, dry-runs, then drains the drafter's daily budget and sends 20 requests labelled daily_exhausted.
$py = ".\.venv\Scripts\python.exe"
$start = (Get-Date).Date.AddHours(18).AddMinutes(5)
if ((Get-Date) -lt $start) { Start-Sleep -Seconds ([int]($start - (Get-Date)).TotalSeconds) }
Write-Output "=== drain dry run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
& $py -m load.run_load drain --source real --dry-run 2>&1 | Where-Object { "$_" -notmatch 'NativeCommandError|CategoryInfo|^At |^\s*\+' }
Write-Output "=== drain start $(Get-Date -Format HH:mm:ss)"
& $py -m load.run_load drain --source real --confirm-real 2>&1 | Where-Object { "$_" -notmatch 'NativeCommandError|CategoryInfo|^At |^\s*\+' }
Write-Output "=== done $(Get-Date -Format HH:mm:ss)"
