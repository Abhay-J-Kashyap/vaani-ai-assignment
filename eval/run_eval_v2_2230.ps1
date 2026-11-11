# Table 3 rerun on the fixed pipeline (author approved 2026-11-11: dry run first, start around 22:30).
# Waits until 22:30 local, dry-runs, and starts only if the drafter's ledger level covers at least 80% of the
# projection; otherwise it stops and says so. The runner waits for budget before each request, so any shortfall
# just paces the last answers at the refill rate (~3 answers/hour) instead of failing them. Run from the repo root.
$py = ".\.venv\Scripts\python.exe"
$start = (Get-Date).Date.AddHours(22).AddMinutes(30)
if ((Get-Date) -lt $start) { Start-Sleep -Seconds ([int]($start - (Get-Date)).TotalSeconds) }
Write-Output "=== dry run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$dry = & $py -m eval.run_eval --dry-run --scenario eval_v2 2>&1 | Out-String
Write-Output $dry
$m = [regex]::Match($dry, 'gpt-oss-120b: projected ~([\d,]+) tokens; ledger level ~([\d,]+)')
$need = [int]($m.Groups[1].Value -replace ',', '')
$level = [int]($m.Groups[2].Value -replace ',', '')
if ($level -lt 0.8 * $need) {
    Write-Output "=== NOT STARTED: drafter level $level < 80% of projected $need. Re-run this script later."
    exit 1
}
Write-Output "=== starting: level $level vs projected $need (the rest refills during the run)"
& $py -m eval.run_eval --confirm-real --force --scenario eval_v2 2>&1
Write-Output "=== finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
