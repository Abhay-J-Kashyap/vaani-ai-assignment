# Real steady then real burst (author approved, 2026-11-12). Run from the repo root.
$py = ".\.venv\Scripts\python.exe"
foreach ($s in @("steady", "burst")) {
    Write-Output "=== $s dry run $(Get-Date -Format HH:mm:ss)"
    & $py -m load.run_load $s --source real --dry-run 2>&1 | Where-Object { "$_" -notmatch 'NativeCommandError|CategoryInfo|^At |^\s*\+' }
    Write-Output "=== $s start $(Get-Date -Format HH:mm:ss)"
    & $py -m load.run_load $s --source real --confirm-real 2>&1 | Where-Object { "$_" -notmatch 'NativeCommandError|CategoryInfo|^At |^\s*\+' }
}
Write-Output "=== done $(Get-Date -Format HH:mm:ss)"
