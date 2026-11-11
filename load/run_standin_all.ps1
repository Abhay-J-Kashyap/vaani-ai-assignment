# The stand-in rows of Table 2a/2b plus the heading-6 retry experiment, one after another (no Groq calls).
# Run from the repo root: powershell -File load\run_standin_all.ps1
$py = ".\.venv\Scripts\python.exe"
foreach ($s in @("overload", "tpm_half", "daily_exhausted", "retry_naive", "retry_policy")) {
    Write-Output "=== $s $(Get-Date -Format HH:mm:ss)"
    & $py -m load.run_load $s 2>&1 | Where-Object { "$_" -notmatch 'NativeCommandError|CategoryInfo|^At |^\s*\+' }
}
Write-Output "=== all done $(Get-Date -Format HH:mm:ss)"
