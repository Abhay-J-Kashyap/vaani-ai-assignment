# Render WRITEUP.md to WRITEUP.pdf (A4, compact print layout) to check the brief's 2-page limit.
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\render_writeup.ps1 [-In other.md] [-Out other.pdf]
# Needs pandoc and Edge (or Chrome). Prints the page count.
param([string]$In = "", [string]$Out = "")
$root = Split-Path -Parent $PSScriptRoot
if (-not $In) { $In = Join-Path $root "WRITEUP.md" }
if (-not $Out) { $Out = Join-Path $root "WRITEUP.pdf" }
$html = Join-Path $env:TEMP "relay_writeup.html"
pandoc $In -f gfm -t html5 -s --metadata pagetitle="The Relay: write-up" `
    --css (Join-Path $root "tools\writeup_print.css") --embed-resources -o $html
$browser = @("C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
             "C:\Program Files\Microsoft\Edge\Application\msedge.exe",
             "C:\Program Files\Google\Chrome\Application\chrome.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (Test-Path $Out) { Remove-Item $Out }
& $browser --headless=new --disable-gpu --no-pdf-header-footer "--print-to-pdf=$Out" $html 2>$null | Out-Null
for ($i = 0; $i -lt 20 -and -not (Test-Path $Out); $i++) { Start-Sleep -Milliseconds 500 }
$bytes = [System.IO.File]::ReadAllText($Out, [System.Text.Encoding]::GetEncoding(28591))
$pages = ([regex]::Matches($bytes, "/Type\s*/Page[^s]")).Count
Write-Output "$(Split-Path -Leaf $Out): $pages page(s)"
