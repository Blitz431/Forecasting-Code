$ProjectRoot = "J:\Coding\StockChart"
$Python = "C:\Users\arnie\AppData\Local\Microsoft\WindowsApps\python.exe"
$LogDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$LogFile = Join-Path $LogDir ("trade_" + (Get-Date -Format "yyyyMMdd") + ".log")
Set-Location $ProjectRoot
Add-Content -Path $LogFile -Encoding utf8 -Value "=== Run at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
# Use cmd.exe to merge stdout+stderr — avoids PowerShell 5.1 wrapping native stderr as ErrorRecord.
cmd.exe /c "`"$Python`" cli\trade.py --mode paper >> `"$LogFile`" 2>&1"
exit $LASTEXITCODE
