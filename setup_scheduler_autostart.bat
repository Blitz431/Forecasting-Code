@echo off
title StockChart Scheduler - Autostart Setup

schtasks /create ^
  /tn "StockChart Scheduler" ^
  /tr "\"J:\Coding\StockChart\launch_scheduler.bat\"" ^
  /sc onlogon ^
  /ru "%USERNAME%" ^
  /it ^
  /delay 0000:30 ^
  /rl limited ^
  /f

if %ERRORLEVEL% neq 0 (
    echo.
    echo  [ERROR] Could not register the task. See message above.
) else (
    echo.
    echo  Registered: "StockChart Scheduler" runs at every logon.
)
pause
