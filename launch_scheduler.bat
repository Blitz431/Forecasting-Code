@echo off
title StockChart Scheduler
cd /d "J:\Coding\StockChart"

echo.
echo  Starting StockChart Scheduler...
echo  Daily jobs run on a UTC schedule; this PC must be on and awake.
echo  Close this window or press Ctrl-C to stop.
echo.

python cli/scheduler.py --start

if %ERRORLEVEL% neq 0 (
    echo.
    echo  [ERROR] Scheduler exited unexpectedly. See message above.
    pause
)
