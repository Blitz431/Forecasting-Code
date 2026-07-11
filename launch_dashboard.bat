@echo off
title StockChart Dashboard
cd /d "J:\Coding\StockChart"

echo.
echo  Starting StockChart Dashboard...
echo  The browser will open automatically.
echo  Close this window to stop the server.
echo.

python -m streamlit run dashboard/app.py --server.headless false --browser.gatherUsageStats false

if %ERRORLEVEL% neq 0 (
    echo.
    echo  [ERROR] Dashboard failed to start. See message above.
    pause
)
