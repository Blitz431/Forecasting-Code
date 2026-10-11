@echo off
title StockChart Scheduler - Remove Autostart
schtasks /delete /tn "StockChart Scheduler" /f
pause
