@echo off
title Fetch Latest Concall Digest
cd /d D:\EMA_Screener\claude\signals-india

set PYTHON=C:\Users\vaido\.conda\envs\signals-india\python.exe

echo.
echo  Fetching latest daily concall digest from Drive...
echo  Tables will be fixed for Obsidian rendering.
echo.

"%PYTHON%" scripts/fetch_latest_concall.py

if errorlevel 1 (
    echo.
    echo  [ERROR] Something went wrong - check output above
    pause
) else (
    echo.
    echo  Done. File saved and opened in Obsidian.
)
