@echo off
title Fetch Concalls by Date Range
cd /d D:\EMA_Screener\claude\signals-india

set PYTHON=C:\Users\vaido\.conda\envs\signals-india\python.exe

echo.
echo  Download concall digests for a date range.
echo  All files will be fixed for Obsidian and opened automatically.
echo.
echo  Date format: 20may2026  or  2026-05-20
echo.

set /p FROM_DATE="Start date (e.g. 20may2026): "
if "%FROM_DATE%"=="" (
    echo  No start date entered - exiting.
    pause
    exit /b
)

set /p TO_DATE="End date   (press Enter for today): "

if "%TO_DATE%"=="" (
    "%PYTHON%" scripts/fetch_concalls_range.py --from %FROM_DATE%
) else (
    "%PYTHON%" scripts/fetch_concalls_range.py --from %FROM_DATE% --to %TO_DATE%
)

if errorlevel 1 (
    echo.
    echo  [ERROR] Check output above.
    pause
)
