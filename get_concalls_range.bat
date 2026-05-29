@echo off
title Fetch Concalls by Date Range
cd /d D:\EMA_Screener\claude\signals-india

set PYTHON=C:\Users\vaido\.conda\envs\signals-india\python.exe
set SAVE_DIR=D:\EMA_Screener\Reports\signals-india\concalls

echo.
echo  ============================================================
echo   Download concall digests for a date range
echo  ============================================================
echo.
echo  FILES WILL BE SAVED TO:
echo    %SAVE_DIR%
echo.
echo  All files will be fixed for Obsidian table rendering and
echo  opened automatically in Obsidian.
echo.
echo  DATE FORMAT: DDmmmYYYY  e.g.  20may2026  or  01jan2026
echo              Also accepted:    2026-05-20  or  20-may-2026
echo.

set /p FROM_DATE="  Start date (required):  "
if "%FROM_DATE%"=="" (
    echo  No start date entered - exiting.
    pause
    exit /b
)

echo.
echo  End date: press Enter to download everything up to TODAY
set /p TO_DATE="  End date   (or Enter for today): "
echo.

if "%TO_DATE%"=="" (
    echo  Fetching from %FROM_DATE% to today...
    echo.
    "%PYTHON%" scripts/fetch_concalls_range.py --from %FROM_DATE%
) else (
    echo  Fetching from %FROM_DATE% to %TO_DATE%...
    echo.
    "%PYTHON%" scripts/fetch_concalls_range.py --from %FROM_DATE% --to %TO_DATE%
)

echo.
echo  ============================================================
echo  Files saved to: %SAVE_DIR%
echo  ============================================================
echo.
pause
