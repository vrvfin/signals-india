@echo off
title List Available Concall Digests
cd /d D:\EMA_Screener\claude\signals-india

call conda activate signals-india 2>nul
if errorlevel 1 ( call activate signals-india 2>nul )

echo.
python scripts/fetch_latest_concall.py --list
echo.

set /p CHOICE="Enter number or date (e.g. 28may2026) to fetch, or press Enter to exit: "
if "%CHOICE%"=="" goto end

:: If numeric, map to date — user can type the date fragment directly
python scripts/fetch_latest_concall.py --date %CHOICE%

:end
