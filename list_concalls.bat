@echo off
title List Available Concall Digests
cd /d D:\EMA_Screener\claude\signals-india

echo.
echo  Available concall digest files on Drive:
echo.

conda run -n signals-india python scripts/fetch_latest_concall.py --list

echo.
set /p CHOICE="Enter date to fetch (e.g. 29may2026), or press Enter to exit: "
if "%CHOICE%"=="" goto end

conda run -n signals-india python scripts/fetch_latest_concall.py --date %CHOICE%

:end
