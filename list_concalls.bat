@echo off
title List Available Concall Digests
cd /d D:\EMA_Screener\claude\signals-india

set PYTHON=C:\Users\vaido\.conda\envs\signals-india\python.exe

echo.
echo  Available concall digest files on Drive:
echo.

"%PYTHON%" scripts/fetch_latest_concall.py --list

echo.
set /p CHOICE="Enter date to fetch (e.g. 29may2026), or press Enter to exit: "
if "%CHOICE%"=="" goto end

"%PYTHON%" scripts/fetch_latest_concall.py --date %CHOICE%

:end
