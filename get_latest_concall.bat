@echo off
title Fetch Latest Concall Digest
cd /d D:\EMA_Screener\claude\signals-india

:: Activate conda env
call conda activate signals-india 2>nul
if errorlevel 1 (
    call activate signals-india 2>nul
)

echo.
echo  Fetching latest concall digest from Drive...
echo  (Tables will be fixed for Obsidian rendering)
echo.
python scripts/fetch_latest_concall.py

if errorlevel 1 (
    echo.
    echo  [ERROR] Something went wrong - check output above
    pause
)
