@echo off
title Fetch Latest Concall Digest
cd /d D:\EMA_Screener\claude\signals-india

echo.
echo  Fetching latest daily concall digest from Drive...
echo  Tables will be fixed for Obsidian rendering.
echo.

conda run -n signals-india python scripts/fetch_latest_concall.py

if errorlevel 1 (
    echo.
    echo  [ERROR] Something went wrong - check output above
    pause
) else (
    echo.
    echo  Done. File saved to OUTPUT_DIR and opened in Obsidian.
)
