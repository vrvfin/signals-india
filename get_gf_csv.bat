@echo off
title Download GF Tables as CSV (for Excel)
cd /d D:\EMA_Screener\claude\signals-india

set PYTHON=C:\Users\vaido\.conda\envs\signals-india\python.exe
set SAVE_DIR=D:\EMA_Screener\Reports\signals-india\gf_tables_csv

echo.
echo  ============================================================
echo   GF Tables CSV Downloader  (Option C)
echo  ============================================================
echo.
echo  Downloads all GF tables as CSV files ready for Excel:
echo    guidance_tracker.csv        - Table A structured guidance
echo    gf1_guidance_statements.csv - Raw forward-looking statements
echo    gf2_historical_guidance.csv - Past guidance vs actuals
echo    gf3_operational_visibility.csv
echo    gf4_quality_flags.csv
echo.
echo  FILES WILL BE SAVED TO:
echo    %SAVE_DIR%
echo.
echo  Folder will open in Explorer automatically after download.
echo  ============================================================
echo.

"%PYTHON%" scripts/fetch_gf_csv.py

echo.
echo  Files saved to: %SAVE_DIR%
echo.
pause
