@echo off
title Download Quarterly Guidance (Table A + GF summaries)
cd /d D:\EMA_Screener\claude\signals-india

set PYTHON=C:\Users\vaido\.conda\envs\signals-india\python.exe
set SAVE_DIR=D:\EMA_Screener\Reports\signals-india\quarterly_guidance

echo.
echo  ============================================================
echo   Quarterly Guidance Downloader  (Option A)
echo  ============================================================
echo.
echo  Downloads quarterly .md files containing Table A + GF1-GF4
echo  summaries for ALL companies in that quarter.
echo.
echo  FILES WILL BE SAVED TO:
echo    %SAVE_DIR%
echo.
echo  ============================================================
echo.

"%PYTHON%" scripts/fetch_quarterly_guidance.py

echo.
echo  Files saved to: %SAVE_DIR%
echo.
pause
