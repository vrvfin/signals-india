@echo off
title Phase 1 Daily Report (Signals + Portfolio)
cd /d D:\EMA_Screener\claude\signals-india

set PYTHON=C:\Users\vaido\.conda\envs\signals-india\python.exe
set SAVE_DIR=D:\EMA_Screener\Reports\signals-india\phase1_reports

echo.
echo  ============================================================
echo   Phase 1 Daily Report Generator
echo  ============================================================
echo.
echo  Section 1: Conviction Signals  (stocks with >=2 strategies)
echo  Section 2: Portfolio            (holdings with signal overlay)
echo.
echo  Charts: collapsible  -- click triangle to expand any chart.
echo  All charts embed interactively (hover/zoom/pan).
echo  Requires internet to load Plotly.js (CDN, loads once).
echo.
echo  FILES WILL BE SAVED TO:
echo    %SAVE_DIR%
echo  To save as PDF: Ctrl+P - Save as PDF - use Landscape
echo  ============================================================
echo.
echo  Options (approx time including chart downloads from Drive):
echo.
echo    1) Tables only, no charts             (~15 sec)
echo    2) Top 10 charts + portfolio charts   (~3-4 min) *recommended*
echo    3) Top 20 charts + portfolio charts   (~5-7 min)
echo    4) Top 50 charts + portfolio charts   (~12-15 min)
echo    5) Top 100 charts + portfolio charts  (~25-30 min)
echo    6) Top 200 charts + portfolio charts  (~50-60 min)
echo    7) Top 300 charts + portfolio charts  (~75-90 min)
echo    8) Strict (>=3 strats) + top 10 charts
echo    9) Signals only, no portfolio
echo.
set /p CHOICE="  Choose (1-9) or Enter for default (2): "

if "%CHOICE%"=="1" (
    "%PYTHON%" scripts/fetch_phase1_report.py
) else if "%CHOICE%"=="3" (
    "%PYTHON%" scripts/fetch_phase1_report.py --with-charts --max-charts 20
) else if "%CHOICE%"=="4" (
    "%PYTHON%" scripts/fetch_phase1_report.py --with-charts --max-charts 50
) else if "%CHOICE%"=="5" (
    "%PYTHON%" scripts/fetch_phase1_report.py --with-charts --max-charts 100
) else if "%CHOICE%"=="6" (
    "%PYTHON%" scripts/fetch_phase1_report.py --with-charts --max-charts 200
) else if "%CHOICE%"=="7" (
    "%PYTHON%" scripts/fetch_phase1_report.py --with-charts --max-charts 300
) else if "%CHOICE%"=="8" (
    "%PYTHON%" scripts/fetch_phase1_report.py --min-strats 3 --with-charts --max-charts 10
) else if "%CHOICE%"=="9" (
    "%PYTHON%" scripts/fetch_phase1_report.py --no-portfolio
) else (
    "%PYTHON%" scripts/fetch_phase1_report.py --with-charts --max-charts 10
)

echo.
echo  ============================================================
echo  Report saved to: %SAVE_DIR%
echo  Tip: Ctrl+P - Save as PDF - Landscape orientation
echo  ============================================================
echo.
pause
