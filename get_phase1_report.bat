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
echo  Generates a styled HTML report:
echo    Section 1: Conviction Signals (stocks with >= 2 strategies)
echo    Section 2: Portfolio with signal + feature overlay
echo.
echo  FILES WILL BE SAVED TO:
echo    %SAVE_DIR%
echo.
echo  To save as PDF: Ctrl+P in browser - Save as PDF (Landscape)
echo  ============================================================
echo.
echo  Options:
echo    1) Tables only              (fast, ~15 sec)
echo    2) Tables + charts top 10   (interactive Plotly, ~2-3 min)
echo    3) Tables + charts top 20   (more coverage, ~4-5 min)
echo    4) Strict signals (>=3 strats) + charts top 10
echo    5) Signals only, no portfolio
echo.
set /p CHOICE="  Choose (1-5) or Enter for default (1): "

if "%CHOICE%"=="2" (
    "%PYTHON%" scripts/fetch_phase1_report.py --with-charts --max-charts 10
) else if "%CHOICE%"=="3" (
    "%PYTHON%" scripts/fetch_phase1_report.py --with-charts --max-charts 20
) else if "%CHOICE%"=="4" (
    "%PYTHON%" scripts/fetch_phase1_report.py --min-strats 3 --with-charts --max-charts 10
) else if "%CHOICE%"=="5" (
    "%PYTHON%" scripts/fetch_phase1_report.py --no-portfolio
) else (
    "%PYTHON%" scripts/fetch_phase1_report.py
)

echo.
echo  ============================================================
echo  Report saved to: %SAVE_DIR%
echo  Tip: Ctrl+P in browser - Save as PDF - use Landscape
echo  ============================================================
echo.
pause
