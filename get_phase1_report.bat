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
echo  Generates a styled HTML report with:
echo    Section 1: Conviction Signals (stocks with >= 2 strategies)
echo               Sorted by strategy count, with RS rank + features
echo    Section 2: Portfolio page with signal overlay
echo               All holdings ranked by 3M return
echo.
echo  Opens in browser automatically.
echo  To save as PDF: Ctrl+P in browser -> Save as PDF (use Landscape)
echo.
echo  FILES WILL BE SAVED TO:
echo    %SAVE_DIR%
echo.
echo  ============================================================
echo.
echo  Options:
echo    1) Standard report (min 2 strategies)
echo    2) Strict report   (min 3 strategies)
echo    3) Signals only    (no portfolio section)
echo.
set /p CHOICE="  Choose (1/2/3) or press Enter for default (1): "

if "%CHOICE%"=="2" (
    "%PYTHON%" scripts/fetch_phase1_report.py --min-strats 3
) else if "%CHOICE%"=="3" (
    "%PYTHON%" scripts/fetch_phase1_report.py --no-portfolio
) else (
    "%PYTHON%" scripts/fetch_phase1_report.py
)

echo.
echo  ============================================================
echo  Report saved to: %SAVE_DIR%
echo  Tip: Ctrl+P in browser -> Save as PDF
echo  ============================================================
echo.
pause
