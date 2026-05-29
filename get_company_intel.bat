@echo off
title Fetch Company Intel (Table A / GF1-GF4)
cd /d D:\EMA_Screener\claude\signals-india

echo.
echo  Downloads company_page.md (Table A, GF1-GF4, summaries) from Drive,
echo  fixes table formatting and opens in Obsidian.
echo.
echo  Use the ISIN (e.g. INE002A01018) for best results.
echo  Symbol lookup (e.g. RELIANCE) works if the folder is named by symbol.
echo.
set /p SYMBOL="Enter NSE symbol or ISIN: "

if "%SYMBOL%"=="" (
    echo  No input - exiting.
    pause
    exit /b
)

conda run -n signals-india python scripts/fetch_company_intel.py --symbol %SYMBOL%

if errorlevel 1 (
    echo.
    echo  [ERROR] Check output above. Tip: try ISIN instead of symbol.
    pause
)
