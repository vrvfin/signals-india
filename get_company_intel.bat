@echo off
title Fetch Company Intel (Table A / GF1-GF4)
cd /d D:\EMA_Screener\claude\signals-india

call conda activate signals-india 2>nul
if errorlevel 1 ( call activate signals-india 2>nul )

echo.
echo  This fetches the company_page.md for a symbol from Drive,
echo  fixes table formatting and opens in Obsidian.
echo.
set /p SYMBOL="Enter NSE symbol or ISIN (e.g. RELIANCE): "

if "%SYMBOL%"=="" (
    echo No input — exiting.
    pause
    exit /b
)

python scripts/fetch_company_intel.py --symbol %SYMBOL%

if errorlevel 1 (
    echo.
    echo  [ERROR] Check output above
    pause
)
