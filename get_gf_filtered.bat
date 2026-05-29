@echo off
title GF Guidance Filter — Find companies by guidance criteria
cd /d D:\EMA_Screener\claude\signals-india

set PYTHON=C:\Users\vaido\.conda\envs\signals-india\python.exe

echo.
echo  ============================================================
echo   GF Guidance Filter
echo  ============================================================
echo.
echo  This tool lets you:
echo    1. Pick a metric  (PAT / Revenue / EBITDA / Margin / etc.)
echo    2. Pick a horizon (FY27 = next year, FY28 = 2yr forward)
echo    3. Set a minimum growth %% threshold (e.g. 40)
echo    4. See all matching companies with their guidance values
echo    5. Download company_page.md for each match (Table A + GF1-GF4)
echo    6. Open all in Obsidian automatically
echo.
echo  Example use case:
echo    "Which companies guided PAT growth above 40%% for FY27?"
echo.
echo  ============================================================
echo.

"%PYTHON%" scripts/fetch_gf_filtered.py

if errorlevel 1 (
    echo.
    echo  [ERROR] Check output above.
    pause
)
