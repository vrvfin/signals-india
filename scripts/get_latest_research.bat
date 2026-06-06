@echo off
title Daily Research Summariser (Workflow A)
cd /d D:\EMA_Screener\claude\signals-india

set PYTHON=C:\Users\vaido\.conda\envs\signals-india\python.exe
set SYNTH_DIR=D:\EMA_Screener\Reports\signals-india\company_synthesis

echo.
echo  [1/4] Summarising intake folder (source PDFs stay local)...
echo.
"%PYTHON%" scripts/daily_research_summary.py
if errorlevel 1 (
    echo.
    echo  [ERROR] Summariser failed - check output above
    pause
    exit /b 1
)

echo.
echo  [2/4] Refreshing per-company synthesis on Drive (not opened here)...
echo.
"%PYTHON%" scripts/synthesise_company_docs.py --all-new --upload --queue --no-open --outdir "%SYNTH_DIR%"

echo.
echo  [3/4] Building Daily Focus note (cross-report triage)...
echo.
"%PYTHON%" scripts/daily_focus.py

echo.
echo  [4/4] Fetching and opening today's digest...
echo.
"%PYTHON%" scripts/fetch_latest_research.py

echo.
echo  Done. Opened: Daily Focus note + today's digest.
echo  Per-company synthesis_latest.md refreshed on Drive (open via synthesise_company.bat).
echo.
pause
