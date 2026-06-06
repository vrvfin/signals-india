@echo off
REM run_daily_research.bat  —  UNATTENDED daily run for Windows Task Scheduler.
REM Summarises the intake folder only (no viewer, no pause). Logs each run.
REM Manual/interactive use -> use get_latest_research.bat instead.

setlocal
set PROJ=D:\EMA_Screener\claude\signals-india
set LOGDIR=%PROJ%\logs\research
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set LOG=%LOGDIR%\daily_%date:~-4%%date:~3,2%%date:~0,2%.log

cd /d "%PROJ%"
set PYTHON=C:\Users\vaido\.conda\envs\signals-india\python.exe

echo ==== run %date% %time% ==== >> "%LOG%"
"%PYTHON%" scripts\daily_research_summary.py >> "%LOG%" 2>&1
echo ==== summariser exit %errorlevel% ==== >> "%LOG%"

REM Auto-synthesise new ISINs and refresh synthesis_latest.md on Drive.
REM Also enqueues those ISINs for the next OT7 deep dive run.
"%PYTHON%" scripts\synthesise_company_docs.py --all-new --upload --queue --no-open >> "%LOG%" 2>&1
echo ==== synthesis exit %errorlevel% ==== >> "%LOG%"

REM Cross-report Daily Focus triage note -> Drive (no browser in unattended mode).
"%PYTHON%" scripts\daily_focus.py --no-open >> "%LOG%" 2>&1
echo ==== focus exit %errorlevel% ==== >> "%LOG%"
endlocal
