@echo off
title Deep Dive — Company Research
cd /d D:\EMA_Screener\claude\signals-india
set PYTHON=C:\Users\vaido\.conda\envs\signals-india\python.exe

echo.
echo ============================================================
echo   DEEP DIVE — Forensic Company Research
echo ============================================================
echo.
echo Enter company name, NSE/BSE symbol, or ISIN.
echo Examples: TCS  /  VENUSREM  /  INE467B01029  /  Venus Remedies
echo.
set /p COMPANY="Company: "

if "%COMPANY%"=="" (
    echo No input provided. Exiting.
    pause
    exit /b 1
)

echo.
echo Resolving company from universe...
echo.

REM Use --interactive so user can pick if partial name matches multiple companies
REM We do a dry-resolve first to confirm before choosing run mode
"%PYTHON%" scripts\company_deep_report.py --resolve-only --names "%COMPANY%" --interactive
if errorlevel 1 (
    echo.
    echo Could not resolve company. Try a different name, symbol, or ISIN.
    pause
    exit /b 1
)

echo.
echo ------------------------------------------------------------
echo   Run mode:
echo     1. Run NOW  - local Gemini call (~3-4 min)
echo                   Report saved to Drive + opened locally.
echo                   Laptop must stay on until complete.
echo     2. Queue    - CI runs at 08:00 IST tomorrow.
echo                   You get an email with report + Drive link.
echo ------------------------------------------------------------
echo.
set /p MODE="Enter 1 or 2: "

if "%MODE%"=="1" (
    echo.
    echo Running deep dive locally — please wait...
    echo.
    "%PYTHON%" scripts\company_deep_report.py --names "%COMPANY%" --open --interactive
    if errorlevel 1 (
        echo.
        echo Deep dive failed. Check output above for details.
        pause
        exit /b 1
    )
    echo.
    echo Deep dive complete. Report opened.
) else if "%MODE%"=="2" (
    echo.
    echo Queuing company for CI run...
    "%PYTHON%" scripts\company_deep_report.py --add "%COMPANY%" --interactive
    if errorlevel 1 (
        echo.
        echo Failed to queue. Check output above for details.
        pause
        exit /b 1
    )
    echo.
    echo Queued. CI will run at 08:00 IST and email you when done.
) else (
    echo Invalid choice. Please enter 1 or 2.
)

echo.
pause
