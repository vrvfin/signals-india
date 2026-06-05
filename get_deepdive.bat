@echo off
title Get Deep Dive Report
cd /d D:\EMA_Screener\claude\signals-india
set PYTHON=C:\Users\vaido\.conda\envs\signals-india\python.exe

echo.
echo ============================================================
echo   GET DEEP DIVE REPORT — fetch existing report from Drive
echo ============================================================
echo.
echo Enter company name, NSE/BSE symbol, or ISIN.
echo (Report must already exist — use run_deepdive.bat to generate one)
echo.
set /p COMPANY="Company: "

if "%COMPANY%"=="" (
    echo No input provided. Exiting.
    pause
    exit /b 1
)

echo.
echo Fetching report from Drive...
echo.
"%PYTHON%" scripts\fetch_deepdive.py "%COMPANY%"
if errorlevel 1 (
    echo.
    echo Could not fetch report. It may not exist yet — run run_deepdive.bat first.
    pause
    exit /b 1
)

echo.
pause
