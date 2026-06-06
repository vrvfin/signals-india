@echo off
title Upload PF Holdings to Drive
cd /d D:\EMA_Screener\claude\signals-india

set PYTHON=C:\Users\vaido\.conda\envs\signals-india\python.exe

echo.
echo  Upload Holdings Statement to Drive (pf_tracking/)
echo  ---------------------------------------------------
echo.

"%PYTHON%" scripts\upload_pf_to_drive.py %*

echo.
pause
