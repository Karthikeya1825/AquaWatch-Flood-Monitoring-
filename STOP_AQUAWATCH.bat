@echo off
title Stop AquaWatch Servers
cd /d "%~dp0"

echo ======================================================================
echo                  Stopping AquaWatch Servers
echo ======================================================================
echo.

taskkill /F /IM node.exe /T 2>nul
taskkill /F /FI "WINDOWTITLE eq AquaWatch Backend*" /T 2>nul
taskkill /F /FI "WINDOWTITLE eq AquaWatch Frontend*" /T 2>nul
powershell -Command "Get-Process python*, py* -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*uvicorn*' } | Stop-Process -Force" 2>nul

echo.
echo All AquaWatch background services and dev servers have been stopped.
echo ======================================================================
pause
