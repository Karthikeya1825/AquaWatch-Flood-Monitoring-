@echo off
title AquaWatch - Public Online Sharing
cd /d "%~dp0"

echo ======================================================================
echo             AquaWatch - Instant Online Public Access
echo ======================================================================
echo.
echo Make sure AquaWatch is already running (via start_aquawatch.bat).
echo Generating a secure, temporary HTTPS link you can open on your phone
echo or share with your teacher / evaluator anywhere...
echo.
echo If it asks for an "Endpoint IP", your public IP is shown below.
echo.

powershell -Command "(Invoke-WebRequest -Uri 'https://api.ipify.org' -UseBasicParsing).Content" 2>nul
echo.
echo ----------------------------------------------------------------------
echo Please wait 5 seconds while your secure public URL connects...
echo When the URL appears below (e.g. https://xxxx.loca.lt), copy and open it!
echo ----------------------------------------------------------------------
echo.

npx --yes localtunnel --port 5173
pause

