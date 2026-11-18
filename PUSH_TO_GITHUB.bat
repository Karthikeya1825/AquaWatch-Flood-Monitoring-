@echo off
title Push AquaWatch to GitHub
cd /d "%~dp0"

:: Set local x64 MinGit and GCM in PATH
set "PATH=%~dp0tools\git\cmd;%~dp0tools\gcm;%PATH%"
set "GIT_EXEC_PATH=%~dp0tools\git\mingw64\libexec\git-core"

echo ======================================================================
echo                  Pushing AquaWatch to GitHub
echo ======================================================================
echo Target Repo: https://github.com/Karthikeya1825/AquaWatch-Flood-Monitoring-.git
echo Branch:      main
echo.
echo Choose how you would like to authenticate:
echo.
echo [1] Browser Sign-In (Default - Git Credential Manager will open GitHub)
echo [2] Paste a GitHub Personal Access Token (Instant 5-second push)
echo.
set /p CHOICE="Enter choice [1 or 2, default is 1]: "

if "%CHOICE%"=="2" goto TOKEN_PUSH

:BROWSER_PUSH
echo.
echo Initiating push via Browser Sign-in...
echo If your browser opens, click "Authorize git-credential-manager".
echo.
git push -u origin main
if %ERRORLEVEL% EQU 0 goto SUCCESS
echo.
echo Browser sign-in did not complete or was cancelled.
echo Let's try Personal Access Token instead!
echo.

:TOKEN_PUSH
echo ----------------------------------------------------------------------
echo How to get a token in 30 seconds:
echo 1. Open: https://github.com/settings/tokens/new
echo 2. Give it a note, check the "repo" box, and click "Generate token".
echo 3. Copy the token (starts with ghp_).
echo ----------------------------------------------------------------------
echo.
set /p GITHUB_TOKEN="Paste your GitHub Token here: "

if "%GITHUB_TOKEN%"=="" (
    echo No token entered. Exiting.
    pause
    exit /b 1
)

echo.
echo Pushing to GitHub using your token...
git push https://%GITHUB_TOKEN%@github.com/Karthikeya1825/AquaWatch-Flood-Monitoring-.git main
if %ERRORLEVEL% EQU 0 goto SUCCESS

echo.
echo [ERROR] Push failed. Please verify your token has 'repo' permissions.
pause
exit /b 1

:SUCCESS
echo.
echo ======================================================================
echo [SUCCESS] AquaWatch has been successfully uploaded to GitHub!
echo View your repository now at:
echo https://github.com/Karthikeya1825/AquaWatch-Flood-Monitoring-
echo ======================================================================
pause
