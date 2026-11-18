@echo off
title Push AquaWatch to GitHub
cd /d "%~dp0"

:: Ensure x64 MinGit and GCM are prioritized in PATH over broken ARM64 Git
set "PATH=%~dp0tools\git\cmd;%~dp0tools\gcm;%PATH%"
set "GIT_EXEC_PATH=%~dp0tools\git\mingw64\libexec\git-core"

echo ======================================================================
echo                  Pushing AquaWatch to GitHub
echo ======================================================================
echo Target Repo: https://github.com/Karthikeya1825/AquaWatch-Flood-Monitoring-.git
echo Branch:      main
echo.
echo If a GitHub browser login window pops up, click "Authorize git-credential-manager"
echo to complete your one-time sign in.
echo.
echo Uploading files now, please wait...
echo ----------------------------------------------------------------------

git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================================
    echo [SUCCESS] AquaWatch has been successfully pushed to GitHub!
    echo Visit your repo at:
    echo https://github.com/Karthikeya1825/AquaWatch-Flood-Monitoring-
    echo ======================================================================
) else (
    echo.
    echo ======================================================================
    echo [NOTE] If you need to enter credentials or a Personal Access Token:
    echo 1. Generate a token at: https://github.com/settings/tokens
    echo 2. Check the "repo" scope.
    echo 3. Paste the token when prompted.
    echo ======================================================================
)

echo.
pause
