@echo off
echo ======================================================================
echo           Starting AquaWatch Deep Learning Flood Monitoring System
echo ======================================================================
echo.

cd /d "%~dp0\aquawatch"

echo [1/2] Starting FastAPI backend on http://127.0.0.1:8000 ...
start "AquaWatch Backend" cmd /k "py -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000"

timeout /t 2 /nobreak > nul

echo [2/2] Starting Vite Frontend on http://127.0.0.1:5173 ...
cd frontend
start "AquaWatch Frontend" cmd /k "npm run dev"

timeout /t 2 /nobreak > nul
echo.
echo Launching AquaWatch Dashboard in your browser...
start http://127.0.0.1:5173

echo.
echo ======================================================================
echo AquaWatch is now running!
echo - Website URL: http://127.0.0.1:5173
echo - API Docs:    http://127.0.0.1:8000/docs
echo Close the two command prompt windows when you want to stop the servers.
echo ======================================================================
