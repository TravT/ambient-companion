@echo off
rem ==============================================================================
rem Ambient Multimodal Companion: Windows client setup (idempotent, no secrets)
rem Installs a private venv with the CLI dependencies and checks the homelab
rem service. The companion runs on the homelab; Windows is a client only.
rem Set AMBIENT_API_TOKEN in your user environment (never in this file).
rem ==============================================================================
setlocal
set "AMBIENT_URL=%AMBIENT_URL%"
if "%AMBIENT_URL%"=="" set "AMBIENT_URL=http://ambient.home.arpa"
set "VENV=%LOCALAPPDATA%\ambient-companion\venv"

echo ======================================================
echo Ambient Companion: Windows Client Setup
echo ======================================================

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

echo [1/3] Creating or reusing virtual environment at %VENV% ...
if not exist "%VENV%\Scripts\python.exe" python -m venv "%VENV%"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Could not create the virtual environment.
    pause
    exit /b 1
)

echo [2/3] Installing client dependencies...
"%VENV%\Scripts\python.exe" -m pip install --quiet --upgrade pip
"%VENV%\Scripts\python.exe" -m pip install --quiet -r "%~dp0requirements.txt"

echo [3/3] Checking the homelab service at %AMBIENT_URL% ...
curl -s -m 5 "%AMBIENT_URL%/health" >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [OK] ambient-companion is reachable.
) else (
    echo [WARN] %AMBIENT_URL%/health not reached. Connect to Tailscale or the home LAN,
    echo        and check that the Nomad job ambient-companion is running.
)

if "%AMBIENT_API_TOKEN%"=="" (
    echo [WARN] AMBIENT_API_TOKEN is not set. POST /mcp needs it as a Bearer token.
)

echo.
echo ======================================================
echo Windows Client Setup Complete!
echo Tool calls: POST %AMBIENT_URL%/mcp  with  Authorization: Bearer %%AMBIENT_API_TOKEN%%
echo ======================================================
pause
endlocal
