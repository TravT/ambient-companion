@echo off
rem ==============================================================================
rem Ambient Multimodal Companion: Windows Client Setup Script
rem Connects Windows environments to the Homelab Ambient Companion
rem ==============================================================================
echo ======================================================
echo Ambient Companion: Windows Client Setup
echo ======================================================

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

echo [1/2] Installing required client dependencies...
pip install --upgrade pillow requests

echo [2/2] Checking connectivity to Homelab Ambient Endpoint...
curl -s -m 3 http://192.168.0.48:8085/health >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [OK] Homelab llama-server is reachable on 192.168.0.48:8085.
) else (
    echo [WARN] Homelab llama-server (192.168.0.48:8085) not reached directly.
    echo        Ensure you are connected to Tailscale.
)

echo.
echo ======================================================
echo Windows Client Setup Complete!
echo ======================================================
pause
