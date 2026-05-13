@echo off
title Elena - QQ Bot
cd /d "D:\code\QQ chat"

echo.
echo ========================================
echo     Elena QQ Bot Launcher
echo ========================================
echo.

:: 1. NapCat
echo [1/2] Starting NapCat...
start "NapCat" /min cmd /c "cd /d D:\NapCat\NapCat.Shell.Windows.Node && node.exe ./index.js"
echo        NapCat started in background
echo.

:: 2. Bot
echo [2/2] Starting Bot...
start "ElenaBot" cmd /c "cd /d D:\code\QQ chat && venv\Scripts\python bot.py"
echo        Bot window opened
echo.

:: 3. Health check
echo        Checking if Bot is ready (max 30s)...

powershell -NoProfile -Command ^
"for ($i=0; $i -lt 30; $i++) { ^
  Start-Sleep 1; ^
  try { ^
    Invoke-WebRequest -Uri 'http://127.0.0.1:18080' -TimeoutSec 2 -UseBasicParsing ^| Out-Null; ^
    exit 0 ^
  } catch { ^
    if ($_.Exception.Response -ne $null) { exit 0 } ^
  } ^
} ^
exit 1"

if %errorlevel% equ 0 (
    goto success
) else (
    goto fail
)

:fail
echo.
echo ========================================
echo    [FAIL] Bot did not start in time.
echo.
echo    Please check:
echo    1. .env API Key is correct
echo    2. Port 18080 is not in use
echo    3. Bot window for error messages
echo ========================================
echo.
pause
exit

:success
echo.
echo ========================================
echo    [OK] Elena Bot is running!
echo.
echo    Bot address: 127.0.0.1:18080
echo    Check system tray for NapCat icon
echo    Send a private message in QQ to chat
echo ========================================
echo.
echo    You can close this window safely.
echo    Bot and NapCat run independently.
echo.
pause
exit
