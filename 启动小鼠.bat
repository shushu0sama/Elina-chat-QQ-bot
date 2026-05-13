@echo off
chcp 65001 >nul
title 小鼠 - QQ Bot

echo.
echo ========================================
echo      小鼠 QQ Bot 启动中...
echo ========================================
echo.

:: 1. NapCat
echo [1/2] 启动 NapCat QQ 协议端...
start "NapCat" /min cmd /c "cd /d D:\NapCat\NapCat.Shell.Windows.Node && node.exe ./index.js"
echo        NapCat 已在后台启动
echo.

:: 2. Bot
echo [2/2] 启动小鼠 Bot...
echo.
cd /d "D:\code\QQ chat"
start "小鼠Bot" cmd /c "venv\Scripts\python bot.py"

:: 3. Health check
echo        等待 Bot 启动中...
ping -n 5 127.0.0.1 >nul

set /a tries=0
:check
set /a tries+=1
curl -s -o NUL http://127.0.0.1:18080 2>nul
if %errorlevel% equ 0 goto success
if %tries% geq 10 goto fail
ping -n 2 127.0.0.1 >nul
goto check

:success
echo.
echo ========================================
echo    [OK] 小鼠已成功启动!
echo.
echo    Bot 运行在 127.0.0.1:18080
echo    NapCat 状态请查看托盘图标
echo    QQ 里私聊 bot 即可开始对话
echo ========================================
echo.
echo 提示: 关闭此窗口不会停止 Bot.
echo       Bot 和 NapCat 在独立窗口中运行.
echo.
pause
exit

:fail
echo.
echo ========================================
echo    [FAIL] 启动超时, 请检查:
echo    1. .env 中的 API Key 是否正确
echo    2. 端口 18080 是否被占用
echo    3. 查看 Bot 窗口的错误日志
echo ========================================
echo.
pause
exit
