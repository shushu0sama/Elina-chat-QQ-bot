@echo off
chcp 65001 >nul
title 小鼠 - QQ Bot

echo ──────────────────────────────
echo   小鼠 QQ Bot 启动中...
echo ──────────────────────────────
echo.

echo [1/2] 启动 NapCat...
start "NapCat" /min cmd /c "cd /d D:\NapCat\NapCat.Shell.Windows.Node && node.exe ./index.js"
echo        NapCat 已在后台启动

echo [2/2] 启动小鼠 Bot...
echo.
cd /d "D:\code\QQ chat"
venv\Scripts\python bot.py

echo.
echo 小鼠已退出。按任意键关闭窗口...
pause >nul
