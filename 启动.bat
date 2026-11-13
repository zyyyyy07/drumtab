@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 鼓谱
if not exist ".venv\Scripts\python.exe" (
  echo 还没有安装运行环境，请先双击“安装.bat”。
  pause
  exit /b 1
)
echo 鼓谱启动中…… 浏览器会自动打开 http://127.0.0.1:8765
echo 用完直接关掉这个窗口即可。
start "" /b cmd /c "timeout /t 4 /nobreak >nul & start http://127.0.0.1:8765"
".venv\Scripts\python.exe" -m uvicorn app.server:app --host 127.0.0.1 --port 8765 --log-level warning
pause
