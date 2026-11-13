@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 鼓谱 - 安装运行环境
rem 只在换电脑或 .venv 文件夹被删掉时才需要运行。所有东西都装在本文件夹内。
set PIP_CACHE_DIR=%~dp0.pip-cache
py -3.12 -m venv .venv || (echo 需要先安装 Python 3.12 & pause & exit /b 1)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
".venv\Scripts\python.exe" -m pip install -r requirements.txt
".venv\Scripts\python.exe" -m pip install --no-deps basic-pitch
if not exist "vendor\ADTOF-pytorch" (
  rem 鼓识别模型 ADTOF-pytorch（作者 xavriley），固定到测试过的版本
  git clone https://github.com/xavriley/ADTOF-pytorch.git "vendor\ADTOF-pytorch" || (echo 下载 ADTOF-pytorch 失败，请检查网络或是否安装了 git & pause & exit /b 1)
  git -C "vendor\ADTOF-pytorch" checkout 85c192e78f716ea0b111cc8a5ee4a8f6a3a4f8a9
)
".venv\Scripts\python.exe" -m pip install --no-deps "vendor\ADTOF-pytorch"
rmdir /s /q .pip-cache 2>nul
echo.
echo 安装完成。双击“启动.bat”开始使用。
pause
