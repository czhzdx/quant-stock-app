@echo off
chcp 65001 >nul
title 量化选股软件

echo ========================================
echo      量化选股软件 启动中...
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 检查依赖是否安装
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo [提示] 首次运行，正在安装依赖...
    pip install -r requirements.txt
    echo.
)

echo [启动] 正在启动应用...
echo [提示] 请在浏览器中使用 http://localhost:8501 访问
echo [提示] 按 Ctrl+C 停止应用
echo.

REM 延迟3秒后打开浏览器
start "" cmd /c "timeout /t 3 >nul && start http://localhost:8501"

REM 启动Streamlit
streamlit run web_app.py

pause
