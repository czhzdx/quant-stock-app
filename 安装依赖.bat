@echo off
chcp 65001 >nul
title 安装依赖

echo ========================================
echo      量化选股软件 - 安装依赖
echo ========================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo Python版本:
python --version
echo.

echo [安装] 正在安装依赖包...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [警告] TA-Lib安装失败，尝试使用whl文件安装
    echo 请从 https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib 下载对应版本
    echo 或运行: pip install TA_Lib-0.4.28-cp312-cp312-win_amd64.whl
)

echo.
echo ========================================
echo      安装完成！
echo ========================================
echo.
echo 双击 "启动应用.bat" 即可运行
echo.
pause
