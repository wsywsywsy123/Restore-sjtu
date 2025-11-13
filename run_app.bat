@echo off
chcp 65001 >nul
echo ========================================
echo 壁画病害诊断系统启动器
echo ========================================
echo.

REM 检查端口8501是否被占用
netstat -ano | findstr :8501 >nul
if %errorlevel% == 0 (
    echo 检测到端口8501已被占用
    echo 正在查找占用端口的进程...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8501 ^| findstr LISTENING') do (
        set PID=%%a
        echo 找到进程ID: %%a
        echo 正在终止该进程...
        taskkill /F /PID %%a >nul 2>&1
        if %errorlevel% == 0 (
            echo 进程已终止
        ) else (
            echo 无法终止进程，可能需要管理员权限
            echo 请手动关闭占用端口的程序，或使用"启动应用_备用端口.bat"
            timeout /t 3
            exit /b 1
        )
    )
    timeout /t 2 >nul
)

echo.
echo 正在启动应用...
echo 前端地址: http://localhost:8501
echo 按 Ctrl+C 停止服务
echo.
python -m streamlit run app.py --server.port 8501
pause
