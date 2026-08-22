@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
call agent_env\Scripts\activate.bat
echo ========================================
echo   校园智能问答助手 启动中...
echo ========================================
echo.
streamlit run app.py
pause
