@echo off
echo Starting Devstral Travel Advisory Bot...
echo.

REM Set working directory
cd /d "%~dp0"

REM Install dependencies
echo Installing dependencies...
pip install -q streamlit requests python-dotenv
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Start the Streamlit application
echo Starting Streamlit application...
start "" http://localhost:8501
streamlit run src/travel_bot/ui/streamlit_app.py --server.port 8501 --server.headless true

pause
