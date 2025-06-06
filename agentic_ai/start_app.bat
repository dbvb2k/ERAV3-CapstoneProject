@echo off
echo Starting AI Travel Planner...
echo.

REM Set working directory
cd /d "%~dp0"

REM Check if Python is installed
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Install dependencies using python -m pip
echo Installing dependencies...
python -m pip install -q streamlit requests python-dotenv transformers torch python-weather forex-python overpass geopy nest-asyncio --no-cache-dir
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    echo Please make sure you have internet connection and try again
    pause
    exit /b 1
)

REM Start the Streamlit application
echo Starting Streamlit application...
start "" http://localhost:8501

REM Run streamlit
python -m streamlit run app.py --server.port 8501 --server.headless true

pause 