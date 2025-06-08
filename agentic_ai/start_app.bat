@echo off
echo Starting AI Travel Planner...
echo.

REM Check for environment variables
if not defined OPENROUTER_API_KEY (
    echo ❌ OPENROUTER_API_KEY environment variable not set
    echo Please set it before running the application
    echo.
)

if not defined RAPIDAPI_KEY (
    echo ⚠️ RAPIDAPI_KEY environment variable not set
    echo For real flight and hotel data, please:
    echo 1. Sign up at https://rapidapi.com
    echo 2. Subscribe to:
    echo    - Travelpayouts Flight Data API
    echo    - Hotels.com API
    echo 3. Get your API key and set it as RAPIDAPI_KEY
    echo.
    echo The app will use simulated data for now.
    echo.
)

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