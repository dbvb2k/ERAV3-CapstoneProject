@echo off
echo Starting AI Travel Planner...
echo.

REM Check for Python
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check for OpenRouter API key
if not defined OPENROUTER_API_KEY (
    echo ❌ OPENROUTER_API_KEY environment variable not set
    echo Please set it before running the application
    echo You can get an API key from https://openrouter.ai/
    pause
    exit /b 1
)

REM Install dependencies with compatible versions
echo Installing dependencies...
python -m pip install -q ^
    streamlit>=1.32.0 ^
    fastapi>=0.109.0 ^
    uvicorn>=0.27.0 ^
    langchain>=0.1.9 ^
    langchain-core>=0.3.64 ^
    langchain-community>=0.0.24 ^
    langchain-anthropic>=0.3.15 ^
    langchain-openai>=0.3.21 ^
    python-dotenv>=1.0.1 ^
    transformers>=4.41.0 ^
    torch>=2.7.0 ^
    torchvision>=0.22.0 ^
    aiohttp>=3.12.4 ^
    python-weather>=2.1.0 ^
    requests>=2.31.0 ^
    nest-asyncio>=1.6.0 ^
    protobuf>=4.25.2 ^
    sentence-transformers>=4.1.0 ^
    --no-cache-dir

REM Start the application
echo Starting application...
python run_app.py --server.port 8501 --server.headless true

pause 