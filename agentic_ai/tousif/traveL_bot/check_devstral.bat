@echo off
echo Checking Devstral Travel Advisory Bot Setup...
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+
    pause
    exit /b 1
)
echo ✅ Python is available

REM Check Ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Ollama not running. Please start Ollama with: ollama serve
    pause
    exit /b 1
)
echo ✅ Ollama is running

REM Check Devstral model
curl -s http://localhost:11434/api/tags | find "devstral" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Devstral model not found. Please install with: ollama pull devstral
    pause
    exit /b 1
)
echo ✅ Devstral model is available

REM Check project files
if not exist "src\travel_bot\__init__.py" (
    echo ❌ Project source files not found
    pause
    exit /b 1
)
echo ✅ Project files are in place

echo.
echo 🎉 All checks passed! Your setup is ready.
echo To start the application, run: start_devstral.bat
pause
