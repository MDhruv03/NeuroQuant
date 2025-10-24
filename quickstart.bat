@echo off
REM NeuroQuant Quick Start Script for Windows

echo.
echo 🚀 NeuroQuant Trading System - Quick Start
echo ===========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Create .env if it doesn't exist
if not exist ".env" (
    echo ⚙️  Creating .env file...
    copy .env.example .env
    echo ✏️  Please edit .env file with your configuration
)

REM Initialize database
echo 💾 Initializing database...
python -c "from database.database import create_db_and_tables; create_db_and_tables()"

REM Create necessary directories
echo 📁 Creating directories...
if not exist "logs\" mkdir logs
if not exist "models\" mkdir models
if not exist "checkpoints\" mkdir checkpoints

echo.
echo ✅ Setup complete!
echo.
echo To start the application:
echo   Development: python backend\main.py
echo   Production:  uvicorn backend.main:app --host 0.0.0.0 --port 8000
echo.
echo Or use Docker:
echo   docker-compose up -d
echo.
echo Access the application:
echo   - API: http://localhost:8000
echo   - Docs: http://localhost:8000/docs
echo   - Frontend: Open frontend\index.html
echo.
pause
