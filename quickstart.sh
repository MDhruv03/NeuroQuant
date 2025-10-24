#!/bin/bash
# NeuroQuant Quick Start Script

echo "🚀 NeuroQuant Trading System - Quick Start"
echo "==========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "✏️  Please edit .env file with your configuration"
fi

# Initialize database
echo "💾 Initializing database..."
python -c "from database.database import create_db_and_tables; create_db_and_tables()"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs models checkpoints

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "  Development: python backend/main.py"
echo "  Production:  uvicorn backend.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "Or use Docker:"
echo "  docker-compose up -d"
echo ""
echo "Access the application:"
echo "  - API: http://localhost:8000"
echo "  - Docs: http://localhost:8000/docs"
echo "  - Frontend: Open frontend/index.html"
echo ""
