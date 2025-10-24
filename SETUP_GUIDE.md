# NeuroQuant Setup Guide

Complete step-by-step guide for setting up NeuroQuant on different platforms.

## Table of Contents
- [System Requirements](#system-requirements)
- [Platform-Specific Setup](#platform-specific-setup)
  - [Windows](#windows-setup)
  - [macOS](#macos-setup)
  - [Linux (Ubuntu/Debian)](#linux-ubuntu-debian-setup)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **CPU:** 2+ cores
- **RAM:** 4GB
- **Storage:** 2GB free space
- **OS:** Windows 10+, macOS 10.15+, Ubuntu 20.04+

### Recommended for Training
- **CPU:** 4+ cores
- **RAM:** 8GB+
- **GPU:** NVIDIA GPU with CUDA support (optional)
- **Storage:** 5GB+ free space

### Software Prerequisites
- Python 3.11 or higher
- pip (Python package manager)
- Git
- Docker (optional, for containerized deployment)

---

## Platform-Specific Setup

### Windows Setup

#### 1. Install Python

Download Python 3.11+ from [python.org](https://www.python.org/downloads/)

During installation:
- ✅ Check "Add Python to PATH"
- ✅ Check "Install pip"

Verify installation:
```powershell
python --version
pip --version
```

#### 2. Install TA-Lib

**Option A: Using Pre-built Wheel (Recommended)**

1. Download TA-Lib wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
   - Choose the file matching your Python version and architecture
   - Example: `TA_Lib-0.4.25-cp311-cp311-win_amd64.whl` for Python 3.11, 64-bit

2. Install the wheel:
```powershell
pip install TA_Lib-0.4.25-cp311-cp311-win_amd64.whl
```

**Option B: Build from Source (Advanced)**

Requires Visual Studio Build Tools:
1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
2. Download and extract TA-Lib source
3. Build and install

#### 3. Clone Repository

```powershell
git clone https://github.com/MDhruv03/NeuroQuant.git
cd NeuroQuant
```

#### 4. Run Quick Start Script

```powershell
.\quickstart.bat
```

This will:
- Create virtual environment
- Install all dependencies
- Initialize database
- Create necessary directories

#### 5. Configure Environment

```powershell
copy .env.example .env
notepad .env
```

Edit the `.env` file with your settings.

---

### macOS Setup

#### 1. Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### 2. Install Python

```bash
brew install python@3.11
```

Verify:
```bash
python3 --version
pip3 --version
```

#### 3. Install TA-Lib

```bash
brew install ta-lib
```

#### 4. Clone Repository

```bash
git clone https://github.com/MDhruv03/NeuroQuant.git
cd NeuroQuant
```

#### 5. Run Quick Start Script

```bash
chmod +x quickstart.sh
./quickstart.sh
```

#### 6. Configure Environment

```bash
cp .env.example .env
nano .env  # or use your preferred editor
```

---

### Linux (Ubuntu/Debian) Setup

#### 1. Update System

```bash
sudo apt update && sudo apt upgrade -y
```

#### 2. Install Python and Dependencies

```bash
sudo apt install -y python3.11 python3.11-venv python3-pip git
```

Verify:
```bash
python3.11 --version
pip3 --version
```

#### 3. Install TA-Lib

```bash
# Install build dependencies
sudo apt install -y build-essential wget

# Download and install TA-Lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Update library cache
sudo ldconfig
```

#### 4. Clone Repository

```bash
git clone https://github.com/MDhruv03/NeuroQuant.git
cd NeuroQuant
```

#### 5. Run Quick Start Script

```bash
chmod +x quickstart.sh
./quickstart.sh
```

#### 6. Configure Environment

```bash
cp .env.example .env
nano .env  # or vim, emacs, etc.
```

---

## Configuration

### Essential Configuration

Edit `.env` file with these required settings:

```env
# Database (default is fine for local)
DATABASE_URL=./database/neuroquant.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/neuroquant.log

# Financial Settings
INITIAL_PORTFOLIO=10000
TRANSACTION_COST=0.001
```

### Optional Configuration

#### Enable Redis Caching

```env
REDIS_ENABLED=true
REDIS_HOST=localhost
REDIS_PORT=6379
```

Requires Redis installation:
```bash
# Ubuntu/Debian
sudo apt install redis-server

# macOS
brew install redis

# Windows
# Download from https://github.com/microsoftarchive/redis/releases
```

#### Enable Sentiment Analysis

```env
SENTIMENT_ENABLED=true
SENTIMENT_MODEL=ProsusAI/finbert

# Optional: Add API keys for news
NEWS_API_KEY=your_newsapi_key
FINNHUB_API_KEY=your_finnhub_key
```

Get API keys:
- NewsAPI: [https://newsapi.org/](https://newsapi.org/)
- Finnhub: [https://finnhub.io/](https://finnhub.io/)

#### Configure RL Training

```env
# Training parameters
DEFAULT_TIMESTEPS=20000
LEARNING_RATE=0.0003
GAMMA=0.99
BATCH_SIZE=64
```

---

## Verification

### 1. Test Installation

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Run tests
pytest
```

Expected output:
```
collected 20+ items
tests/test_helpers.py ......
tests/test_market_data.py .....
...
========= 20 passed in 5.23s =========
```

### 2. Start the Server

```bash
python backend/main.py
```

Expected output:
```
============================================================
🚀 Starting NeuroQuant Trading System
📊 Version: 2.0.0
🌐 Environment: development
...
============================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Test API

Open browser and navigate to:
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

Or use curl:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-24T12:00:00",
  "version": "2.0.0",
  ...
}
```

### 4. Test Frontend

Open `frontend/index.html` in a web browser. You should see the NeuroQuant dashboard.

---

## Docker Setup (Alternative)

For containerized deployment:

### 1. Install Docker

- **Windows/Mac:** [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux:** 
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

### 2. Build and Run

```bash
# Single container
docker build -t neuroquant:latest .
docker run -d -p 8000:8000 neuroquant:latest

# Full stack with Docker Compose
docker-compose up -d
```

### 3. Verify

```bash
# Check containers
docker ps

# View logs
docker-compose logs -f neuroquant-api

# Access API
curl http://localhost:8000/health
```

### 4. Stop Services

```bash
docker-compose down
```

---

## Troubleshooting

### Common Issues

#### TA-Lib Installation Fails

**Windows:**
- Ensure you downloaded the correct wheel for your Python version
- Check if Visual Studio Build Tools are installed

**macOS:**
- Try: `brew reinstall ta-lib`
- Ensure Xcode Command Line Tools: `xcode-select --install`

**Linux:**
- Check if build tools are installed: `sudo apt install build-essential`
- Verify TA-Lib was installed: `ldconfig -p | grep ta-lib`

#### Import Errors

```
ModuleNotFoundError: No module named 'xyz'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### Database Errors

```
sqlite3.OperationalError: unable to open database file
```

**Solution:**
```bash
# Ensure database directory exists
mkdir -p database

# Reinitialize database
python -c "from database.database import create_db_and_tables; create_db_and_tables()"
```

#### Port Already in Use

```
Error: [Errno 48] Address already in use
```

**Solution:**
```bash
# Find process using port 8000
# Linux/Mac:
lsof -i :8000
kill -9 <PID>

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or use different port
python backend/main.py --port 8001
```

#### GPU/CUDA Issues

If training is slow and you have a GPU:

```bash
# Check if PyTorch detects GPU
python -c "import torch; print(torch.cuda.is_available())"
```

If False, install CUDA-enabled PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### Redis Connection Errors

```
redis.exceptions.ConnectionError
```

**Solution:**
1. Ensure Redis is running:
```bash
# Check Redis status
redis-cli ping  # Should return PONG

# Start Redis
# Linux: sudo systemctl start redis
# macOS: brew services start redis
# Windows: Start Redis service
```

2. Or disable Redis:
```env
REDIS_ENABLED=false
```

#### Out of Memory During Training

**Solution:**
```env
# Reduce batch size
BATCH_SIZE=32  # Default is 64

# Reduce memory size
MEMORY_SIZE=50000  # Default is 100000
```

### Getting Help

If you encounter issues not covered here:

1. **Check Documentation:**
   - README.md
   - API_DOCUMENTATION.md
   - CHANGELOG.md

2. **Search Issues:**
   - GitHub Issues: Look for similar problems

3. **Create an Issue:**
   - Include error message
   - System information
   - Steps to reproduce

4. **Community:**
   - Start a GitHub Discussion
   - Provide context and details

---

## Next Steps

After successful setup:

1. **Explore the API:**
   - Visit http://localhost:8000/docs
   - Try running a backtest
   - Create a custom agent

2. **Read Documentation:**
   - API_DOCUMENTATION.md for endpoint details
   - CONTRIBUTING.md if you want to contribute

3. **Experiment:**
   - Try different trading strategies
   - Tune hyperparameters
   - Upload custom datasets

4. **Deploy:**
   - Use Docker for production
   - Configure monitoring
   - Set up backups

---

**🎉 Congratulations! NeuroQuant is now ready to use.**

For questions or issues, please refer to the documentation or open an issue on GitHub.
