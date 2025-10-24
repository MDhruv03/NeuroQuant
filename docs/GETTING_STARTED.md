# Getting Started with NeuroQuant

Complete guide to set up and run NeuroQuant in under 10 minutes.

## Prerequisites

- **Python 3.11+** ([Download](https://www.python.org/downloads/))
- **Git** ([Download](https://git-scm.com/downloads))
- **TA-Lib** (see platform-specific instructions below)

## Quick Start (All Platforms)

### Step 1: Clone Repository

```bash
git clone https://github.com/MDhruv03/NeuroQuant.git
cd NeuroQuant
```

### Step 2: Install TA-Lib

#### Windows
```powershell
# Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# Choose file matching your Python version (e.g., cp311 = Python 3.11)
pip install TA_Lib-0.4.25-cp311-cp311-win_amd64.whl
```

#### macOS
```bash
brew install ta-lib
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y libta-lib-dev build-essential
```

### Step 3: Run Quick Start Script

#### Windows
```powershell
.\quickstart.bat
```

#### macOS/Linux
```bash
chmod +x quickstart.sh
./quickstart.sh
```

This automatically:
- Creates virtual environment
- Installs all dependencies
- Initializes database
- Sets up configuration

### Step 4: Configure (Optional)

```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys (optional)
nano .env  # or use any text editor
```

### Step 5: Start the Server

```bash
# Activate virtual environment first
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Start server
python backend/main.py
```

Server starts at: **http://localhost:8000**

### Step 6: Access the Interface

Open your browser:
- **Web Dashboard**: http://localhost:8000 (or http://127.0.0.1:5500 with Live Server)
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Docker Quick Start

If you prefer Docker:

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

Access at: **http://localhost**

---

## First Backtest

1. Open the web dashboard
2. Upload sample CSV or use default data
3. Select symbol (e.g., AAPL)
4. Choose agent type (DQN recommended)
5. Click "Run Backtest"
6. View results and performance metrics

---

## Manual Installation

If quick start scripts don't work:

```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from database.database import create_db_and_tables; create_db_and_tables()"

# Start server
python backend/main.py
```

---

## Verification

Check if everything works:

```bash
# Test import
python -c "import talib; print('TA-Lib OK')"

# Test API
curl http://localhost:8000/health

# Run tests
pytest
```

---

## Next Steps

- Read [Architecture](./ARCHITECTURE.md) to understand the system
- Check [Configuration](./CONFIGURATION.md) for customization
- See [Agent Development](./AGENTS.md) to create custom strategies
- Review [API Reference](./API.md) for integration

---

## Troubleshooting

### TA-Lib Installation Fails

**Windows**: Make sure wheel matches Python version exactly
**macOS**: Try `brew reinstall ta-lib`
**Linux**: Install build tools: `sudo apt-get install build-essential`

### Port Already in Use

```bash
# Change port in .env
API_PORT=8001
```

### Database Errors

```bash
# Reset database
rm database/neuroquant.db
python -c "from database.database import create_db_and_tables; create_db_and_tables()"
```

### Module Not Found

```bash
# Make sure virtual environment is activated
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

For more issues, see [FAQ](./FAQ.md)
