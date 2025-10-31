# NeuroQuant - Enterprise Algorithmic Trading Platform 🚀

**Status: ✅ WORLD-CLASS TRANSFORMATION COMPLETE**

Professional reinforcement learning trading system with institutional-grade analytics, comprehensive backtesting, professional dashboard, and enterprise-ready infrastructure.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-blue.svg)](https://kubernetes.io/)
[![AWS/GCP/Azure Ready](https://img.shields.io/badge/Cloud-Ready-brightgreen.svg)](https://aws.amazon.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **📖 READ FIRST**: [TRANSFORMATION_COMPLETE.md](TRANSFORMATION_COMPLETE.md) - Learn about the enterprise evolution
>
> **🚀 QUICK START**: See [GETTING_STARTED_ENTERPRISE.md](GETTING_STARTED_ENTERPRISE.md)

---

## Features

- **🤖 Advanced RL Agents** - DQN, PPO, and indicator-based strategies
- **📊 Real-time Analysis** - FinBERT sentiment + 20+ technical indicators
- **⚡ High Performance** - Caching, vectorized operations, optimized backtesting
- **🎨 Modern UI** - Minimalist monochrome interface with real-time charts
- **🐳 Docker Ready** - Full containerization with docker-compose
- **📈 Comprehensive Metrics** - Sharpe, Sortino, drawdown, win rate, and more
- **🔒 Production Ready** - Logging, error handling, validation, testing

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/MDhruv03/NeuroQuant.git
cd NeuroQuant

# Run quick start script
# Windows:
.\quickstart.bat
# macOS/Linux:
chmod +x quickstart.sh && ./quickstart.sh

# Start server
python backend/main.py
```

Server starts at **http://localhost:8000**

### Docker

```bash
docker-compose up -d
```

Access at **http://localhost**

---

## Documentation

📖 **[Complete Documentation](./docs/README.md)**

- [Getting Started](./docs/GETTING_STARTED.md) - Installation and setup
- [Architecture](./docs/ARCHITECTURE.md) - System design and components
- [API Reference](./docs/API.md) - Complete endpoint documentation
- [Configuration](./docs/CONFIGURATION.md) - Environment and settings
- [Agent Development](./docs/AGENTS.md) - Creating custom strategies
- [Deployment](./docs/DEPLOYMENT.md) - Production deployment guide

---

## Architecture

```
Frontend (HTML/CSS/JS) → FastAPI → Agent Manager → RL Agents (DQN/PPO)
                              ↓           ↓
                        Market Data  Sentiment
                        (yfinance)   (FinBERT)
                              ↓
                         SQLite DB
```

See [Architecture Documentation](./docs/ARCHITECTURE.md) for details.

---

## Project Structure

```
NeuroQuant/
├── backend/          # FastAPI application
│   ├── api/         # Routes and endpoints
│   ├── models/      # Pydantic schemas
│   └── services/    # Business logic
├── frontend/        # Web interface
├── rl/             # RL agents and environment
├── database/       # SQLite management
├── utils/          # Logging, exceptions, helpers
├── tests/          # Test suite
├── docs/           # Documentation
├── config.py       # Configuration
└── requirements.txt
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.11, FastAPI, Uvicorn |
| ML/RL | PyTorch, Stable-Baselines3, Gymnasium |
| NLP | Transformers, FinBERT |
| Data | yfinance, TA-Lib, Pandas, NumPy |
| Frontend | HTML5, CSS3, JavaScript, Tailwind CSS |
| Database | SQLite3 |
| Deployment | Docker, Docker Compose |

---

## Performance Metrics

Calculate comprehensive trading metrics:

- **Risk-Adjusted Returns** - Sharpe Ratio, Sortino Ratio
- **Drawdown Analysis** - Maximum Drawdown, Calmar Ratio
- **Trade Statistics** - Win Rate, Profit Factor
- **Portfolio Tracking** - Equity curves, position sizing

---

## API Endpoints

```
POST /backtest           - Run backtest simulation
GET  /agents             - List all agents
POST /agents             - Create new agent
GET  /backtest_runs      - Historical results
POST /upload_dataset     - Upload custom data
GET  /health             - System status
```

Interactive docs: **http://localhost:8000/docs**

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Contact

**Repository**: [MDhruv03/NeuroQuant](https://github.com/MDhruv03/NeuroQuant)

---

⭐ **Star this repository if you find it helpful!**
- **FastAPI Backend** with comprehensive API endpoints and Pydantic validation
- **Advanced RL Agents**: DQN (Deep Q-Network), PPO, and Indicator-Based strategies
- **Real Sentiment Analysis**: FinBERT integration for market sentiment scoring
- **Technical Analysis**: 20+ indicators via TA-Lib (SMA, RSI, MACD, Bollinger Bands, etc.)
- **Comprehensive Backtesting**: Detailed performance metrics and trade analysis
- **Interactive Web Dashboard**: Real-time visualization and agent management
- **Data Caching**: Optimized performance with Redis support
- **Robust Error Handling**: Custom exceptions and structured logging
- **Docker Support**: Fully containerized deployment

### Performance Metrics
- Sharpe Ratio & Sortino Ratio
- Maximum Drawdown & Calmar Ratio
- Win Rate & Profit Factor
- Portfolio value tracking with equity curves

---

## �️ Architecture

```
NeuroQuant/
├── backend/              # FastAPI application
│   ├── api/             # API routes and endpoints
│   ├── models/          # Pydantic data models
│   ├── services/        # Business logic services
│   │   ├── market_data.py
│   │   ├── trading_agent.py
│   │   ├── agent_manager.py
│   │   └── sentiment_analysis.py
│   └── main.py         # Application entry point
├── database/           # SQLite database management
├── frontend/           # HTML/CSS/JS web interface
├── rl/                # Reinforcement learning agents
│   ├── agent.py       # Base agent classes
│   ├── dqn_agent.py   # Deep Q-Network implementation
│   ├── environment.py  # Trading environment
│   └── train.py       # Training scripts
├── utils/             # Utilities and helpers
│   ├── logging_config.py
│   ├── exceptions.py
│   └── helpers.py
├── tests/             # Comprehensive test suite
├── config.py          # Centralized configuration
├── requirements.txt   # Python dependencies
├── Dockerfile        # Docker container definition
└── docker-compose.yml # Multi-service orchestration
```

---

## 📋 Installation

### Prerequisites
- Python 3.11+
- TA-Lib library (system-level installation required)
- Redis (optional, for caching)
- Docker & Docker Compose (for containerized deployment)

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y libta-lib-dev build-essential
```

**macOS:**
```bash
brew install ta-lib
```

**Windows:**
Download from [https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)

### Python Installation

1. **Clone the repository:**
```bash
git clone https://github.com/MDhruv03/NeuroQuant.git
cd NeuroQuant
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your settings
```

5. **Initialize database:**
```bash
python -c "from database.database import create_db_and_tables; create_db_and_tables()"
```

---

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

This will start:
- **NeuroQuant API** on `http://localhost:8000`
- **Frontend** on `http://localhost:80`
- **Redis Cache** on `localhost:6379`

### Manual Docker Build

```bash
# Build image
docker build -t neuroquant:latest .

# Run container
docker run -d -p 8000:8000 --name neuroquant neuroquant:latest
```

---

## 🎮 Usage

### Starting the Application

**Development mode:**
```bash
python backend/main.py
```

**Production mode:**
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints

Access the interactive API documentation:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

#### Key Endpoints:

- `POST /backtest` - Run backtest simulation
- `GET /agents` - List all trading agents
- `POST /agents` - Create new agent
- `GET /backtest_runs` - Historical backtest results
- `POST /upload_dataset` - Upload custom CSV dataset
- `GET /symbols` - Available trading symbols
- `GET /health` - System health check

### Web Interface

Open `http://localhost:8000` (or `http://localhost` with Docker) to access:
- **Dashboard**: Run backtests and view results
- **Agents**: Create and manage trading agents
- **Backtest History**: Analyze past performance

---

## � Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_helpers.py -v
```

---

## ⚙️ Configuration

Edit `.env` file or `config.py` to customize:

### RL Hyperparameters
```env
LEARNING_RATE=0.0003
GAMMA=0.99
EPSILON_START=1.0
EPSILON_END=0.01
BATCH_SIZE=64
```

### Financial Settings
```env
INITIAL_PORTFOLIO=10000
TRANSACTION_COST=0.001
SLIPPAGE=0.0005
```

### Sentiment Analysis
```env
SENTIMENT_ENABLED=true
SENTIMENT_MODEL=ProsusAI/finbert
NEWS_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
```

### Caching & Performance
```env
REDIS_ENABLED=true
DATA_CACHE_ENABLED=true
DATA_CACHE_TTL=3600
```

---

## 🤖 Creating Custom Agents

```python
from rl.agent import Agent
import numpy as np

class CustomAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        # Your initialization
    
    def train(self, timesteps=10000):
        # Your training logic
        pass
    
    def predict(self, obs):
        # Your prediction logic
        action = np.random.randint(0, 3)  # 0=Hold, 1=Buy, 2=Sell
        return action, None
```

Register via API:
```bash
curl -X POST "http://localhost:8000/agents" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Custom Agent",
    "type": "Custom",
    "parameters": {"param1": "value1"}
  }'
```

---

## 📊 Performance Metrics

The system calculates comprehensive metrics:

- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return vs. max drawdown
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profits / gross losses

---

## 🔐 Security & Production

### API Security
- Rate limiting enabled by default
- CORS configured for specific origins
- Input validation with Pydantic models
- Structured error handling

### Logging
- Rotating file handlers (10MB per file, 5 backups)
- Colored console output
- Configurable log levels
- Structured logging format

---

## 🛣️ Roadmap

### ✅ Phase 1: MVP (Complete)
- Basic RL agent implementation (DQN/PPO)
- Technical indicator integration
- FastAPI backend with comprehensive endpoints
- SQLite database for persistence
- Interactive web dashboard
- Backtesting engine with multiple data sources

### 🚧 Phase 2: Competitive Agents (In Progress)
- Multiple agents running in parallel
- Genetic evolution of strategies
- Performance tracking and comparison
- Strategy evolution visualization

### 📋 Phase 3: NLP & Market Intelligence
- Market regime detection (clustering)
- Adaptive strategy switching
- LLM-based news theme extraction
- Topic modeling and zero-shot classification

### 🔮 Phase 4: Production SaaS
- Microservices architecture (FastAPI + Celery + Redis)
- User authentication and multi-tenancy
- Async simulation with WebSockets
- Modern frontend
- Multi-stock portfolio allocation

### 🎯 Stretch Goals
- Real-time paper trading (Alpaca/Interactive Brokers)
- Agent leaderboard system
- Discord bot integration
- Deep Meta RL (MAML)
- Advanced risk management
