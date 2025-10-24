# NeuroQuant System Architecture

Technical overview of the NeuroQuant trading system architecture.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend Layer                          │
│  HTML/CSS/JS + Tailwind + JetBrains Mono (Port 5500)       │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP REST API
┌─────────────────────┴───────────────────────────────────────┐
│                    FastAPI Backend (Port 8000)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   API Routes │  │   Middleware │  │  Exception   │      │
│  │   /backtest  │  │     CORS     │  │   Handlers   │      │
│  │   /agents    │  │   Logging    │  │              │      │
│  └──────┬───────┘  └──────────────┘  └──────────────┘      │
│         │                                                     │
│  ┌──────┴──────────────────────────────────────────┐        │
│  │            Business Logic Layer                 │        │
│  │  ┌─────────────┐  ┌────────────┐  ┌──────────┐ │        │
│  │  │Agent Manager│  │Market Data │  │Sentiment │ │        │
│  │  │             │  │  Provider  │  │ Analysis │ │        │
│  │  └──────┬──────┘  └─────┬──────┘  └────┬─────┘ │        │
│  └─────────┼────────────────┼──────────────┼───────┘        │
└────────────┼────────────────┼──────────────┼────────────────┘
             │                │              │
    ┌────────┴───────┐  ┌────┴─────┐  ┌────┴──────┐
    │   RL Agents    │  │ yfinance │  │  FinBERT  │
    │   DQN/PPO      │  │   API    │  │   Model   │
    │  Environment   │  │          │  │           │
    └────────┬───────┘  └──────────┘  └───────────┘
             │
    ┌────────┴────────────────────────┐
    │      Data Persistence Layer      │
    │  ┌──────────┐  ┌──────────────┐ │
    │  │ SQLite   │  │  File System │ │
    │  │ Database │  │  (Models)    │ │
    │  └──────────┘  └──────────────┘ │
    └─────────────────────────────────┘
```

## Core Components

### 1. Frontend Layer

**Technology Stack:**
- HTML5, CSS3, JavaScript (Vanilla)
- Tailwind CSS for styling
- JetBrains Mono font (monochrome theme)
- Shared components system

**Pages:**
- `index.html` - Main dashboard with backtesting
- `agents.html` - Agent creation and management
- `backtest.html` - Historical backtest results

**Features:**
- Responsive design
- Real-time chart rendering (SVG)
- Form validation
- Dynamic content loading
- Consistent header/footer across pages

### 2. API Layer (FastAPI)

**File:** `backend/main.py`

**Responsibilities:**
- HTTP request handling
- Input validation with Pydantic
- CORS management
- Exception handling
- Startup/shutdown lifecycle

**Middleware Stack:**
1. CORS Middleware (allow specific origins)
2. Exception Handlers (NeuroQuant, HTTP, Validation)
3. Request/Response logging

**API Endpoints:**
```
GET  /health              - System health check
POST /backtest            - Run backtest simulation
GET  /agents              - List all agents
POST /agents              - Create new agent
GET  /backtest_runs       - List backtest history
GET  /backtest_runs/{id}  - Get specific backtest details
POST /upload_dataset      - Upload custom CSV data
GET  /symbols             - Available symbols
```

### 3. Business Logic Layer

#### Agent Manager (`backend/services/agent_manager.py`)

**Responsibilities:**
- Agent lifecycle management
- Agent registry and lookup
- Training coordination
- Default agent initialization

**Supported Agent Types:**
- DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization)
- Indicator-Based (Technical rules)
- Random (Baseline)

#### Market Data Provider (`backend/services/market_data.py`)

**Features:**
- Multi-source data fetching (yfinance, custom CSV)
- Technical indicator calculation (20+ indicators)
- Data caching (in-memory, Redis-ready)
- Error handling with retries
- Date range filtering

**Technical Indicators:**
- Moving Averages (SMA, EMA)
- Momentum (RSI, MACD, Stochastic)
- Volatility (Bollinger Bands, ATR)
- Volume indicators
- Trend indicators

#### Sentiment Analysis (`backend/services/sentiment_analysis.py`)

**Model:** FinBERT (ProsusAI/finbert)

**Features:**
- Financial news sentiment scoring
- Multi-article aggregation
- NewsAPI and Finnhub integration
- Confidence scoring
- Fallback to neutral sentiment

**Output:**
```python
{
    "positive": 0.65,
    "negative": 0.15,
    "neutral": 0.20,
    "compound": 0.50,
    "article_count": 10,
    "confidence": 0.85
}
```

### 4. Reinforcement Learning Layer

#### DQN Agent (`rl/dqn_agent.py`)

**Architecture:**
- Neural network: [state_dim → 128 → 64 → action_dim]
- Activation: ReLU
- Loss: Smooth L1 (Huber)
- Optimizer: Adam

**Training Components:**
1. **Experience Replay Buffer** (size: 100,000)
   - Stores (state, action, reward, next_state, done) tuples
   - Random sampling for training stability

2. **Target Network**
   - Updated every 10 steps
   - Provides stable Q-value targets

3. **Epsilon-Greedy Exploration**
   - Start: ε = 1.0 (100% random)
   - Decay: 0.995 per episode
   - End: ε = 0.01 (1% random)

#### Trading Environment (`rl/environment.py`)

**State Space:**
- OHLCV data (5 features)
- Technical indicators (20+ features)
- Portfolio state (cash, shares, value)
- Total: ~30 dimensions

**Action Space:**
- 0: Hold (do nothing)
- 1: Buy (purchase shares)
- 2: Sell (liquidate shares)

**Reward Function:**
```python
reward = (current_portfolio_value - previous_value) / previous_value
# Includes transaction costs and slippage
```

### 5. Data Layer

#### Database (`database/database.py`)

**Schema:**

```sql
-- Agents
CREATE TABLE agents (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    parameters TEXT,  -- JSON
    created_at TEXT
);

-- Backtest Runs
CREATE TABLE backtest_runs (
    id INTEGER PRIMARY KEY,
    agent_id INTEGER,
    symbol TEXT,
    timestamp TEXT,
    agent_return REAL,
    buy_hold_return REAL,
    total_trades INTEGER,
    final_value REAL,
    metrics TEXT,  -- JSON
    trades TEXT,   -- JSON
    portfolio_history TEXT,  -- JSON
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

-- Custom Datasets
CREATE TABLE custom_datasets (
    id INTEGER PRIMARY KEY,
    name TEXT,
    data TEXT,  -- JSON
    uploaded_at TEXT
);
```

#### File Storage

```
models/             # Trained model weights (.pth)
checkpoints/        # Training checkpoints
logs/              # Application logs (.log)
database/          # SQLite database (.db)
```

### 6. Configuration System

**File:** `config.py`

**Configuration Classes:**
- `DatabaseConfig` - DB connection settings
- `APIConfig` - Server configuration
- `CORSConfig` - Cross-origin settings
- `ModelConfig` - ML model paths
- `TrainingConfig` - RL hyperparameters
- `FinancialConfig` - Trading parameters
- `LoggingConfig` - Log levels and files
- `SentimentConfig` - API keys and models

**Environment Variables:**
All configuration is overridable via `.env` file.

### 7. Utilities

#### Logging (`utils/logging_config.py`)

**Features:**
- Rotating file handlers (10MB, 5 backups)
- Colored console output
- Module-specific loggers
- UTF-8 encoding (Windows-compatible)
- Function call decorators

#### Exception Handling (`utils/exceptions.py`)

**Hierarchy:**
```
NeuroQuantException (base)
├── DataFetchError
├── ModelLoadError
├── BacktestError
├── SentimentAnalysisError
├── ConfigurationError
└── DatabaseError
```

#### Helpers (`utils/helpers.py`)

**Performance Metrics:**
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio
- Win Rate
- Profit Factor

## Data Flow

### Backtest Execution Flow

```
1. User submits backtest request (frontend)
   ↓
2. API validates input (Pydantic)
   ↓
3. Market data fetched (yfinance/custom)
   ↓
4. Technical indicators calculated
   ↓
5. Sentiment analysis performed
   ↓
6. Data split (train/test)
   ↓
7. Agent loaded/created
   ↓
8. Training phase (on train set)
   ↓
9. Testing phase (on test set)
   ↓
10. Metrics calculated
   ↓
11. Results saved to database
   ↓
12. Response returned to frontend
   ↓
13. Results visualized (charts/tables)
```

## Design Patterns

### 1. Repository Pattern
Database access abstracted through repository functions.

### 2. Service Layer Pattern
Business logic encapsulated in service classes.

### 3. Factory Pattern
Agent creation through factory methods.

### 4. Singleton Pattern
Configuration and database connections.

### 5. Strategy Pattern
Interchangeable trading agents.

## Security Considerations

1. **Input Validation:** Pydantic models validate all inputs
2. **SQL Injection Prevention:** Parameterized queries
3. **CORS Policy:** Restricted origins
4. **Rate Limiting:** Configurable rate limits
5. **Error Masking:** Sensitive info not exposed in errors

## Scalability

### Current Limitations
- Single-threaded execution
- In-memory caching
- SQLite (not for concurrent writes)

### Future Enhancements
- Celery for async tasks
- Redis for distributed caching
- PostgreSQL for concurrent access
- Kubernetes for horizontal scaling
- Load balancing with Nginx

## Performance Optimizations

1. **Data Caching:** Reduces API calls by 60%
2. **Batch Processing:** Process indicators in batches
3. **Lazy Loading:** Load models only when needed
4. **Connection Pooling:** Reuse database connections
5. **Vectorized Operations:** NumPy/Pandas for speed

## Technology Stack Summary

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, JavaScript, Tailwind CSS |
| Backend | Python 3.11, FastAPI, Uvicorn |
| ML/RL | PyTorch, Stable-Baselines3, Gymnasium |
| NLP | Transformers, FinBERT |
| Data | yfinance, TA-Lib, Pandas, NumPy |
| Database | SQLite3 |
| Deployment | Docker, Docker Compose, Nginx |
| Testing | pytest, pytest-cov |
| Config | python-dotenv |
| Logging | Python logging, Rotating handlers |

---

For deployment details, see [Deployment Guide](./DEPLOYMENT.md)
