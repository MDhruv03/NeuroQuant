# Configuration Guide

Complete reference for configuring NeuroQuant Trading System.

---

## Table of Contents

1. [Environment Variables](#environment-variables)
2. [Configuration File](#configuration-file)
3. [Database Configuration](#database-configuration)
4. [Logging Configuration](#logging-configuration)
5. [Agent Parameters](#agent-parameters)
6. [API Settings](#api-settings)

---

## Environment Variables

### Required Variables

Create a `.env` file in the project root:

```bash
# Database
DATABASE_URL=sqlite:///./trading.db

# API Keys (Optional - for production data)
ALPHA_VANTAGE_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here

# Sentiment Analysis
HUGGINGFACE_TOKEN=your_token_here  # Optional: for private models

# Cache (Optional)
REDIS_URL=redis://localhost:6379/0
REDIS_ENABLED=false

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/neuroquant.log
LOG_MAX_BYTES=10485760  # 10MB
LOG_BACKUP_COUNT=5
```

### Optional Variables

```bash
# Development
DEBUG=false
RELOAD=false  # Auto-reload on code changes

# Security
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=["http://localhost:3000"]

# Performance
WORKERS=4  # Number of worker processes
MAX_CONNECTIONS=100
```

---

## Configuration File

### `config.py`

Main configuration file with default settings:

```python
# config.py structure
class Config:
    # Application
    APP_NAME = "NeuroQuant Trading System"
    VERSION = "1.0.0"
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./trading.db")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR = "logs"
    LOG_FILE = "neuroquant.log"
    
    # API
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # Trading
    DEFAULT_INITIAL_BALANCE = 100000
    DEFAULT_COMMISSION = 0.001  # 0.1%
    
    # RL Training
    TOTAL_TIMESTEPS = 100000
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 32
    BUFFER_SIZE = 100000
```

### Customization

Override defaults by setting environment variables or modifying `config.py`:

```python
# Example: Custom configuration
from config import Config

Config.DEFAULT_INITIAL_BALANCE = 50000
Config.DEFAULT_COMMISSION = 0.002
```

---

## Database Configuration

### SQLite (Default)

No additional setup required:

```python
DATABASE_URL=sqlite:///./trading.db
```

### PostgreSQL (Production)

Install PostgreSQL adapter:

```bash
pip install psycopg2-binary
```

Set connection string:

```bash
DATABASE_URL=postgresql://user:password@localhost:5432/neuroquant
```

### MySQL

Install MySQL adapter:

```bash
pip install mysqlclient
```

Set connection string:

```bash
DATABASE_URL=mysql://user:password@localhost:3306/neuroquant
```

### Database Schema

Auto-generated on startup. Manual migration:

```python
from database.database import Base, engine

# Create all tables
Base.metadata.create_all(bind=engine)
```

---

## Logging Configuration

### Log Levels

Available levels (set via `LOG_LEVEL` env var):

- `DEBUG` - Detailed diagnostic information
- `INFO` - General informational messages (default)
- `WARNING` - Warning messages for potential issues
- `ERROR` - Error messages for failures
- `CRITICAL` - Critical errors causing system failure

### Log Files

Rotating file handler configuration in `utils/logging_config.py`:

```python
# Default settings
LOG_FILE = "logs/neuroquant.log"
MAX_BYTES = 10485760  # 10MB per file
BACKUP_COUNT = 5  # Keep 5 backup files
```

### Log Format

```
2024-01-15 14:30:22 [INFO] backend.main: Server started on port 8000
2024-01-15 14:30:25 [DEBUG] services.market_data: Fetching data for AAPL
2024-01-15 14:30:27 [ERROR] services.trading_agent: Trade execution failed
```

### Custom Logging

```python
from utils.logging_config import get_logger

logger = get_logger(__name__)

logger.info("Custom log message")
logger.debug("Debug information", extra={"context": data})
logger.error("Error occurred", exc_info=True)
```

---

## Agent Parameters

### DQN Agent

```python
# Training parameters
config = {
    "total_timesteps": 100000,
    "learning_rate": 0.0001,
    "batch_size": 32,
    "buffer_size": 100000,
    "learning_starts": 1000,
    "target_update_interval": 1000,
    "exploration_fraction": 0.1,
    "exploration_final_eps": 0.05,
    "gamma": 0.99,
    "tau": 0.005
}
```

### PPO Agent

```python
# Training parameters
config = {
    "total_timesteps": 100000,
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5
}
```

### Indicator-Based Agent

```python
# Strategy parameters
config = {
    "sma_short_period": 50,
    "sma_long_period": 200,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9
}
```

### Backtest Parameters

```python
# Backtest configuration
backtest_config = {
    "symbol": "AAPL",
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "initial_balance": 100000,
    "commission": 0.001,
    "slippage": 0.0005,
    "position_size": 1.0  # Fraction of portfolio per trade
}
```

---

## API Settings

### CORS Configuration

Configure allowed origins in `backend/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8000",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Rate Limiting

Configure rate limits (future feature):

```python
RATE_LIMIT_REQUESTS = 100  # Requests per minute
RATE_LIMIT_WINDOW = 60  # Time window in seconds
```

### API Timeouts

```python
API_TIMEOUT = 30  # Seconds
BACKTEST_TIMEOUT = 300  # 5 minutes for long operations
```

### Authentication (Future)

```python
# JWT configuration
JWT_SECRET_KEY = os.getenv("SECRET_KEY")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = 3600  # 1 hour
```

---

## Performance Tuning

### Memory Management

```python
# Limit data in memory
MAX_DATAPOINTS = 10000
CACHE_TTL = 3600  # 1 hour

# Clear cache periodically
CACHE_CLEANUP_INTERVAL = 3600
```

### CPU Optimization

```python
# Multi-processing
N_WORKERS = os.cpu_count()
PARALLEL_BACKTESTS = True

# Training optimization
USE_GPU = False  # Set True if CUDA available
N_ENVS = 4  # Parallel environments
```

### Network Settings

```python
# Connection pooling
DB_POOL_SIZE = 10
DB_MAX_OVERFLOW = 20
DB_POOL_TIMEOUT = 30

# API timeouts
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 30
```

---

## Docker Configuration

### Environment Variables in Docker

`docker-compose.yml`:

```yaml
environment:
  - DATABASE_URL=sqlite:///./data/trading.db
  - LOG_LEVEL=INFO
  - REDIS_ENABLED=false
```

### Volume Mounts

```yaml
volumes:
  - ./database:/app/database
  - ./logs:/app/logs
  - ./models:/app/models
  - ./checkpoints:/app/checkpoints
```

### Resource Limits

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

---

## Security Best Practices

1. **Never commit `.env` files** - Use `.env.example` template
2. **Rotate API keys regularly** - Update keys in production
3. **Use strong SECRET_KEY** - Generate with `secrets.token_urlsafe(32)`
4. **Enable HTTPS in production** - Use SSL certificates
5. **Restrict CORS origins** - Only allow trusted domains
6. **Validate user inputs** - Use Pydantic schemas
7. **Monitor logs** - Watch for suspicious activity
8. **Keep dependencies updated** - Regular security patches

---

## Troubleshooting

### Common Issues

**Database locked errors**:
```bash
# Solution: Use PostgreSQL for concurrent access
DATABASE_URL=postgresql://localhost/neuroquant
```

**Out of memory**:
```python
# Solution: Reduce data size
MAX_DATAPOINTS = 5000
BATCH_SIZE = 16
```

**Slow API responses**:
```bash
# Solution: Enable Redis caching
REDIS_ENABLED=true
REDIS_URL=redis://localhost:6379/0
```

**Import errors**:
```bash
# Solution: Install missing dependencies
pip install -r requirements.txt
```

---

## See Also

- [Getting Started Guide](GETTING_STARTED.md)
- [Architecture Overview](ARCHITECTURE.md)
- [API Reference](API.md)
- [Agent Development](AGENTS.md)
- [Deployment Guide](DEPLOYMENT.md)
