# Frequently Asked Questions (FAQ)

Common questions and troubleshooting guide for NeuroQuant Trading System.

---

## Table of Contents

1. [General Questions](#general-questions)
2. [Installation Issues](#installation-issues)
3. [Configuration Problems](#configuration-problems)
4. [Trading & Backtesting](#trading--backtesting)
5. [Agent Training](#agent-training)
6. [Performance Issues](#performance-issues)
7. [Deployment](#deployment)
8. [Development](#development)

---

## General Questions

### What is NeuroQuant?

NeuroQuant is an AI-powered algorithmic trading system that uses Deep Reinforcement Learning (DQN, PPO) and technical analysis to make trading decisions. It includes backtesting capabilities, sentiment analysis, and a web-based dashboard.

### What can I do with NeuroQuant?

- Train custom RL agents for trading strategies
- Backtest strategies on historical data
- Analyze market sentiment from news
- Compare different trading algorithms
- Visualize performance metrics
- Deploy automated trading systems

### Is NeuroQuant suitable for live trading?

NeuroQuant is primarily designed for **backtesting and research**. While it can be adapted for live trading, you should:
- Thoroughly backtest your strategies
- Implement proper risk management
- Start with paper trading
- Never risk more than you can afford to lose

### What markets does it support?

Currently supports:
- US stocks (via yfinance)
- Can be extended to: crypto, forex, commodities

### Is programming knowledge required?

- **Basic usage**: No, use the web interface
- **Custom agents**: Yes, Python knowledge needed
- **Advanced features**: Python + ML experience recommended

---

## Installation Issues

### TA-Lib installation fails on Windows

**Problem**: `error: Microsoft Visual C++ 14.0 or greater is required`

**Solution**:
```bash
# Use pre-built wheel
pip install TA-Lib-0.4.28-cp311-cp311-win_amd64.whl

# Or use conda
conda install -c conda-forge ta-lib
```

### TA-Lib installation fails on macOS

**Problem**: `fatal error: 'ta-lib/ta_libc.h' file not found`

**Solution**:
```bash
# Install via Homebrew
brew install ta-lib

# Then install Python wrapper
pip install TA-Lib
```

### "No module named 'talib'" error

**Problem**: TA-Lib not installed or wrong Python environment

**Solution**:
```bash
# Verify installation
python -c "import talib; print(talib.__version__)"

# If fails, reinstall
pip uninstall TA-Lib
pip install TA-Lib
```

### Docker build fails

**Problem**: Timeout during build or dependency errors

**Solution**:
```bash
# Increase Docker memory (Docker Desktop settings)
# Use buildkit for faster builds
DOCKER_BUILDKIT=1 docker-compose build

# Build with no cache
docker-compose build --no-cache
```

### Python version incompatibility

**Problem**: `SyntaxError` or compatibility issues

**Solution**:
```bash
# Check Python version
python --version

# NeuroQuant requires Python 3.9+
# Install correct version
pyenv install 3.11.0
pyenv local 3.11.0
```

---

## Configuration Problems

### "Database locked" error

**Problem**: SQLite database locked (concurrent access)

**Solution**:
```bash
# Option 1: Use PostgreSQL for production
DATABASE_URL=postgresql://user:pass@localhost/trading

# Option 2: Close other connections
# Option 3: Delete database and recreate
rm database/trading.db
```

### Environment variables not loading

**Problem**: `.env` file not read

**Solution**:
```bash
# Verify .env location (project root)
ls -la .env

# Check file format (no spaces around =)
# Correct:   DATABASE_URL=sqlite:///./trading.db
# Incorrect: DATABASE_URL = sqlite:///./trading.db

# Verify loading in code
python -c "from config import Config; print(Config.DATABASE_URL)"
```

### API keys not working

**Problem**: External API errors (Alpha Vantage, Polygon)

**Solution**:
```bash
# Verify key in .env
cat .env | grep API_KEY

# Test key manually
curl "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey=YOUR_KEY"

# yfinance doesn't require API key (default)
```

### Redis connection errors

**Problem**: `ConnectionError: Error connecting to Redis`

**Solution**:
```bash
# Disable Redis if not using
REDIS_ENABLED=false

# Or start Redis
docker run -d -p 6379:6379 redis:alpine

# Or via docker-compose
docker-compose up -d redis
```

---

## Trading & Backtesting

### Backtest returns are unrealistic

**Problem**: Extremely high returns in backtests

**Possible causes**:
- **Overfitting**: Agent memorized training data
- **Look-ahead bias**: Using future data
- **No transaction costs**: Missing commissions/slippage

**Solution**:
```python
# Split train/test data properly
train_data = data[:int(len(data)*0.8)]
test_data = data[int(len(data)*0.8):]

# Add transaction costs
backtest_config = {
    "commission": 0.001,  # 0.1%
    "slippage": 0.0005     # 0.05%
}

# Use walk-forward validation
# Prevent overfitting with regularization
```

### Agent only holds (no trades)

**Problem**: Agent never buys or sells

**Possible causes**:
- **Insufficient training**: Not enough timesteps
- **Poor reward function**: Holding is "safest"
- **Wrong exploration rate**: Too low epsilon

**Solution**:
```python
# Increase training time
agent.train(total_timesteps=100000)  # Instead of 10000

# Adjust reward to encourage trading
def _calculate_reward(self):
    return portfolio_return - 0.01 * num_trades  # Small trade penalty

# Increase exploration
config = {
    "exploration_final_eps": 0.1  # Instead of 0.05
}
```

### Sharpe ratio is negative

**Problem**: Negative or very low Sharpe ratio

**Explanation**: Sharpe ratio measures risk-adjusted returns:
- **< 0**: Strategy loses money
- **0-1**: Acceptable
- **1-2**: Good
- **>2**: Excellent

**Solution**:
```python
# Improve strategy:
# 1. Better features (indicators, sentiment)
# 2. Longer training
# 3. Different agent type
# 4. Parameter tuning
```

### "Insufficient data for indicators" error

**Problem**: Not enough historical data for technical indicators

**Solution**:
```python
# Minimum data points needed:
# - SMA(200): 200 days
# - RSI(14): 14 days
# - MACD: 26 days

# Fetch more data
data = fetch_stock_data("AAPL", "2020-01-01", "2024-01-01")  # 4 years

# Or reduce indicator periods
config = {
    "sma_period": 50  # Instead of 200
}
```

---

## Agent Training

### Training is very slow

**Problem**: Takes hours to train agent

**Solution**:
```python
# 1. Reduce data size
data = data[-1000:]  # Use last 1000 days only

# 2. Use fewer timesteps initially
agent.train(total_timesteps=10000)  # Quick test

# 3. Use GPU (if available)
# Install: pip install stable-baselines3[extra]

# 4. Vectorized environments
from stable_baselines3.common.vec_env import DummyVecEnv
env = DummyVecEnv([lambda: TradingEnvironment(data) for _ in range(4)])
```

### "Out of memory" error during training

**Problem**: RAM exhausted

**Solution**:
```python
# Reduce buffer size
config = {
    "buffer_size": 10000  # Instead of 100000
}

# Reduce batch size
config = {
    "batch_size": 16  # Instead of 64
}

# Use smaller network
# Limit data points
data = data[-500:]
```

### Agent performance degrades over time

**Problem**: Good initial training, but worse on new data

**Cause**: Overfitting to training data

**Solution**:
```python
# Use validation set
train_data, val_data = train_test_split(data, test_size=0.2)

# Monitor validation performance
if val_performance < threshold:
    stop_training()

# Use regularization
# Add dropout to network
# Early stopping
```

### Saved model won't load

**Problem**: `FileNotFoundError` or compatibility errors

**Solution**:
```bash
# Verify file exists
ls -la models/

# Check file permissions
chmod 644 models/agent.zip

# Ensure same library versions
pip list | grep stable-baselines3

# Reload model properly
agent = DQNAgent.load("models/agent.zip")
```

---

## Performance Issues

### API responses are slow

**Problem**: Requests take >5 seconds

**Solution**:
```bash
# Enable Redis caching
REDIS_ENABLED=true
docker-compose up -d redis

# Use database connection pooling
# Optimize database queries
# Add indexes to database tables

# Profile code
python -m cProfile -o profile.stats backend/main.py
```

### High memory usage

**Problem**: Application uses >4GB RAM

**Solution**:
```python
# Limit data in memory
MAX_DATAPOINTS = 5000

# Clear cache periodically
cache.clear()

# Use generators instead of lists
def get_data():
    for chunk in chunks:
        yield chunk

# Monitor memory
import tracemalloc
tracemalloc.start()
```

### Database grows too large

**Problem**: `trading.db` file is several GB

**Solution**:
```bash
# Archive old data
sqlite3 trading.db "DELETE FROM trades WHERE date < '2020-01-01'"

# Vacuum database
sqlite3 trading.db "VACUUM"

# Use PostgreSQL with partitioning
# Implement data retention policy
```

---

## Deployment

### Docker container keeps restarting

**Problem**: `Status: Restarting` in `docker ps`

**Solution**:
```bash
# Check logs
docker-compose logs backend

# Common causes:
# - Port already in use
# - Missing environment variables
# - Database connection error
# - Insufficient resources

# Increase resources in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G
```

### CORS errors in browser

**Problem**: `Access-Control-Allow-Origin` error

**Solution**:
```python
# backend/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "https://your-domain.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### SSL certificate errors

**Problem**: HTTPS not working

**Solution**:
```bash
# Generate self-signed cert (development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem -out ssl/cert.pem

# Production: Use Let's Encrypt
sudo certbot --nginx -d your-domain.com
```

### Can't access from external network

**Problem**: Works on localhost but not remotely

**Solution**:
```bash
# Check firewall
sudo ufw allow 8000
sudo ufw allow 80
sudo ufw allow 443

# Bind to 0.0.0.0 (not 127.0.0.1)
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Check cloud provider security groups
# AWS: Allow inbound on ports 80, 443, 8000
```

---

## Development

### How to add a new indicator?

```python
# services/market_data.py
import talib

def calculate_custom_indicator(data):
    """Calculate custom indicator"""
    # Example: Relative Vigor Index
    rvi = talib.RVI(data['close'], data['high'], data['low'], timeperiod=10)
    return rvi

# Add to indicator pipeline
def calculate_indicators(data):
    indicators = {
        'sma': talib.SMA(data['close']),
        'rsi': talib.RSI(data['close']),
        'custom': calculate_custom_indicator(data)  # Add here
    }
    return indicators
```

### How to create a custom agent?

See [Agent Development Guide](AGENTS.md#creating-custom-agents)

### How to add a new API endpoint?

```python
# backend/api/routes.py
from fastapi import APIRouter

router = APIRouter()

@router.post("/custom_endpoint")
async def custom_endpoint(request: CustomRequest):
    """Custom endpoint description"""
    # Your logic here
    return {"result": "success"}

# Register in backend/main.py
from backend.api.routes import router
app.include_router(router)
```

### How to debug training issues?

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Print observations
print(f"Observation: {obs}")
print(f"Action: {action}")
print(f"Reward: {reward}")

# Visualize training
from stable_baselines3.common.callbacks import EvalCallback
callback = EvalCallback(eval_env, log_path="./logs/")
agent.train(callback=callback)

# Use TensorBoard
tensorboard --logdir ./logs/
```

### How to contribute?

```bash
# 1. Fork repository
# 2. Create feature branch
git checkout -b feature/amazing-feature

# 3. Make changes and test
pytest

# 4. Commit with meaningful message
git commit -m "Add amazing feature"

# 5. Push and create PR
git push origin feature/amazing-feature
```

---

## Additional Resources

- **Documentation**: [docs/](.)
- **GitHub Issues**: Report bugs and feature requests
- **Discussions**: Community Q&A
- **Discord**: Real-time help (if available)

---

## Still Having Issues?

If your problem isn't covered here:

1. **Check logs**: `logs/neuroquant.log`
2. **Search issues**: GitHub repository
3. **Create issue**: Provide:
   - Error message
   - Steps to reproduce
   - Environment (OS, Python version, etc.)
   - Relevant logs

---

## See Also

- [Getting Started Guide](GETTING_STARTED.md)
- [Configuration Guide](CONFIGURATION.md)
- [API Reference](API.md)
- [Architecture Overview](ARCHITECTURE.md)
- [Testing Guide](TESTING.md)
