# Testing Guide

Complete guide for testing NeuroQuant Trading System components.

---

## Table of Contents

1. [Test Setup](#test-setup)
2. [Running Tests](#running-tests)
3. [Unit Tests](#unit-tests)
4. [Integration Tests](#integration-tests)
5. [End-to-End Tests](#end-to-end-tests)
6. [Writing Tests](#writing-tests)

---

## Test Setup

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio httpx
```

### Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_helpers.py          # Test utilities
├── test_market_data.py      # Market data tests
├── test_agents.py           # Agent tests
├── test_api.py              # API endpoint tests
└── test_integration.py      # Integration tests
```

### Configuration

**`conftest.py`**:

```python
import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)

@pytest.fixture
def sample_data():
    """Sample market data"""
    return {
        "symbol": "AAPL",
        "date": "2024-01-01",
        "open": 180.0,
        "high": 185.0,
        "low": 178.0,
        "close": 183.0,
        "volume": 1000000
    }
```

---

## Running Tests

### All Tests

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=. --cov-report=html

# Verbose output
pytest -v

# Show print statements
pytest -s
```

### Specific Tests

```bash
# Run specific file
pytest tests/test_market_data.py

# Run specific test
pytest tests/test_market_data.py::test_fetch_data

# Run by marker
pytest -m "unit"
pytest -m "integration"

# Run by keyword
pytest -k "agent"
```

### Coverage

```bash
# Generate coverage report
pytest --cov=. --cov-report=html

# View report
# Open htmlcov/index.html in browser

# Coverage threshold
pytest --cov=. --cov-fail-under=80
```

---

## Unit Tests

### Market Data Tests

**`tests/test_market_data.py`**:

```python
import pytest
from services.market_data import fetch_stock_data, calculate_indicators

def test_fetch_stock_data():
    """Test fetching stock data"""
    data = fetch_stock_data("AAPL", "2024-01-01", "2024-01-31")
    
    assert data is not None
    assert len(data) > 0
    assert "close" in data.columns
    assert "volume" in data.columns

def test_fetch_invalid_symbol():
    """Test fetching invalid symbol"""
    with pytest.raises(ValueError):
        fetch_stock_data("INVALID", "2024-01-01", "2024-01-31")

def test_calculate_indicators():
    """Test technical indicator calculation"""
    import pandas as pd
    import numpy as np
    
    # Create sample data
    data = pd.DataFrame({
        'close': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(100, 200, 100),
        'low': np.random.uniform(100, 200, 100)
    })
    
    indicators = calculate_indicators(data)
    
    assert 'sma_50' in indicators
    assert 'rsi' in indicators
    assert 'macd' in indicators
    assert not indicators['sma_50'].isna().all()

@pytest.mark.parametrize("period,expected_length", [
    (14, 100),
    (20, 100),
    (50, 100)
])
def test_indicator_periods(period, expected_length):
    """Test different indicator periods"""
    import pandas as pd
    import numpy as np
    
    data = pd.DataFrame({
        'close': np.random.uniform(100, 200, 100)
    })
    
    from services.market_data import calculate_rsi
    rsi = calculate_rsi(data['close'], period)
    
    assert len(rsi) == expected_length
```

---

### Agent Tests

**`tests/test_agents.py`**:

```python
import pytest
import numpy as np
from rl.environment import TradingEnvironment
from rl.dqn_agent import DQNAgent

@pytest.fixture
def env():
    """Create test environment"""
    import pandas as pd
    
    data = pd.DataFrame({
        'close': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(100, 200, 100),
        'low': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000000, 5000000, 100)
    })
    
    return TradingEnvironment(data, initial_balance=100000)

def test_environment_reset(env):
    """Test environment reset"""
    obs = env.reset()
    
    assert obs is not None
    assert len(obs) == env.observation_space.shape[0]
    assert env.current_step == 0
    assert env.balance == env.initial_balance

def test_environment_step(env):
    """Test environment step"""
    env.reset()
    
    # Buy action
    obs, reward, done, info = env.step(1)
    
    assert obs is not None
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

def test_agent_predict(env):
    """Test agent prediction"""
    agent = DQNAgent(env)
    obs = env.reset()
    
    action, _ = agent.predict(obs)
    
    assert action in [0, 1, 2]  # hold, buy, sell

def test_agent_training(env):
    """Test agent training"""
    agent = DQNAgent(env)
    
    # Short training
    agent.train(total_timesteps=1000)
    
    # Check model exists
    assert agent.model is not None

@pytest.mark.slow
def test_agent_save_load(env, tmp_path):
    """Test agent save/load"""
    agent = DQNAgent(env)
    agent.train(total_timesteps=1000)
    
    # Save
    model_path = tmp_path / "test_agent.zip"
    agent.save(str(model_path))
    
    assert model_path.exists()
    
    # Load
    new_agent = DQNAgent(env)
    new_agent.load(str(model_path))
    
    # Compare predictions
    obs = env.reset()
    action1, _ = agent.predict(obs)
    action2, _ = new_agent.predict(obs)
    
    assert action1 == action2
```

---

### API Tests

**`tests/test_api.py`**:

```python
import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):
    """Test health endpoint"""
    response = client.get("/health")
    
    assert response.status_code == 200
    assert "status" in response.json()

def test_fetch_market_data(client):
    """Test market data endpoint"""
    response = client.post("/market_data", json={
        "symbol": "AAPL",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0

def test_backtest_endpoint(client):
    """Test backtest endpoint"""
    response = client.post("/backtest", json={
        "agent_type": "indicator",
        "symbol": "AAPL",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "initial_balance": 100000
    })
    
    assert response.status_code == 200
    results = response.json()
    
    assert "total_return" in results
    assert "sharpe_ratio" in results
    assert "max_drawdown" in results

def test_create_agent(client):
    """Test agent creation endpoint"""
    response = client.post("/agents", json={
        "name": "Test Agent",
        "type": "dqn",
        "config": {
            "learning_rate": 0.0001
        }
    })
    
    assert response.status_code == 200
    agent = response.json()
    
    assert "id" in agent
    assert agent["name"] == "Test Agent"

def test_list_agents(client):
    """Test list agents endpoint"""
    response = client.get("/agents")
    
    assert response.status_code == 200
    agents = response.json()
    
    assert isinstance(agents, list)

def test_invalid_request(client):
    """Test invalid request handling"""
    response = client.post("/market_data", json={
        "symbol": "",  # Invalid symbol
        "start_date": "2024-01-01"
    })
    
    assert response.status_code == 422  # Validation error

def test_sentiment_analysis(client):
    """Test sentiment endpoint"""
    response = client.post("/sentiment", json={
        "text": "Apple stock reaches new all-time high"
    })
    
    assert response.status_code == 200
    result = response.json()
    
    assert "sentiment" in result
    assert "score" in result
    assert result["sentiment"] in ["positive", "negative", "neutral"]
```

---

## Integration Tests

**`tests/test_integration.py`**:

```python
import pytest
from services.agent_manager import AgentManager
from services.market_data import fetch_stock_data
from services.trading_agent import backtest_agent

@pytest.mark.integration
def test_full_backtest_workflow():
    """Test complete backtest workflow"""
    # Fetch data
    data = fetch_stock_data("AAPL", "2024-01-01", "2024-01-31")
    assert data is not None
    
    # Run backtest
    results = backtest_agent(
        agent_type="indicator",
        symbol="AAPL",
        start_date="2024-01-01",
        end_date="2024-01-31",
        initial_balance=100000
    )
    
    # Verify results
    assert results["total_return"] is not None
    assert results["num_trades"] >= 0
    assert results["sharpe_ratio"] is not None

@pytest.mark.integration
def test_agent_manager():
    """Test agent manager operations"""
    manager = AgentManager()
    
    # Create agent
    agent_id = manager.create_agent(
        name="Test Agent",
        agent_type="dqn"
    )
    assert agent_id is not None
    
    # List agents
    agents = manager.list_agents()
    assert len(agents) > 0
    
    # Get agent
    agent = manager.get_agent(agent_id)
    assert agent["name"] == "Test Agent"
    
    # Delete agent
    manager.delete_agent(agent_id)
    agents = manager.list_agents()
    assert agent_id not in [a["id"] for a in agents]

@pytest.mark.integration
@pytest.mark.slow
def test_train_and_backtest():
    """Test training and backtesting"""
    from rl.environment import TradingEnvironment
    from rl.dqn_agent import DQNAgent
    
    # Get data
    data = fetch_stock_data("AAPL", "2020-01-01", "2023-12-31")
    
    # Split train/test
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # Train
    env = TradingEnvironment(train_data)
    agent = DQNAgent(env)
    agent.train(total_timesteps=10000)
    
    # Test
    test_env = TradingEnvironment(test_data)
    obs = test_env.reset()
    total_reward = 0
    
    done = False
    while not done:
        action, _ = agent.predict(obs)
        obs, reward, done, _ = test_env.step(action)
        total_reward += reward
    
    # Should make some profit
    assert total_reward > 0
```

---

## End-to-End Tests

**`tests/test_e2e.py`**:

```python
import pytest
import requests
import time

@pytest.mark.e2e
class TestE2E:
    """End-to-end tests"""
    
    BASE_URL = "http://localhost:8000"
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Wait for server to be ready"""
        max_retries = 10
        for _ in range(max_retries):
            try:
                response = requests.get(f"{self.BASE_URL}/health")
                if response.status_code == 200:
                    break
            except:
                time.sleep(1)
    
    def test_complete_workflow(self):
        """Test complete user workflow"""
        # 1. Check health
        response = requests.get(f"{self.BASE_URL}/health")
        assert response.status_code == 200
        
        # 2. Create agent
        response = requests.post(f"{self.BASE_URL}/agents", json={
            "name": "E2E Test Agent",
            "type": "indicator"
        })
        assert response.status_code == 200
        agent_id = response.json()["id"]
        
        # 3. Run backtest
        response = requests.post(f"{self.BASE_URL}/backtest", json={
            "agent_type": "indicator",
            "symbol": "AAPL",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "initial_balance": 100000
        })
        assert response.status_code == 200
        results = response.json()
        
        # 4. Verify results
        assert "total_return" in results
        
        # 5. Get backtest history
        response = requests.get(f"{self.BASE_URL}/backtest_runs")
        assert response.status_code == 200
        
        # 6. Cleanup
        response = requests.delete(f"{self.BASE_URL}/agents/{agent_id}")
        assert response.status_code == 200
```

---

## Writing Tests

### Best Practices

1. **Test naming**: Use descriptive names (`test_fetch_data_returns_dataframe`)
2. **Fixtures**: Reuse common setup code
3. **Mocking**: Mock external dependencies (APIs, databases)
4. **Parametrization**: Test multiple inputs efficiently
5. **Markers**: Organize tests by type/speed

### Example Test Template

```python
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_data():
    """Fixture for test data"""
    return {
        "symbol": "AAPL",
        "price": 180.0
    }

@pytest.mark.unit
def test_function_success(mock_data):
    """Test function with valid input"""
    # Arrange
    expected = 180.0
    
    # Act
    result = my_function(mock_data)
    
    # Assert
    assert result == expected

@pytest.mark.unit
def test_function_failure():
    """Test function with invalid input"""
    # Arrange & Act & Assert
    with pytest.raises(ValueError):
        my_function(None)

@pytest.mark.integration
@patch('services.market_data.yfinance')
def test_with_mock(mock_yf, mock_data):
    """Test with mocked external dependency"""
    # Arrange
    mock_yf.download.return_value = mock_data
    
    # Act
    result = fetch_stock_data("AAPL")
    
    # Assert
    assert result is not None
    mock_yf.download.assert_called_once()
```

### Markers

```python
# conftest.py
def pytest_configure(config):
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow tests")
```

### Coverage Goals

- **Unit tests**: 80%+ coverage
- **Integration tests**: Critical paths
- **E2E tests**: Main user workflows

---

## See Also

- [Architecture Overview](ARCHITECTURE.md)
- [API Reference](API.md)
- [Agent Development](AGENTS.md)
- [Configuration Guide](CONFIGURATION.md)
