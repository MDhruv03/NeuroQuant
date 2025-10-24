# NeuroQuant API Reference

Complete API documentation for all endpoints.

## Base URL

```
http://localhost:8000
```

For Docker deployment: `http://localhost`

## Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Authentication

Currently no authentication required. For production:
- Add API key headers
- Implement JWT tokens
- Use OAuth2 integration

---

## Endpoints

### Health Check

Check system status and configuration.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-24T12:00:00",
  "version": "2.0.0",
  "environment": "development",
  "database_status": "connected",
  "cache_status": "enabled"
}
```

**Status Codes:**
- `200 OK` - System healthy
- `503 Service Unavailable` - System unhealthy

---

### Run Backtest

Execute a backtest simulation with specified parameters.

**Endpoint:** `POST /backtest`

**Request Body:**
```json
{
  "symbol": "AAPL",
  "train_split": 0.7,
  "agent_id": 1,
  "data_source": "yfinance",
  "custom_dataset_id": null,
  "start_date": "2023-01-01",
  "end_date": "2023-12-31"
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `symbol` | string | Yes | - | Stock ticker (e.g., "AAPL", "TSLA") |
| `train_split` | float | No | 0.7 | Train/test split ratio (0.1-0.9) |
| `agent_id` | integer | No | null | Agent ID to use (null = default) |
| `data_source` | string | Yes | - | "yfinance" or "custom" |
| `custom_dataset_id` | integer | No | null | Custom dataset ID if using custom data |
| `start_date` | string | No | null | Start date (YYYY-MM-DD) |
| `end_date` | string | No | null | End date (YYYY-MM-DD) |

**Response:**
```json
{
  "symbol": "AAPL",
  "test_period": "2023-08-01 to 2023-12-31",
  "agent_return": 15.43,
  "buy_hold_return": 12.56,
  "outperformance": 2.87,
  "total_trades": 45,
  "final_value": 11543.21,
  "trades": [
    {
      "date": "2023-08-01",
      "action": "buy",
      "shares": 10.5,
      "price": 175.23,
      "pnl": 0
    }
  ],
  "portfolio_history": [10000, 10100, 10250],
  "portfolio_dates": ["2023-08-01", "2023-08-02", "2023-08-03"],
  "metrics": {
    "sharpe_ratio": 1.85,
    "max_drawdown": 8.23,
    "win_rate": 58.5,
    "profit_factor": 1.67,
    "sortino_ratio": 2.12,
    "calmar_ratio": 1.88
  }
}
```

**Status Codes:**
- `200 OK` - Backtest successful
- `400 Bad Request` - Invalid parameters
- `500 Internal Server Error` - Backtest failed

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/backtest" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "train_split": 0.7,
    "data_source": "yfinance"
  }'
```

**Example (Python):**
```python
import requests

response = requests.post(
    "http://localhost:8000/backtest",
    json={
        "symbol": "AAPL",
        "train_split": 0.7,
        "data_source": "yfinance"
    }
)
result = response.json()
print(f"Agent Return: {result['agent_return']}%")
```

---

### List Agents

Get all registered trading agents.

**Endpoint:** `GET /agents`

**Response:**
```json
[
  {
    "id": 1,
    "name": "DQN Agent (Default)",
    "type": "DQN",
    "parameters": {
      "learning_rate": 0.001,
      "gamma": 0.99,
      "epsilon_decay": 0.995
    }
  },
  {
    "id": 2,
    "name": "PPO Agent",
    "type": "PPO",
    "parameters": {
      "learning_rate": 0.0003,
      "n_steps": 2048
    }
  }
]
```

**Status Codes:**
- `200 OK` - Agents retrieved
- `500 Internal Server Error` - Database error

---

### Create Agent

Register a new trading agent.

**Endpoint:** `POST /agents`

**Request Body:**
```json
{
  "name": "My Custom Agent",
  "type": "DQN",
  "parameters": {
    "learning_rate": 0.001,
    "gamma": 0.99,
    "batch_size": 64
  }
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique agent name |
| `type` | string | Yes | "DQN", "PPO", or "IndicatorBased" |
| `parameters` | object | No | Custom hyperparameters (JSON) |

**Response:**
```json
{
  "id": 3,
  "name": "My Custom Agent",
  "type": "DQN",
  "parameters": {
    "learning_rate": 0.001,
    "gamma": 0.99,
    "batch_size": 64
  }
}
```

**Status Codes:**
- `200 OK` - Agent created
- `400 Bad Request` - Invalid parameters
- `500 Internal Server Error` - Creation failed

---

### List Backtest Runs

Get historical backtest results.

**Endpoint:** `GET /backtest_runs`

**Response:**
```json
[
  {
    "id": 1,
    "symbol": "AAPL",
    "agent_name": "DQN Agent (Default)",
    "timestamp": "2025-10-24 10:30:00",
    "agent_return": 15.43,
    "buy_hold_return": 12.56,
    "total_trades": 45
  }
]
```

**Status Codes:**
- `200 OK` - Results retrieved
- `500 Internal Server Error` - Database error

---

### Get Backtest Details

Get detailed results for a specific backtest run.

**Endpoint:** `GET /backtest_runs/{id}`

**Path Parameters:**
- `id` (integer) - Backtest run ID

**Response:**
```json
{
  "id": 1,
  "symbol": "AAPL",
  "agent_name": "DQN Agent (Default)",
  "test_period": "2023-08-01 to 2023-12-31",
  "agent_return": 15.43,
  "buy_hold_return": 12.56,
  "outperformance": 2.87,
  "total_trades": 45,
  "final_value": 11543.21,
  "trades": [...],
  "portfolio_history": [...],
  "portfolio_dates": [...]
}
```

**Status Codes:**
- `200 OK` - Details retrieved
- `404 Not Found` - Run ID not found
- `500 Internal Server Error` - Database error

---

### Upload Dataset

Upload custom CSV dataset for backtesting.

**Endpoint:** `POST /upload_dataset`

**Request Body:**
```json
{
  "name": "My Custom Data",
  "data": "date,open,high,low,close,volume\n2023-01-01,100,105,99,102,1000000\n..."
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Dataset name |
| `data` | string | Yes | CSV data as string |

**CSV Format:**
- Required columns: `date`, `open`, `high`, `low`, `close`, `volume`
- Date format: `YYYY-MM-DD`
- Numeric values for OHLCV

**Response:**
```json
{
  "id": 1,
  "name": "My Custom Data",
  "row_count": 252
}
```

**Status Codes:**
- `200 OK` - Dataset uploaded
- `400 Bad Request` - Invalid CSV format
- `500 Internal Server Error` - Upload failed

---

### Get Symbols

Get list of available stock symbols.

**Endpoint:** `GET /symbols`

**Response:**
```json
{
  "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
}
```

**Status Codes:**
- `200 OK` - Symbols retrieved

---

## Error Responses

All error responses follow this format:

```json
{
  "error": "ErrorType",
  "detail": "Detailed error message",
  "timestamp": "2025-10-24T12:00:00"
}
```

**Common Error Types:**
- `ValidationError` - Invalid input parameters
- `HTTPException` - HTTP-level errors
- `DataFetchError` - Failed to fetch market data
- `BacktestError` - Backtest execution failed
- `DatabaseError` - Database operation failed

---

## Rate Limiting

Default: 60 requests per minute per IP

Response headers when rate limited:
```
HTTP/1.1 429 Too Many Requests
Retry-After: 60
```

---

## CORS Policy

Allowed origins (configurable in `.env`):
- http://localhost:3000
- http://localhost:5500
- http://localhost:8000
- http://127.0.0.1:5500
- http://127.0.0.1:8000

---

## Webhook Support (Future)

Coming in v2.1:
- POST backtest completion notifications
- WebSocket for real-time updates
- Event streaming

---

## SDKs & Libraries

### Python SDK Example

```python
class NeuroQuantClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def run_backtest(self, symbol, **kwargs):
        response = requests.post(
            f"{self.base_url}/backtest",
            json={"symbol": symbol, **kwargs}
        )
        return response.json()
    
    def list_agents(self):
        response = requests.get(f"{self.base_url}/agents")
        return response.json()
    
    def create_agent(self, name, agent_type, parameters=None):
        response = requests.post(
            f"{self.base_url}/agents",
            json={"name": name, "type": agent_type, "parameters": parameters}
        )
        return response.json()

# Usage
client = NeuroQuantClient()
result = client.run_backtest("AAPL", train_split=0.7)
print(f"Return: {result['agent_return']}%")
```

### JavaScript SDK Example

```javascript
class NeuroQuantClient {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async runBacktest(symbol, options = {}) {
    const response = await fetch(`${this.baseUrl}/backtest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, ...options })
    });
    return response.json();
  }

  async listAgents() {
    const response = await fetch(`${this.baseUrl}/agents`);
    return response.json();
  }
}

// Usage
const client = new NeuroQuantClient();
const result = await client.runBacktest('AAPL', { train_split: 0.7 });
console.log(`Return: ${result.agent_return}%`);
```

---

## Best Practices

1. **Use appropriate train_split** (0.7-0.8 recommended)
2. **Handle errors gracefully** (check status codes)
3. **Cache agent IDs** (avoid recreating agents)
4. **Validate input** before sending requests
5. **Monitor rate limits** (implement backoff)
6. **Log requests** for debugging

---

## Version History

- **v2.0** (Current) - Full production release
- **v1.0** - MVP with basic features

---

For integration examples, see [Agent Development Guide](./AGENTS.md)
