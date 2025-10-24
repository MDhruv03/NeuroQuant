# NeuroQuant API Documentation

## Base URL
```
http://localhost:8000
```

## Authentication
Currently, the API does not require authentication. In production, implement API keys or JWT tokens.

---

## Endpoints

### Health Check

#### `GET /health`
Check the system status and configuration.

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

---

### Backtesting

#### `POST /backtest`
Run a backtest simulation.

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
- `symbol` (string, required): Stock ticker symbol
- `train_split` (float, optional): Train/test split ratio (0.1-0.9, default: 0.7)
- `agent_id` (integer, optional): Agent ID to use (null for default)
- `data_source` (string, required): "yfinance" or "custom"
- `custom_dataset_id` (integer, optional): ID of custom dataset if using custom data
- `start_date` (string, optional): Start date for data (YYYY-MM-DD)
- `end_date` (string, optional): End date for data (YYYY-MM-DD)

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
  "trades": [...],
  "portfolio_history": [10000, 10100, ...],
  "portfolio_dates": ["2023-08-01", "2023-08-02", ...],
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

---

### Agents

#### `GET /agents`
List all registered trading agents.

**Response:**
```json
[
  {
    "id": 1,
    "name": "DQN Agent (Default)",
    "type": "DQN",
    "parameters": {
      "learning_rate": 0.001,
      "gamma": 0.99
    },
    "description": "Deep Q-Network agent",
    "created_at": "2025-10-24T12:00:00"
  },
  ...
]
```

#### `POST /agents`
Create a new trading agent.

**Request Body:**
```json
{
  "name": "My Custom Agent",
  "type": "DQN",
  "parameters": {
    "learning_rate": 0.0005,
    "gamma": 0.95,
    "epsilon_start": 1.0
  },
  "description": "Custom DQN agent with modified hyperparameters"
}
```

**Parameters:**
- `name` (string, required): Agent name (3-100 characters)
- `type` (string, required): Agent type ("DQN", "PPO", "IndicatorBased", "Random")
- `parameters` (object, required): Agent-specific parameters
- `description` (string, optional): Agent description

**Response:**
```json
{
  "id": 5,
  "name": "My Custom Agent",
  "type": "DQN",
  "parameters": {...},
  "description": "Custom DQN agent",
  "created_at": "2025-10-24T12:30:00"
}
```

---

### Backtest History

#### `GET /backtest_runs`
Get all historical backtest runs.

**Response:**
```json
[
  {
    "id": 1,
    "timestamp": "2025-10-24T12:00:00",
    "symbol": "AAPL",
    "agent_id": 1,
    "agent_name": "DQN Agent",
    "test_period": "2023-08-01 to 2023-12-31",
    "agent_return": 15.43,
    "buy_hold_return": 12.56,
    "outperformance": 2.87,
    "total_trades": 45,
    "final_value": 11543.21,
    "trades": [...],
    "portfolio_history": [...],
    "portfolio_dates": [...],
    "metrics": {...}
  },
  ...
]
```

#### `GET /backtest_runs/{run_id}`
Get details of a specific backtest run.

**Parameters:**
- `run_id` (integer, required): Backtest run ID

**Response:** Same as single item from `/backtest_runs`

---

### Symbols

#### `GET /symbols`
Get available trading symbols and custom datasets.

**Response:**
```json
{
  "yfinance_symbols": [
    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", ...
  ],
  "custom_datasets": [
    {
      "id": 1,
      "name": "Custom: My Dataset"
    },
    ...
  ]
}
```

---

### Custom Datasets

#### `POST /upload_dataset`
Upload a custom CSV dataset for backtesting.

**Request (multipart/form-data):**
- `name` (string, required): Dataset name
- `file` (file, required): CSV file with columns: Date, Open, High, Low, Close, Volume
- `description` (string, optional): Dataset description

**CSV Format:**
```csv
Date,Open,High,Low,Close,Volume
2023-01-01,100.0,105.0,99.0,103.5,1000000
2023-01-02,103.5,108.0,102.0,107.2,1200000
...
```

**Response:**
```json
{
  "message": "Dataset 'My Dataset' uploaded successfully!",
  "id": 1
}
```

#### `GET /custom_datasets`
List all custom datasets.

**Response:**
```json
[
  {
    "id": 1,
    "name": "My Dataset",
    "description": "Custom stock data",
    "created_at": "2025-10-24T12:00:00",
    "row_count": 365
  },
  ...
]
```

---

## Error Responses

All endpoints may return these error formats:

### Validation Error (422)
```json
{
  "error": "ValidationError",
  "detail": "Field 'symbol' is required",
  "timestamp": "2025-10-24T12:00:00"
}
```

### Not Found (404)
```json
{
  "error": "HTTPException",
  "detail": "Agent with ID 999 not found",
  "timestamp": "2025-10-24T12:00:00"
}
```

### Server Error (500)
```json
{
  "error": "DataFetchError",
  "detail": "Failed to fetch data for symbol XYZ",
  "timestamp": "2025-10-24T12:00:00"
}
```

---

## Rate Limiting

Default rate limit: 60 requests per minute per IP address.

Exceeding the limit returns:
```json
{
  "error": "RateLimitExceeded",
  "detail": "Rate limit exceeded. Please try again later."
}
```

---

## WebSocket Support (Coming Soon)

Real-time updates for training progress and live trading will be available via WebSocket connections.

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/training');
ws.onmessage = (event) => {
  const progress = JSON.parse(event.data);
  console.log(progress);
};
```

---

## Code Examples

### Python
```python
import requests

# Run backtest
response = requests.post(
    'http://localhost:8000/backtest',
    json={
        'symbol': 'AAPL',
        'train_split': 0.7,
        'agent_id': 1
    }
)
result = response.json()
print(f"Agent Return: {result['agent_return']}%")
```

### JavaScript
```javascript
// Create agent
fetch('http://localhost:8000/agents', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    name: 'My Agent',
    type: 'DQN',
    parameters: {learning_rate: 0.001}
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

### cURL
```bash
# Get agents
curl http://localhost:8000/agents

# Run backtest
curl -X POST http://localhost:8000/backtest \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","train_split":0.7}'
```

---

## Interactive Documentation

For interactive API testing, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
