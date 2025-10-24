# Agent Development Guide

Complete guide for developing, training, and customizing trading agents in NeuroQuant.

---

## Table of Contents

1. [Agent Architecture](#agent-architecture)
2. [Built-in Agents](#built-in-agents)
3. [Creating Custom Agents](#creating-custom-agents)
4. [Training Agents](#training-agents)
5. [Backtesting](#backtesting)
6. [Advanced Techniques](#advanced-techniques)

---

## Agent Architecture

### Base Agent Interface

All agents inherit from a common base structure:

```python
class BaseAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
    def predict(self, observation):
        """Make trading decision based on observation"""
        raise NotImplementedError
        
    def train(self):
        """Train the agent"""
        raise NotImplementedError
        
    def save(self, path):
        """Save agent to disk"""
        raise NotImplementedError
        
    def load(self, path):
        """Load agent from disk"""
        raise NotImplementedError
```

### Agent Types

NeuroQuant supports three types of agents:

1. **Deep Reinforcement Learning (DQN, PPO)**
   - Learn optimal policies through trial and error
   - Adapt to market conditions
   - Require training on historical data

2. **Indicator-Based Strategies**
   - Rule-based trading using technical indicators
   - No training required
   - Transparent decision-making

3. **Hybrid Agents**
   - Combine RL and indicators
   - Best of both worlds
   - Enhanced performance

---

## Built-in Agents

### 1. DQN Agent

Deep Q-Network agent using neural networks.

**File**: `rl/dqn_agent.py`

**Features**:
- Experience replay buffer
- Target network for stability
- Epsilon-greedy exploration
- Custom reward shaping

**Usage**:
```python
from rl.dqn_agent import DQNAgent
from rl.environment import TradingEnvironment

# Create environment
env = TradingEnvironment(data, initial_balance=100000)

# Initialize agent
agent = DQNAgent(
    env=env,
    learning_rate=0.0001,
    buffer_size=100000,
    batch_size=32
)

# Train
agent.train(total_timesteps=100000)

# Save
agent.save("models/dqn_aapl.zip")
```

**Hyperparameters**:
```python
{
    "learning_rate": 0.0001,
    "buffer_size": 100000,
    "batch_size": 32,
    "learning_starts": 1000,
    "target_update_interval": 1000,
    "exploration_fraction": 0.1,
    "exploration_final_eps": 0.05,
    "gamma": 0.99,
    "tau": 0.005
}
```

---

### 2. PPO Agent

Proximal Policy Optimization agent.

**File**: `rl/agent.py`

**Features**:
- On-policy learning
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Multiple parallel environments

**Usage**:
```python
from rl.agent import PPOAgent

agent = PPOAgent(
    env=env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64
)

agent.train(total_timesteps=100000)
agent.save("models/ppo_aapl.zip")
```

**Hyperparameters**:
```python
{
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

---

### 3. Indicator-Based Agent

Rule-based agent using technical indicators.

**File**: `services/trading_agent.py`

**Features**:
- SMA crossover strategy
- RSI overbought/oversold signals
- MACD trend following
- Bollinger Bands mean reversion

**Usage**:
```python
from services.trading_agent import IndicatorAgent

agent = IndicatorAgent(
    sma_short=50,
    sma_long=200,
    rsi_period=14,
    rsi_overbought=70,
    rsi_oversold=30
)

# Get trading signal
signal = agent.predict(market_data)
# Returns: 1 (buy), -1 (sell), 0 (hold)
```

**Strategy Logic**:
```python
def predict(self, data):
    # SMA Crossover
    if sma_short > sma_long:
        signal = 1  # Buy
    elif sma_short < sma_long:
        signal = -1  # Sell
    
    # RSI Filter
    if rsi > 70:
        signal = -1  # Overbought, sell
    elif rsi < 30:
        signal = 1  # Oversold, buy
    
    # MACD Confirmation
    if macd > signal_line:
        signal = 1  # Bullish
    elif macd < signal_line:
        signal = -1  # Bearish
    
    return signal
```

---

## Creating Custom Agents

### Step 1: Define Agent Class

```python
# custom_agents/my_agent.py
from rl.agent import BaseAgent
import numpy as np

class MyCustomAgent(BaseAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, kwargs)
        self.model = self._build_model()
        
    def _build_model(self):
        """Build your custom model architecture"""
        # Example: Simple neural network
        from tensorflow.keras import Sequential, layers
        
        model = Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.env.observation_space.shape[0],)),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.env.action_space.n, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def predict(self, observation, deterministic=True):
        """Make prediction"""
        q_values = self.model.predict(observation.reshape(1, -1))
        action = np.argmax(q_values)
        return action, None
    
    def train(self, total_timesteps):
        """Custom training loop"""
        for timestep in range(total_timesteps):
            observation = self.env.reset()
            done = False
            
            while not done:
                action, _ = self.predict(observation, deterministic=False)
                next_obs, reward, done, info = self.env.step(action)
                
                # Update model
                target = reward + 0.99 * np.max(self.model.predict(next_obs.reshape(1, -1)))
                target_vec = self.model.predict(observation.reshape(1, -1))
                target_vec[0][action] = target
                
                self.model.fit(observation.reshape(1, -1), target_vec, epochs=1, verbose=0)
                
                observation = next_obs
                
            if timestep % 1000 == 0:
                print(f"Timestep: {timestep}")
    
    def save(self, path):
        """Save model"""
        self.model.save(path)
    
    def load(self, path):
        """Load model"""
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
```

### Step 2: Register Agent

```python
# services/agent_manager.py
from custom_agents.my_agent import MyCustomAgent

AGENT_REGISTRY = {
    "dqn": DQNAgent,
    "ppo": PPOAgent,
    "indicator": IndicatorAgent,
    "custom": MyCustomAgent  # Add your agent
}

def create_agent(agent_type, env, **kwargs):
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return AGENT_REGISTRY[agent_type](env, **kwargs)
```

### Step 3: Use Custom Agent

```python
from services.agent_manager import create_agent
from rl.environment import TradingEnvironment

# Create environment
env = TradingEnvironment(data)

# Create custom agent
agent = create_agent("custom", env, learning_rate=0.001)

# Train
agent.train(total_timesteps=50000)

# Save
agent.save("models/custom_agent.h5")
```

---

## Training Agents

### Training Environment

The trading environment simulates market conditions:

```python
# rl/environment.py
class TradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=100000, commission=0.001):
        self.data = data
        self.initial_balance = initial_balance
        self.commission = commission
        
        # State: [prices, indicators, portfolio]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,)
        )
        
        # Actions: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
    
    def step(self, action):
        # Execute action
        if action == 1:  # Buy
            self._execute_buy()
        elif action == 2:  # Sell
            self._execute_sell()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Next observation
        self.current_step += 1
        done = self.current_step >= len(self.data)
        obs = self._get_observation()
        
        return obs, reward, done, {}
    
    def _calculate_reward(self):
        # Portfolio value change
        current_value = self.balance + self.shares * self.current_price
        prev_value = self.balance + self.shares * self.prev_price
        
        return (current_value - prev_value) / prev_value
```

### Training Script

```python
# scripts/train_agent.py
import argparse
from rl.train import train_agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="dqn")
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--start-date", type=str, default="2020-01-01")
    parser.add_argument("--end-date", type=str, default="2023-12-31")
    parser.add_argument("--timesteps", type=int, default=100000)
    args = parser.parse_args()
    
    # Train
    agent = train_agent(
        agent_type=args.agent,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        total_timesteps=args.timesteps
    )
    
    # Save
    agent.save(f"models/{args.agent}_{args.symbol}.zip")
    print(f"Agent saved to models/{args.agent}_{args.symbol}.zip")

if __name__ == "__main__":
    main()
```

**Run training**:
```bash
python scripts/train_agent.py --agent dqn --symbol AAPL --timesteps 100000
```

---

## Backtesting

### Via API

```bash
curl -X POST "http://localhost:8000/backtest" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "dqn",
    "symbol": "AAPL",
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "initial_balance": 100000
  }'
```

### Via Frontend

1. Navigate to **http://localhost:8000/backtest.html**
2. Select agent type
3. Enter symbol and date range
4. Click "Run Backtest"
5. View results (returns, trades, metrics)

### Programmatically

```python
from services.trading_agent import backtest_agent

results = backtest_agent(
    agent_type="dqn",
    symbol="AAPL",
    start_date="2020-01-01",
    end_date="2023-12-31",
    initial_balance=100000
)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

---

## Advanced Techniques

### 1. Reward Shaping

Customize reward function for better learning:

```python
def _calculate_reward(self):
    # Base reward: portfolio value change
    portfolio_return = (self.portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
    
    # Penalty for excessive trading
    trading_penalty = -0.01 if self.action != 0 else 0
    
    # Bonus for positive Sharpe ratio
    sharpe_bonus = 0.1 if self.sharpe_ratio > 1.0 else 0
    
    # Penalty for drawdown
    drawdown_penalty = -0.05 if self.drawdown > 0.2 else 0
    
    return portfolio_return + trading_penalty + sharpe_bonus + drawdown_penalty
```

### 2. Feature Engineering

Add custom features to observation space:

```python
def _get_observation(self):
    # Price features
    prices = self.data['close'].iloc[self.current_step-self.window:self.current_step].values
    
    # Technical indicators
    sma = talib.SMA(prices, timeperiod=20)
    rsi = talib.RSI(prices, timeperiod=14)
    macd, signal, _ = talib.MACD(prices)
    
    # Sentiment
    sentiment_score = self.sentiment_analyzer.predict(self.news[self.current_step])
    
    # Portfolio state
    portfolio_state = [
        self.balance / self.initial_balance,
        self.shares / 1000,
        self.unrealized_pnl / self.initial_balance
    ]
    
    # Combine all features
    observation = np.concatenate([
        prices, sma, rsi, macd, signal, [sentiment_score], portfolio_state
    ])
    
    return observation
```

### 3. Ensemble Agents

Combine multiple agents for robust predictions:

```python
class EnsembleAgent:
    def __init__(self, agents):
        self.agents = agents
    
    def predict(self, observation):
        # Get predictions from all agents
        predictions = [agent.predict(observation)[0] for agent in self.agents]
        
        # Majority voting
        action = max(set(predictions), key=predictions.count)
        
        return action, None

# Usage
dqn_agent = DQNAgent(env)
ppo_agent = PPOAgent(env)
indicator_agent = IndicatorAgent()

ensemble = EnsembleAgent([dqn_agent, ppo_agent, indicator_agent])
action, _ = ensemble.predict(observation)
```

### 4. Transfer Learning

Transfer knowledge from one symbol to another:

```python
# Train on AAPL
agent = DQNAgent(env_aapl)
agent.train(total_timesteps=100000)
agent.save("models/dqn_aapl.zip")

# Fine-tune on MSFT
agent.load("models/dqn_aapl.zip")
agent.env = env_msft
agent.train(total_timesteps=20000)  # Fewer steps needed
agent.save("models/dqn_msft.zip")
```

### 5. Hyperparameter Tuning

Optimize agent hyperparameters:

```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    "learning_rate": [0.0001, 0.0003, 0.001],
    "batch_size": [16, 32, 64],
    "gamma": [0.95, 0.99, 0.999]
}

best_score = -np.inf
best_params = None

for params in ParameterGrid(param_grid):
    agent = DQNAgent(env, **params)
    agent.train(total_timesteps=50000)
    
    results = backtest_agent(agent, symbol="AAPL")
    score = results['sharpe_ratio']
    
    if score > best_score:
        best_score = score
        best_params = params
        agent.save("models/best_agent.zip")

print(f"Best params: {best_params}")
print(f"Best Sharpe: {best_score:.2f}")
```

---

## See Also

- [Configuration Guide](CONFIGURATION.md)
- [API Reference](API.md)
- [Architecture Overview](ARCHITECTURE.md)
- [Testing Guide](TESTING.md)
