# Advanced Strategy Development Guide

## 📚 Building World-Class Trading Strategies with NeuroQuant

This guide walks you through creating sophisticated, production-ready trading strategies.

---

## Strategy Architecture

### 1. Simple Moving Average Crossover (Baseline)

```python
from rl.agent import BaseAgent
import numpy as np

class SMAStrategy(BaseAgent):
    """Simple Moving Average Crossover Strategy"""
    
    def __init__(self, short_period=20, long_period=50):
        self.short_period = short_period
        self.long_period = long_period
    
    def predict(self, observation):
        """Generate trading signal"""
        close_prices = observation['close']
        
        sma_short = np.mean(close_prices[-self.short_period:])
        sma_long = np.mean(close_prices[-self.long_period:])
        
        if sma_short > sma_long * 1.001:  # Golden cross
            return 1  # BUY
        elif sma_short < sma_long * 0.999:  # Death cross
            return 2  # SELL
        else:
            return 0  # HOLD
```

### 2. Multi-Indicator Strategy (Intermediate)

```python
class MultiIndicatorStrategy(BaseAgent):
    """Combines multiple technical indicators"""
    
    def __init__(self):
        self.indicators = {
            'rsi': 14,
            'macd': (12, 26, 9),
            'bb_period': 20,
        }
    
    def calculate_indicators(self, prices):
        """Calculate all technical indicators"""
        # RSI
        rsi = self.calculate_rsi(prices, self.indicators['rsi'])
        
        # MACD
        macd, signal = self.calculate_macd(prices, *self.indicators['macd'])
        
        # Bollinger Bands
        sma, std = self.calculate_bollinger_bands(prices, self.indicators['bb_period'])
        
        return {
            'rsi': rsi,
            'macd': macd,
            'signal': signal,
            'sma': sma,
            'std': std,
            'price': prices[-1]
        }
    
    def generate_signal(self, indicators):
        """Combine indicators into trading signal"""
        signals = []
        
        # RSI signals
        if indicators['rsi'] < 30:
            signals.append('oversold')
        elif indicators['rsi'] > 70:
            signals.append('overbought')
        
        # MACD signals
        if indicators['macd'] > indicators['signal']:
            signals.append('bullish')
        else:
            signals.append('bearish')
        
        # Bollinger Bands signals
        if indicators['price'] > indicators['sma'] + 2 * indicators['std']:
            signals.append('top_band')
        elif indicators['price'] < indicators['sma'] - 2 * indicators['std']:
            signals.append('bottom_band')
        
        # Combine signals
        if 'oversold' in signals and 'bullish' in signals:
            return 1  # Strong BUY
        elif 'overbought' in signals and 'bearish' in signals:
            return 2  # Strong SELL
        else:
            return 0  # HOLD
    
    def predict(self, observation):
        indicators = self.calculate_indicators(observation['close'])
        return self.generate_signal(indicators)
```

### 3. Ensemble Strategy (Advanced)

```python
from sklearn.ensemble import VotingClassifier
import numpy as np

class EnsembleStrategy(BaseAgent):
    """Combines multiple strategies using ensemble voting"""
    
    def __init__(self, strategies: list):
        self.strategies = strategies  # List of strategy instances
        self.weights = np.ones(len(strategies)) / len(strategies)
    
    def predict(self, observation):
        """Generate ensemble prediction"""
        predictions = []
        
        for strategy in self.strategies:
            pred, confidence = strategy.predict(observation)
            predictions.append({
                'action': pred,
                'confidence': confidence
            })
        
        # Weighted voting
        buy_votes = sum(p['confidence'] if p['action'] == 1 else 0 
                       for p in predictions)
        sell_votes = sum(p['confidence'] if p['action'] == 2 else 0 
                        for p in predictions)
        
        if buy_votes > sell_votes * 1.5:
            return 1, max(buy_votes / sum(w.values()), 0.5)
        elif sell_votes > buy_votes * 1.5:
            return 2, max(sell_votes / sum(w.values()), 0.5)
        else:
            return 0, 0
    
    def update_weights(self, performance):
        """Adapt weights based on strategy performance"""
        # Normalize performance to weights
        total_perf = sum(performance.values())
        self.weights = np.array([
            performance[s] / total_perf 
            for s in range(len(self.strategies))
        ])
```

### 4. Machine Learning Strategy (Expert)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class MLStrategy(BaseAgent):
    """Machine Learning-based trading strategy"""
    
    def __init__(self, model_path=None):
        if model_path:
            self.model = joblib.load(model_path)
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def extract_features(self, observation):
        """Extract machine learning features from market data"""
        features = []
        
        prices = observation['close']
        volumes = observation['volume']
        
        # Price features
        returns = np.diff(prices) / prices[:-1]
        features.append(np.mean(returns))
        features.append(np.std(returns))
        features.append(np.max(returns))
        features.append(np.min(returns))
        
        # Volume features
        vol_ma = np.mean(volumes[-20:])
        features.append(volumes[-1] / vol_ma)  # Volume ratio
        
        # Trend features
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:])
        features.append((prices[-1] - sma_20) / sma_20)
        features.append((sma_20 - sma_50) / sma_50)
        
        # Volatility
        returns_std = np.std(returns[-20:])
        features.append(returns_std)
        
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data, labels):
        """Train the model"""
        X = np.vstack([self.extract_features(obs) for obs in training_data])
        X = self.scaler.fit_transform(X)
        
        self.model.fit(X, labels)
        self.is_trained = True
    
    def predict(self, observation):
        """Generate prediction"""
        if not self.is_trained:
            return 0, 0  # No position until trained
        
        features = self.extract_features(observation)
        features = self.scaler.transform(features)
        
        # Get prediction and probability
        action = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = max(probabilities)
        
        return int(action), float(confidence)
```

### 5. Reinforcement Learning Strategy (Enterprise)

```python
from stable_baselines3 import PPO, DQN
import gymnasium as gym

class RLStrategy(BaseAgent):
    """Deep Reinforcement Learning strategy using Stable-Baselines3"""
    
    def __init__(self, algorithm='ppo', model_path=None):
        self.algorithm = algorithm
        if model_path:
            if algorithm == 'ppo':
                self.model = PPO.load(model_path)
            else:
                self.model = DQN.load(model_path)
        else:
            self.model = None
    
    def train(self, env, timesteps=100000, callback=None):
        """Train RL agent"""
        if self.algorithm == 'ppo':
            self.model = PPO(
                'MlpPolicy',
                env,
                learning_rate=1e-3,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1
            )
        else:
            self.model = DQN(
                'MlpPolicy',
                env,
                learning_rate=1e-3,
                buffer_size=50000,
                learning_starts=1000,
                target_update_interval=1000,
                train_freq=4,
                gamma=0.99,
                exploration_fraction=0.1,
                verbose=1
            )
        
        self.model.learn(total_timesteps=timesteps, callback=callback)
    
    def predict(self, observation):
        """Generate RL prediction"""
        action, _states = self.model.predict(observation, deterministic=True)
        return int(action), 0  # Confidence not directly available
    
    def save(self, path):
        """Save trained model"""
        self.model.save(path)
```

---

## Risk Management Rules

### Position Sizing

```python
def calculate_position_size(
    portfolio_value: float,
    risk_per_trade: float = 0.02,
    stop_loss_pct: float = 0.05,
    entry_price: float = 100
) -> float:
    """
    Calculate optimal position size using Kelly Criterion
    
    Args:
        portfolio_value: Total portfolio value
        risk_per_trade: Risk as % of portfolio (e.g., 0.02 = 2%)
        stop_loss_pct: Stop loss distance as % (e.g., 0.05 = 5%)
        entry_price: Entry price of the trade
    
    Returns:
        Number of shares to buy
    """
    risk_amount = portfolio_value * risk_per_trade
    loss_per_share = entry_price * stop_loss_pct
    shares = risk_amount / loss_per_share
    return max(1, int(shares))
```

### Stop Loss & Take Profit

```python
class RiskManager:
    """Manage stop-loss and take-profit levels"""
    
    @staticmethod
    def set_stops(
        entry_price: float,
        position_type: str = 'long',
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10
    ) -> dict:
        """Calculate stop-loss and take-profit levels"""
        
        if position_type == 'long':
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
        else:  # short
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)
        
        return {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': take_profit_pct / stop_loss_pct
        }
    
    @staticmethod
    def trailing_stop(
        current_price: float,
        max_price: float,
        trailing_pct: float = 0.03
    ) -> float:
        """Calculate trailing stop-loss"""
        return max_price * (1 - trailing_pct)
```

---

## Backtesting Framework

```python
from datetime import datetime, timedelta
import pandas as pd

class BacktestRunner:
    """Run comprehensive backtests"""
    
    def __init__(self, strategy, initial_capital=100000):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.results = None
    
    def run(
        self,
        data: pd.DataFrame,
        start_date: datetime = None,
        end_date: datetime = None,
        commission: float = 0.001
    ) -> dict:
        """Run backtest"""
        
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        shares = 0
        trades = []
        portfolio_history = []
        
        for idx, row in data.iterrows():
            observation = {
                'close': data['close'].values[:idx+1],
                'volume': data['volume'].values[:idx+1],
                'high': data['high'].values[:idx+1],
                'low': data['low'].values[:idx+1],
            }
            
            action, confidence = self.strategy.predict(observation)
            current_price = row['close']
            
            # Execute action
            if action == 1 and shares == 0:  # BUY
                shares = cash / (current_price * (1 + commission))
                cash = 0
                trades.append({
                    'date': idx,
                    'type': 'buy',
                    'price': current_price,
                    'shares': shares
                })
            
            elif action == 2 and shares > 0:  # SELL
                cash = shares * current_price * (1 - commission)
                trades.append({
                    'date': idx,
                    'type': 'sell',
                    'price': current_price,
                    'shares': shares,
                    'pnl': cash - self.initial_capital
                })
                shares = 0
            
            # Record portfolio value
            portfolio_value = cash + (shares * current_price)
            portfolio_history.append(portfolio_value)
        
        # Calculate metrics
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital
        
        return {
            'total_return': total_return,
            'final_value': portfolio_value,
            'trades': trades,
            'portfolio_history': portfolio_history,
            'num_trades': len(trades),
            'win_rate': len([t for t in trades if t.get('pnl', 0) > 0]) / max(len(trades), 1)
        }
```

---

## Performance Optimization

```python
def optimize_strategy_parameters(
    strategy_class,
    data: pd.DataFrame,
    param_grid: dict
):
    """Hyperparameter optimization using grid search"""
    from itertools import product
    
    best_params = None
    best_return = -np.inf
    
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        
        strategy = strategy_class(**param_dict)
        backtester = BacktestRunner(strategy)
        results = backtester.run(data)
        
        if results['total_return'] > best_return:
            best_return = results['total_return']
            best_params = param_dict
    
    return best_params, best_return


# Example usage
param_grid = {
    'short_period': [10, 15, 20, 25],
    'long_period': [40, 50, 60, 70]
}

best_params, best_return = optimize_strategy_parameters(
    SMAStrategy,
    historical_data,
    param_grid
)
print(f"Best params: {best_params}")
print(f"Best return: {best_return:.2%}")
```

---

## Deploy Your Strategy

```bash
# 1. Create strategy file
cat > strategies/my_strategy.py << 'EOF'
from rl.agent import BaseAgent

class MyStrategy(BaseAgent):
    def predict(self, observation):
        # Your strategy logic here
        return 0  # HOLD
EOF

# 2. Test your strategy
python -m pytest tests/strategies/test_my_strategy.py

# 3. Register strategy
python scripts/register_strategy.py --name "My Strategy" --file strategies/my_strategy.py

# 4. Backtest
curl -X POST http://localhost:8000/api/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": "my_strategy",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 100000
  }'

# 5. Deploy to paper trading
curl -X POST http://localhost:8000/api/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": "my_strategy",
    "mode": "paper"
  }'

# 6. Monitor live trading
curl http://localhost:8000/api/strategies/my_strategy/status
```

---

**Happy Trading! 📈**
