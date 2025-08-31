
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.current_step = 0
        self.trades = []
        self.portfolio_history = [] # To track portfolio value over time
        self.portfolio_dates = [] # To track dates corresponding to portfolio history
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space
        self.feature_columns = [
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_signal', 'ADX', 'STOCH_k', 'STOCH_d', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'Volume_SMA', 'OBV', 'Price_to_SMA20', 'Price_to_SMA50', 'BB_position', 'Volume_ratio'
        ]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.feature_columns) + 2,), dtype=np.float32)
        
        # Track portfolio
        self.portfolio = 10000  # Starting cash
        self.holding = 0
        
    def _next_observation(self):
        features = self.df.iloc[self.current_step][self.feature_columns].values
        obs = np.concatenate([
            features,
            np.array([self.portfolio / 10000, self.holding / 1000])
        ], dtype=np.float32)
        return obs
    
    def reset(self, **kwargs):
        self.current_step = 0
        self.portfolio = 10000
        self.holding = 0
        self.trades = []
        self.portfolio_history = [self.portfolio] # Initialize with starting portfolio value
        self.portfolio_dates = [self.df.index[self.current_step].strftime('%Y-%m-%d')] # Initialize with starting date
        return self._next_observation(), {}  # Gymnasium returns (obs, info)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        current_date = self.df.index[self.current_step].strftime('%Y-%m-%d')
        
        if action == 1:  # Buy
            if self.portfolio > 0:
                shares_to_buy = self.portfolio / current_price
                self.holding += shares_to_buy
                self.trades.append({
                    'date': current_date,
                    'action': 'buy',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'portfolio_before': self.portfolio,
                    'holding_before': self.holding - shares_to_buy # Shares before this buy
                })
                self.portfolio = 0
        elif action == 2:  # Sell
            if self.holding > 0:
                # Assuming all holdings are sold for simplicity
                shares_to_sell = self.holding
                pnl = (current_price - self.trades[-1]['price']) * shares_to_sell if self.trades and self.trades[-1]['action'] == 'buy' else 0 # Simplified PnL
                self.portfolio += shares_to_sell * current_price
                self.trades.append({
                    'date': current_date,
                    'action': 'sell',
                    'price': current_price,
                    'shares': shares_to_sell,
                    'pnl': pnl,
                    'portfolio_before': self.portfolio - (shares_to_sell * current_price),
                    'holding_before': self.holding
                })
                self.holding = 0
            
        prev_value = self.portfolio + self.holding * self.df.iloc[self.current_step-1]['Close']
        current_value = self.portfolio + self.holding * current_price
        reward = (current_value - prev_value) / prev_value * 100
        
        self.portfolio_history.append(current_value) # Track portfolio value at each step
        self.portfolio_dates.append(current_date) # Track date corresponding to portfolio value
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        return self._next_observation(), reward, done, False, {}  # Gymnasium format: (obs, reward, terminated, truncated, info)
