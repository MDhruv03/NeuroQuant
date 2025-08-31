
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
        return self._next_observation(), {}  # Gymnasium returns (obs, info)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        
        if action == 1:  # Buy
            if self.portfolio > 0:
                self.holding += self.portfolio / current_price
                self.trades.append({'action': 'buy', 'step': self.current_step, 'price': current_price, 'shares': self.portfolio / current_price})
                self.portfolio = 0
        elif action == 2:  # Sell
            if self.holding > 0:
                self.portfolio += self.holding * current_price
                self.trades.append({'action': 'sell', 'step': self.current_step, 'price': current_price, 'shares': self.holding})
                self.holding = 0
            
        prev_value = self.portfolio + self.holding * self.df.iloc[self.current_step-1]['Close']
        current_value = self.portfolio + self.holding * current_price
        reward = (current_value - prev_value) / prev_value * 100
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        return self._next_observation(), reward, done, False, {}  # Gymnasium format: (obs, reward, terminated, truncated, info)
