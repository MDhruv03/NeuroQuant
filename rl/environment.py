
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, commission=0.001):
        super().__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.commission = commission  # 0.1% commission per trade
        self.current_step = 0
        self.trades = []
        self.portfolio_history = []
        self.portfolio_dates = []
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space - includes technical indicators + portfolio state
        self.feature_columns = [
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_signal', 
            'ADX', 'STOCH_k', 'STOCH_d', 'BB_upper', 'BB_middle', 'BB_lower', 
            'ATR', 'Volume_SMA', 'OBV', 'Price_to_SMA20', 'Price_to_SMA50', 
            'BB_position', 'Volume_ratio'
        ]
        # +3 for: cash ratio, holdings ratio, position (0=cash, 1=invested)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(self.feature_columns) + 3,), 
            dtype=np.float32
        )
        
        # Track portfolio state
        self.cash = initial_balance
        self.shares = 0
        self.buy_price = 0  # Track purchase price for PnL
        
    def _next_observation(self):
        """Get current observation with technical indicators and portfolio state"""
        features = self.df.iloc[self.current_step][self.feature_columns].values
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Calculate total portfolio value
        total_value = self.cash + self.shares * current_price
        
        # Portfolio state features
        cash_ratio = self.cash / total_value if total_value > 0 else 0
        holdings_ratio = (self.shares * current_price) / total_value if total_value > 0 else 0
        has_position = 1.0 if self.shares > 0 else 0.0
        
        obs = np.concatenate([
            features,
            np.array([cash_ratio, holdings_ratio, has_position], dtype=np.float32)
        ], dtype=np.float32)
        return obs
    
    def reset(self, **kwargs):
        """Reset environment to initial state"""
        self.current_step = 0
        self.cash = self.initial_balance
        self.shares = 0
        self.buy_price = 0
        self.trades = []
        
        # Initialize portfolio tracking
        self.portfolio_history = [self.initial_balance]
        self.portfolio_dates = [self.df.index[self.current_step].strftime('%Y-%m-%d')]
        
        return self._next_observation(), {}

    def step(self, action):
        """Execute one step in the environment"""
        current_price = self.df.iloc[self.current_step]['Close']
        current_date = self.df.index[self.current_step].strftime('%Y-%m-%d')
        
        # Calculate portfolio value before action
        prev_value = self.cash + self.shares * current_price
        
        reward = 0
        
        # Execute trading action
        if action == 1:  # Buy
            if self.cash > 0 and self.shares == 0:  # Only buy if we have cash and no position
                # Buy all available cash worth of shares
                cost = current_price * (1 + self.commission)
                shares_to_buy = self.cash / cost
                commission_paid = shares_to_buy * current_price * self.commission
                
                self.shares = shares_to_buy
                self.buy_price = current_price
                self.cash = 0
                
                self.trades.append({
                    'date': current_date,
                    'action': 'buy',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'commission': commission_paid,
                    'portfolio_value': prev_value
                })
                
                # Small penalty for trading (to avoid overtrading)
                reward -= 0.1
                
        elif action == 2:  # Sell
            if self.shares > 0:  # Only sell if we have shares
                # Sell all shares
                sale_value = self.shares * current_price
                commission_paid = sale_value * self.commission
                proceeds = sale_value - commission_paid
                
                # Calculate profit/loss
                cost_basis = self.shares * self.buy_price
                pnl = proceeds - cost_basis
                pnl_percent = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
                
                self.cash = proceeds
                
                self.trades.append({
                    'date': current_date,
                    'action': 'sell',
                    'price': current_price,
                    'shares': self.shares,
                    'commission': commission_paid,
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'portfolio_value': self.cash
                })
                
                self.shares = 0
                self.buy_price = 0
                
                # Reward based on profit/loss
                reward += pnl_percent / 10  # Scale reward
                
        # Calculate portfolio value after action
        current_value = self.cash + self.shares * current_price
        
        # Reward is based on portfolio value change
        if prev_value > 0:
            value_change = ((current_value - prev_value) / prev_value) * 100
            reward += value_change
        
        # Track portfolio value
        self.portfolio_history.append(current_value)
        self.portfolio_dates.append(current_date)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        return self._next_observation(), reward, done, False, {}
    
    @property
    def portfolio(self):
        """Get total portfolio value"""
        if self.current_step < len(self.df):
            current_price = self.df.iloc[self.current_step]['Close']
            return self.cash + self.shares * current_price
        return self.cash
