from stable_baselines3 import PPO
import numpy as np
import pandas as pd # Import pandas for SMA calculation

class Agent:
    def __init__(self, env):
        self.env = env

    def train(self, timesteps=0):
        raise NotImplementedError

    def predict(self, obs):
        raise NotImplementedError

class RLAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        # PPO is a good default, but could be made configurable (e.g., DQN)
        self.model = PPO("MlpPolicy", env, verbose=0)  # Set verbose to 0 for less output

    def train(self, timesteps=10000):  # Reduced from 50000 for faster backtests
        print(f"Training RLAgent for {timesteps} timesteps...")
        self.model.learn(total_timesteps=timesteps)
        print("RLAgent training complete.")

    def save(self, path="models/ppo_trader"):
        self.model.save(path)

    def load(self, path="models/ppo_trader"):
        self.model = PPO.load(path)

    def predict(self, obs):
        action, _states = self.model.predict(obs, deterministic=True)  # Use deterministic for testing
        return action, _states

class IndicatorAgent(Agent):
    def __init__(self, env, short_sma_period=20, long_sma_period=50):
        super().__init__(env)
        self.short_sma_period = short_sma_period
        self.long_sma_period = long_sma_period
        self.df = env.df  # Access the dataframe from the environment

    def train(self, timesteps=0):
        print("IndicatorAgent does not require training.")
        pass  # No training needed for indicator-based agent

    def predict(self, obs):
        current_step = self.env.current_step
        
        if current_step < self.long_sma_period:  # Not enough data for SMAs
            return 0, None  # Hold

        # Get historical close prices up to the current step
        historical_prices = self.df['Close'].iloc[:current_step + 1]

        # Calculate SMAs
        short_sma = historical_prices.rolling(window=self.short_sma_period).mean().iloc[-1]
        long_sma = historical_prices.rolling(window=self.long_sma_period).mean().iloc[-1]

        # Check if we have a previous point for crossover
        if current_step < self.long_sma_period + 1:
            # Not enough data for crossover, use simple comparison
            if short_sma > long_sma and self.env.shares == 0:
                return 1, None  # Buy signal
            elif short_sma < long_sma and self.env.shares > 0:
                return 2, None  # Sell signal
            else:
                return 0, None  # Hold
        
        # Get previous SMAs for crossover detection
        prev_short_sma = historical_prices.rolling(window=self.short_sma_period).mean().iloc[-2]
        prev_long_sma = historical_prices.rolling(window=self.long_sma_period).mean().iloc[-2]

        # Crossover logic with position awareness
        if short_sma > long_sma and prev_short_sma <= prev_long_sma:
            # Golden cross - buy signal if not already invested
            if self.env.shares == 0:
                return 1, None
        elif short_sma < long_sma and prev_short_sma >= prev_long_sma:
            # Death cross - sell signal if invested
            if self.env.shares > 0:
                return 2, None
        
        return 0, None  # Hold