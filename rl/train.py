
import yfinance as yf
import pandas as pd
from .environment import TradingEnv
from .agent import RLAgent
import os

# Download data
def get_data(ticker="SPY", start="2023-01-01", end="2024-01-01"):
    df = yf.download(ticker, start=start, end=end)
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    return df

# Train RL agent
def train_model():
    df = get_data()
    env = TradingEnv(df)
    agent = RLAgent(env)
    
    agent.train()
    
    os.makedirs("models", exist_ok=True)
    agent.save()

if __name__ == "__main__":
    train_model()
