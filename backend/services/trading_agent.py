import numpy as np
import pandas as pd
from typing import Dict

from ..services.market_data import MarketDataProvider
from rl.agent import RLAgent, IndicatorAgent # Import IndicatorAgent
from rl.environment import TradingEnv

# ========================================
# BACKTESTING ENGINE
# ========================================

class BacktestEngine:
    """Simple backtesting engine"""
    
    def __init__(self, data_provider: MarketDataProvider):
        self.data_provider = data_provider
    
    def run_backtest(self, symbol: str, train_split: float = 0.7, agent_config: Dict = None) -> Dict:
        """Run backtest on historical data"""
        # Fetch and prepare data
        data = self.data_provider.fetch_stock_data(symbol)
        data['Return'] = data['Close'].pct_change()
        data.dropna(inplace=True)
        data = self.data_provider.calculate_technical_indicators(data)
        
        # Split data
        split_point = int(len(data) * train_split)
        train_data = data.iloc[:split_point]
        test_data = data.iloc[split_point:]

        # Instantiate and train agent
        print(f"Preparing agent on {len(train_data)} days...")
        train_env = TradingEnv(train_data)
        
        if agent_config and agent_config['type'] == 'IndicatorBased':
            # Rename parameters to match IndicatorAgent's constructor
            indicator_params = {}
            if 'short_sma' in agent_config['parameters']:
                indicator_params['short_sma_period'] = agent_config['parameters']['short_sma']
            if 'long_sma' in agent_config['parameters']:
                indicator_params['long_sma_period'] = agent_config['parameters']['long_sma']
            agent = IndicatorAgent(train_env, **indicator_params)
            print(f"Using Indicator-Based Agent: {agent_config['name']}")
        elif agent_config and agent_config['type'] == 'Random':
            # For a truly random agent, we might not even need an agent class,
            # but for consistency, we can have a simple one.
            # For now, we'll just use RLAgent as a placeholder for Random
            # and rely on its default random exploration if not trained.
            agent = RLAgent(train_env) # Placeholder for Random, will not train
            print(f"Using Random Agent: {agent_config['name']}")
        else: # Default to RLAgent (DQN/PPO)
            agent = RLAgent(train_env)
            print(f"Using RLAgent (DQN/PPO): {agent_config['name'] if agent_config else 'Default'}")
            agent.train() # Only train RLAgent
        
        # Test agent
        print(f"Testing agent on {len(test_data)} days...")
        test_env = TradingEnv(test_data)
        test_results = self._test_agent(agent, test_env)
        
        # Calculate metrics
        buy_hold_return = (test_data.iloc[-1]['Close'] / test_data.iloc[0]['Close'] - 1) * 100
        agent_return = ((test_results['final_value'] / test_results['initial_value']) - 1) * 100
        
        return {
            'symbol': symbol,
            'test_period': f"{test_data.index[0].date()} to {test_data.index[-1].date()}",
            'agent_return': agent_return,
            'buy_hold_return': buy_hold_return,
            'outperformance': agent_return - buy_hold_return,
            'total_trades': len(test_results['trades']),
            'final_value': test_results['final_value'],
            'trades': test_results['trades'],
            'portfolio_history': test_results['portfolio_history'],
            'portfolio_dates': test_results['portfolio_dates']
        }
    
    def _test_agent(self, agent: RLAgent, env: TradingEnv) -> Dict:
        """Test the trained agent"""
        obs, _ = env.reset()
        initial_value = env.portfolio
        done = False
        
        while not done:
            action, _ = agent.predict(obs)
            obs, _, done, _, _ = env.step(action)
            
        return {
            'initial_value': initial_value,
            'final_value': env.portfolio,
            'trades': env.trades,
            'portfolio_history': env.portfolio_history,
            'portfolio_dates': env.portfolio_dates
        }