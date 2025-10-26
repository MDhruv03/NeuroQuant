"""
Portfolio Management System
Handles multi-asset portfolios, position sizing, and risk management
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import yfinance as yf


class PortfolioManager:
    """Advanced portfolio management with risk controls"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Dict] = {}  # symbol -> {shares, avg_price, current_price}
        self.trade_history: List[Dict] = []
        self.portfolio_values: List[Tuple[datetime, float]] = []
        
    def calculate_position_size(self, 
                                symbol: str, 
                                signal_strength: float,
                                risk_per_trade: float = 0.02,
                                max_position_size: float = 0.25) -> float:
        """
        Calculate optimal position size based on Kelly Criterion and risk management
        
        Args:
            symbol: Stock symbol
            signal_strength: Agent's confidence (0-1)
            risk_per_trade: Maximum risk per trade (default 2%)
            max_position_size: Maximum position as % of portfolio (default 25%)
        """
        # Risk-based position sizing
        risk_amount = self.current_capital * risk_per_trade
        
        # Adjust by signal strength
        position_size = risk_amount * signal_strength
        
        # Apply maximum position constraint
        max_allowed = self.current_capital * max_position_size
        position_size = min(position_size, max_allowed)
        
        return position_size
    
    def execute_trade(self, 
                     symbol: str, 
                     action: str, 
                     price: float,
                     signal_strength: float = 1.0,
                     stop_loss_pct: float = 0.05) -> Dict:
        """
        Execute a trade with risk management
        
        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            price: Current price
            signal_strength: Confidence level (0-1)
            stop_loss_pct: Stop loss percentage (default 5%)
        """
        timestamp = datetime.now()
        
        if action == 'BUY':
            position_size = self.calculate_position_size(symbol, signal_strength)
            shares = position_size / price
            
            if shares > 0 and position_size <= self.current_capital:
                self.current_capital -= position_size
                
                if symbol in self.positions:
                    # Average down/up
                    existing = self.positions[symbol]
                    total_shares = existing['shares'] + shares
                    avg_price = ((existing['shares'] * existing['avg_price']) + 
                               (shares * price)) / total_shares
                    self.positions[symbol] = {
                        'shares': total_shares,
                        'avg_price': avg_price,
                        'current_price': price,
                        'stop_loss': price * (1 - stop_loss_pct),
                        'entry_time': existing['entry_time']
                    }
                else:
                    self.positions[symbol] = {
                        'shares': shares,
                        'avg_price': price,
                        'current_price': price,
                        'stop_loss': price * (1 - stop_loss_pct),
                        'entry_time': timestamp
                    }
                
                trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares,
                    'price': price,
                    'value': position_size,
                    'signal_strength': signal_strength
                }
                self.trade_history.append(trade)
                return trade
        
        elif action == 'SELL':
            if symbol in self.positions:
                position = self.positions[symbol]
                shares = position['shares']
                sell_value = shares * price
                
                self.current_capital += sell_value
                profit_loss = (price - position['avg_price']) * shares
                profit_loss_pct = (price - position['avg_price']) / position['avg_price']
                
                trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares,
                    'price': price,
                    'value': sell_value,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss_pct,
                    'hold_duration': (timestamp - position['entry_time']).days
                }
                self.trade_history.append(trade)
                
                del self.positions[symbol]
                return trade
        
        return {}
    
    def check_stop_losses(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Check and execute stop losses"""
        stop_loss_trades = []
        
        for symbol, position in list(self.positions.items()):
            current_price = current_prices.get(symbol, position['current_price'])
            
            if current_price <= position['stop_loss']:
                trade = self.execute_trade(symbol, 'SELL', current_price)
                trade['reason'] = 'STOP_LOSS'
                stop_loss_trades.append(trade)
        
        return stop_loss_trades
    
    def update_positions(self, current_prices: Dict[str, float]):
        """Update current prices for all positions"""
        for symbol in self.positions:
            if symbol in current_prices:
                self.positions[symbol]['current_price'] = current_prices[symbol]
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        self.update_positions(current_prices)
        
        positions_value = sum(
            pos['shares'] * pos['current_price'] 
            for pos in self.positions.values()
        )
        
        total_value = self.current_capital + positions_value
        self.portfolio_values.append((datetime.now(), total_value))
        
        return total_value
    
    def get_portfolio_allocation(self) -> Dict[str, float]:
        """Get current portfolio allocation by symbol"""
        total_value = self.current_capital + sum(
            pos['shares'] * pos['current_price'] 
            for pos in self.positions.values()
        )
        
        allocation = {
            'CASH': self.current_capital / total_value
        }
        
        for symbol, pos in self.positions.items():
            position_value = pos['shares'] * pos['current_price']
            allocation[symbol] = position_value / total_value
        
        return allocation
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trade_history:
            return {}
        
        closed_trades = [t for t in self.trade_history if t['action'] == 'SELL']
        
        if not closed_trades:
            return {'total_trades': len(self.trade_history)}
        
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        profits = [t['profit_loss'] for t in closed_trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else 0
        
        returns = [t['profit_loss_pct'] for t in closed_trades]
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 else 0
        
        return {
            'total_trades': len(closed_trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'largest_win': max(profits) if profits else 0,
            'largest_loss': min(profits) if profits else 0
        }


class MultiSymbolBacktester:
    """Backtest multiple symbols simultaneously"""
    
    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        
    def load_data(self):
        """Load historical data for all symbols"""
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=self.start_date, end=self.end_date)
                self.data[symbol] = df
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
    
    def run_backtest(self, 
                    agent_dict: Dict,
                    initial_capital: float = 10000.0) -> Dict:
        """
        Run backtest across multiple symbols
        
        Args:
            agent_dict: Dictionary of {symbol: agent} pairs
            initial_capital: Starting capital
        """
        portfolio = PortfolioManager(initial_capital)
        
        # Get all unique dates
        all_dates = sorted(set(
            date for df in self.data.values() 
            for date in df.index
        ))
        
        for date in all_dates:
            current_prices = {}
            
            # Get current prices
            for symbol, df in self.data.items():
                if date in df.index:
                    current_prices[symbol] = df.loc[date, 'Close']
            
            # Check stop losses first
            portfolio.check_stop_losses(current_prices)
            
            # Generate signals for each symbol
            for symbol, agent in agent_dict.items():
                if symbol not in self.data or date not in self.data[symbol].index:
                    continue
                
                # Get agent signal (this is simplified - integrate with actual RL agent)
                signal = self._get_agent_signal(agent, symbol, date)
                
                if signal['action'] == 'BUY':
                    portfolio.execute_trade(
                        symbol, 
                        'BUY', 
                        current_prices[symbol],
                        signal['strength']
                    )
                elif signal['action'] == 'SELL' and symbol in portfolio.positions:
                    portfolio.execute_trade(
                        symbol, 
                        'SELL', 
                        current_prices[symbol]
                    )
            
            # Record portfolio value
            portfolio.get_portfolio_value(current_prices)
        
        # Close all positions at end
        final_prices = {
            symbol: df.iloc[-1]['Close'] 
            for symbol, df in self.data.items()
        }
        
        for symbol in list(portfolio.positions.keys()):
            portfolio.execute_trade(symbol, 'SELL', final_prices[symbol])
        
        return {
            'portfolio_manager': portfolio,
            'metrics': portfolio.get_performance_metrics(),
            'final_value': portfolio.current_capital,
            'total_return': (portfolio.current_capital - initial_capital) / initial_capital
        }
    
    def _get_agent_signal(self, agent, symbol: str, date) -> Dict:
        """Get trading signal from agent (placeholder)"""
        # This would integrate with your actual RL agent
        # For now, return random signal
        action = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.1, 0.1, 0.8])
        strength = np.random.uniform(0.5, 1.0)
        
        return {
            'action': action,
            'strength': strength
        }
