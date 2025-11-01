"""
Main Backtesting Engine
Event-driven backtester with realistic market simulation
"""
from typing import List, Dict, Optional
from datetime import datetime
from queue import Queue
import pandas as pd
from collections import defaultdict

from engine.events import (
    Event, EventType, MarketEvent, SignalEvent, OrderEvent,
    FillEvent, OrderType, OrderSide, SignalType
)
from engine.data_handler import DataHandler
from engine.strategy import Strategy
from engine.execution import ExecutionEngine
from engine.portfolio import Portfolio, RiskManager
from engine.position_sizer import PositionSizer


class Backtester:
    """
    Main backtesting engine
    
    Components:
    - DataHandler: Market data
    - Strategy: Signal generation
    - Portfolio: Position tracking
    - ExecutionEngine: Order matching
    - RiskManager: Risk controls
    - PositionSizer: Position sizing
    """
    
    def __init__(self,
                 symbols: List[str],
                 start_date: datetime,
                 end_date: datetime,
                 initial_capital: float = 100000.0,
                 strategy: Optional[Strategy] = None):
        
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        
        # Initialize components
        self.data_handler = DataHandler(symbols, start_date, end_date)
        self.portfolio = Portfolio(initial_capital)
        self.execution_engine = ExecutionEngine()
        self.risk_manager = RiskManager()
        self.position_sizer = PositionSizer(method="percent_equity", percent=0.10)
        
        # Strategy
        self.strategy = strategy
        
        # Event queue
        self.events = Queue()
        
        # Tracking
        self.signals_generated = 0
        self.orders_placed = 0
        self.fills_received = 0
        self.bars_processed = 0
        
        # Results
        self.results: Dict = {}
    
    def _process_market_event(self, event: MarketEvent):
        """Process market data event"""
        # Generate signals from strategy
        if self.strategy:
            signals = self.strategy.calculate_signals(event, self.data_handler)
            for signal in signals:
                self.events.put(signal)
                self.signals_generated += 1
    
    def _process_signal_event(self, event: SignalEvent):
        """Convert signal to order"""
        # Check if circuit breaker triggered
        if self.risk_manager.circuit_breaker_triggered:
            print("⛔ Circuit breaker active - blocking new orders")
            return
        
        current_prices = self.data_handler.get_latest_prices()
        
        # Check risk limits
        if not self.risk_manager.check_all(self.portfolio, current_prices):
            print(f"⚠️  Risk check failed: {self.risk_manager.violations}")
            return
        
        # Determine order side and quantity
        symbol = event.symbol
        current_position = self.portfolio.positions.get(symbol)
        
        if event.signal_type == SignalType.LONG:
            # Close short if exists, then go long
            if current_position and current_position.quantity < 0:
                # Close short
                order = OrderEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side=OrderSide.BUY,
                    quantity=abs(current_position.quantity),
                    strategy_name=event.strategy_name
                )
                self.events.put(order)
                self.orders_placed += 1
            
            # Open long
            quantity = self.position_sizer.calculate_quantity(event, self.portfolio, current_prices)
            if quantity > 0:
                order = OrderEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    strategy_name=event.strategy_name
                )
                self.events.put(order)
                self.orders_placed += 1
        
        elif event.signal_type == SignalType.SHORT:
            # Close long if exists, then go short
            if current_position and current_position.quantity > 0:
                # Close long
                order = OrderEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL,
                    quantity=current_position.quantity,
                    strategy_name=event.strategy_name
                )
                self.events.put(order)
                self.orders_placed += 1
            
            # Open short
            quantity = self.position_sizer.calculate_quantity(event, self.portfolio, current_prices)
            if quantity > 0:
                order = OrderEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL,
                    quantity=quantity,
                    strategy_name=event.strategy_name
                )
                self.events.put(order)
                self.orders_placed += 1
        
        elif event.signal_type == SignalType.EXIT:
            # Close position
            if current_position:
                if current_position.quantity > 0:
                    side = OrderSide.SELL
                    quantity = current_position.quantity
                else:
                    side = OrderSide.BUY
                    quantity = abs(current_position.quantity)
                
                order = OrderEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side=side,
                    quantity=quantity,
                    strategy_name=event.strategy_name
                )
                self.events.put(order)
                self.orders_placed += 1
    
    def _process_order_event(self, event: OrderEvent):
        """Execute order"""
        # Get current market data
        bar = self.data_handler.get_latest_bar(event.symbol)
        if not bar:
            return
        
        # Create market event for execution
        market_event = MarketEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            bid_price=bar.close * 0.9995,  # Approximate
            ask_price=bar.close * 1.0005,
            bid_size=1000,
            ask_size=1000
        )
        
        # Execute through engine
        fill = self.execution_engine.execute_order(event, market_event)
        
        if fill:
            self.events.put(fill)
            self.fills_received += 1
    
    def _process_fill_event(self, event: FillEvent):
        """Update portfolio with fill"""
        current_prices = self.data_handler.get_latest_prices()
        self.portfolio.update_fill(event, current_prices)
        
        # Notify strategy
        if self.strategy:
            self.strategy.on_fill(event)
    
    def run(self):
        """
        Main backtest loop
        """
        print("=" * 60)
        print(f"🚀 Starting Backtest")
        print(f"Symbols: {self.symbols}")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Strategy: {self.strategy.name if self.strategy else 'None'}")
        print("=" * 60)
        
        # Main event loop
        while self.data_handler.continue_backtest() or not self.events.empty():
            # Get new market data
            if self.data_handler.continue_backtest():
                market_events = self.data_handler.update_bars()
                for event in market_events:
                    self.events.put(event)
                    self.bars_processed += 1
            
            # Process all events in queue
            while not self.events.empty():
                event = self.events.get()
                
                if event.event_type == EventType.MARKET:
                    self._process_market_event(event)
                
                elif event.event_type == EventType.SIGNAL:
                    self._process_signal_event(event)
                
                elif event.event_type == EventType.ORDER:
                    self._process_order_event(event)
                
                elif event.event_type == EventType.FILL:
                    self._process_fill_event(event)
        
        # Finalize
        self._calculate_results()
        self._print_results()
        
        return self.results
    
    def _calculate_results(self):
        """Calculate performance metrics"""
        final_prices = self.data_handler.get_latest_prices()
        final_equity = self.portfolio.get_equity(final_prices)
        
        # Basic metrics
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Advanced metrics
        sharpe = self.portfolio.calculate_sharpe_ratio()
        sortino = self.portfolio.calculate_sortino_ratio()
        max_dd = self.portfolio.max_drawdown
        
        # Calculate VaR and CVaR from equity curve
        import numpy as np
        if len(self.portfolio.equity_curve) > 1:
            returns = np.diff(self.portfolio.equity_curve) / self.portfolio.equity_curve[:-1]
            var_95 = float(np.percentile(returns, 5) * 100)  # 5th percentile (95% VaR)
            var_99 = float(np.percentile(returns, 1) * 100)  # 1st percentile (99% VaR)
            cvar_95 = float(returns[returns <= np.percentile(returns, 5)].mean() * 100) if len(returns[returns <= np.percentile(returns, 5)]) > 0 else 0.0
            cvar_99 = float(returns[returns <= np.percentile(returns, 1)].mean() * 100) if len(returns[returns <= np.percentile(returns, 1)]) > 0 else 0.0
        else:
            var_95 = var_99 = cvar_95 = cvar_99 = 0.0
        
        # Trade statistics
        num_trades = len(self.portfolio.fill_history)
        
        # Ensure equity_curve is a list and timestamps are datetime objects
        equity_curve = list(self.portfolio.equity_curve) if not isinstance(self.portfolio.equity_curve, list) else self.portfolio.equity_curve
        timestamps = list(self.portfolio.timestamps) if not isinstance(self.portfolio.timestamps, list) else self.portfolio.timestamps
        
        self.results = {
            'initial_capital': self.initial_capital,
            'final_equity': float(final_equity),
            'total_return': float(total_return),
            'total_return_pct': float(total_return * 100),
            'sharpe_ratio': float(sharpe) if sharpe is not None else 0.0,
            'sortino_ratio': float(sortino) if sortino is not None else 0.0,
            'max_drawdown': float(max_dd) if max_dd is not None else 0.0,
            'max_drawdown_pct': float(max_dd * 100) if max_dd is not None else 0.0,
            'value_at_risk_95': var_95,
            'value_at_risk_99': var_99,
            'conditional_var_95': cvar_95,
            'conditional_var_99': cvar_99,
            'num_trades': int(num_trades),
            'signals_generated': int(self.signals_generated),
            'orders_placed': int(self.orders_placed),
            'fills_received': int(self.fills_received),
            'bars_processed': int(self.bars_processed),
            'total_commission': float(self.portfolio.total_commission_paid),
            'total_slippage': float(self.portfolio.total_slippage_cost),
            'equity_curve': equity_curve,
            'timestamps': timestamps
        }
    
    def _print_results(self):
        """Print backtest results"""
        print("\n" + "=" * 60)
        print("📊 BACKTEST RESULTS")
        print("=" * 60)
        
        print(f"\n💰 Returns:")
        print(f"  Initial Capital:    ${self.results['initial_capital']:>12,.2f}")
        print(f"  Final Equity:       ${self.results['final_equity']:>12,.2f}")
        print(f"  Total Return:       {self.results['total_return_pct']:>12.2f}%")
        
        print(f"\n📈 Risk Metrics:")
        print(f"  Sharpe Ratio:       {self.results['sharpe_ratio']:>12.2f}")
        print(f"  Sortino Ratio:      {self.results['sortino_ratio']:>12.2f}")
        print(f"  Max Drawdown:       {self.results['max_drawdown_pct']:>12.2f}%")
        
        print(f"\n🔄 Trading Activity:")
        print(f"  Signals Generated:  {self.results['signals_generated']:>12,}")
        print(f"  Orders Placed:      {self.results['orders_placed']:>12,}")
        print(f"  Fills Received:     {self.results['fills_received']:>12,}")
        print(f"  Bars Processed:     {self.results['bars_processed']:>12,}")
        
        print(f"\n💸 Costs:")
        print(f"  Total Commission:   ${self.results['total_commission']:>12,.2f}")
        print(f"  Total Slippage:     ${self.results['total_slippage']:>12,.2f}")
        
        print("\n" + "=" * 60)
    
    def get_equity_curve_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        return pd.DataFrame({
            'timestamp': self.results['timestamps'],
            'equity': self.results['equity_curve']
        })
    
    def plot_equity_curve(self):
        """Plot equity curve (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            df = self.get_equity_curve_df()
            
            plt.figure(figsize=(12, 6))
            plt.plot(df['timestamp'], df['equity'], linewidth=2)
            plt.title(f'Equity Curve - {self.strategy.name if self.strategy else "Backtest"}')
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib not installed. Install with: pip install matplotlib")


class WalkForwardAnalysis:
    """
    Walk-forward testing framework
    
    Splits data into training/testing windows
    Optimizes on training, validates on testing
    """
    
    def __init__(self,
                 symbols: List[str],
                 start_date: datetime,
                 end_date: datetime,
                 train_period_days: int = 180,
                 test_period_days: int = 60):
        
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        
        self.results: List[Dict] = []
    
    def run(self, strategy_class, strategy_params: Dict):
        """
        Run walk-forward analysis
        
        Args:
            strategy_class: Strategy class to instantiate
            strategy_params: Parameters for strategy
        """
        from datetime import timedelta
        
        current_date = self.start_date
        window_num = 0
        
        print("=" * 60)
        print("🔄 Walk-Forward Analysis")
        print("=" * 60)
        
        while current_date < self.end_date:
            window_num += 1
            
            # Define periods
            train_start = current_date
            train_end = train_start + timedelta(days=self.train_period_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_period_days)
            
            if test_end > self.end_date:
                break
            
            print(f"\n📅 Window {window_num}")
            print(f"  Train: {train_start.date()} to {train_end.date()}")
            print(f"  Test:  {test_start.date()} to {test_end.date()}")
            
            # Create strategy
            strategy = strategy_class(self.symbols, **strategy_params)
            
            # Run backtest on test period (assumes strategy trained on train period)
            backtester = Backtester(
                symbols=self.symbols,
                start_date=test_start,
                end_date=test_end,
                strategy=strategy
            )
            
            results = backtester.run()
            results['window'] = window_num
            results['train_start'] = train_start
            results['train_end'] = train_end
            results['test_start'] = test_start
            results['test_end'] = test_end
            
            self.results.append(results)
            
            # Move to next window
            current_date = test_end
        
        self._print_summary()
        return self.results
    
    def _print_summary(self):
        """Print walk-forward summary"""
        print("\n" + "=" * 60)
        print("📊 Walk-Forward Summary")
        print("=" * 60)
        
        avg_return = sum(r['total_return_pct'] for r in self.results) / len(self.results)
        avg_sharpe = sum(r['sharpe_ratio'] for r in self.results) / len(self.results)
        avg_max_dd = sum(r['max_drawdown_pct'] for r in self.results) / len(self.results)
        
        print(f"\nWindows Tested: {len(self.results)}")
        print(f"Avg Return:     {avg_return:.2f}%")
        print(f"Avg Sharpe:     {avg_sharpe:.2f}")
        print(f"Avg Max DD:     {avg_max_dd:.2f}%")
        print("\n" + "=" * 60)
