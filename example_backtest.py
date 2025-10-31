"""
Example backtest script
Demonstrates the institutional-grade backtesting engine
"""
from datetime import datetime, timedelta
from engine.backtester import Backtester, WalkForwardAnalysis
from engine.strategy import MovingAverageCrossStrategy, RSIStrategy, MomentumStrategy

def run_simple_backtest():
    """Run a simple backtest with MA crossover strategy"""
    
    # Parameters
    symbols = ['AAPL', 'MSFT']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years
    initial_capital = 100000
    
    # Create strategy
    strategy = MovingAverageCrossStrategy(
        symbols=symbols,
        short_window=20,
        long_window=50
    )
    
    # Create backtester
    backtester = Backtester(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        strategy=strategy
    )
    
    # Run
    results = backtester.run()
    
    return results, backtester


def run_rsi_backtest():
    """Run RSI mean reversion strategy"""
    
    symbols = ['SPY']  # S&P 500 ETF
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)  # 3 years
    
    strategy = RSIStrategy(
        symbols=symbols,
        period=14,
        oversold=30,
        overbought=70
    )
    
    backtester = Backtester(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        strategy=strategy
    )
    
    results = backtester.run()
    return results, backtester


def run_walk_forward():
    """Run walk-forward analysis"""
    
    symbols = ['AAPL']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)  # 3 years
    
    wfa = WalkForwardAnalysis(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        train_period_days=180,  # 6 months training
        test_period_days=60      # 2 months testing
    )
    
    results = wfa.run(
        strategy_class=MovingAverageCrossStrategy,
        strategy_params={'short_window': 20, 'long_window': 50}
    )
    
    return results


if __name__ == '__main__':
    print("🎯 Institutional Backtesting Engine Demo\n")
    
    # Run simple backtest
    print("Running MA Crossover strategy...")
    results, backtester = run_simple_backtest()
    
    # Plot (if matplotlib available)
    try:
        backtester.plot_equity_curve()
    except:
        print("\nInstall matplotlib to see equity curve: pip install matplotlib")
    
    print("\n✅ Backtest complete!")
