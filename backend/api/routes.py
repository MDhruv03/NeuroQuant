from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import traceback, sqlite3, json

from backend.services.market_data import MarketDataProvider
from backend.services.strategy_manager import StrategyManager
from backend.api import analytics_routes
from database.database import get_db
from engine.backtester import Backtester
from engine.strategy import MovingAverageCrossStrategy, RSIStrategy, MomentumStrategy, BuyAndHoldStrategy

router = APIRouter(prefix="/api")
router.include_router(analytics_routes.router, prefix="")
data_provider, strategy_manager = MarketDataProvider(), StrategyManager()

class BacktestRequest(BaseModel):
    symbol: str; start_date: Optional[str] = None; end_date: Optional[str] = None
    strategy: str = "ma_cross"; strategy_params: Optional[Dict] = None

class BacktestResponse(BaseModel):
    symbol: str; test_period: str; agent_return: float; buy_hold_return: float
    outperformance: float; total_trades: int; final_value: float; sharpe_ratio: float
    sortino_ratio: float; max_drawdown: float; portfolio_dates: List[str]

class StrategyCreateRequest(BaseModel):
    name: str; type: str; parameters: Dict

class StrategyResponse(BaseModel):
    id: int; name: str; type: str; parameters: Dict

class BacktestRunResponse(BaseModel):
    id: int; timestamp: datetime; symbol: str; agent_id: Optional[int]; agent_name: Optional[str]
    test_period: str; agent_return: float; buy_hold_return: float; outperformance: float
    total_trades: int; final_value: float; trades: List[Dict]; portfolio_history: List[float]; portfolio_dates: List[str]

@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest, conn: sqlite3.Connection = Depends(get_db)):
    try:
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d") if request.start_date else datetime.now() - timedelta(days=365)
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d") if request.end_date else datetime.now()
        params, symbols = request.strategy_params or {}, [request.symbol]
        if request.strategy == "ma_cross": strategy = MovingAverageCrossStrategy(symbols, params.get("short_window", 20), params.get("long_window", 50))
        elif request.strategy == "rsi": strategy = RSIStrategy(symbols, params.get("period", 14), params.get("oversold", 30), params.get("overbought", 70))
        elif request.strategy == "momentum": strategy = MomentumStrategy(symbols, params.get("lookback", 20))
        elif request.strategy == "buy_hold": strategy = BuyAndHoldStrategy(symbols)
        else: raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.strategy}")
        backtester = Backtester(symbols, start_date, end_date, 100000.0, strategy)
        results = backtester.run()
        df = backtester.data_handler.data.get(request.symbol)
        buy_hold_return = float((float(df.iloc[-1]['Close']) / float(df.iloc[0]['Close']) - 1) * 100) if df is not None and len(df) > 0 else 0.0
        
        # Convert equity_curve and timestamps to lists for JSON serialization
        equity_curve = results['equity_curve'] if isinstance(results['equity_curve'], list) else list(results['equity_curve'])
        timestamps = results['timestamps'] if isinstance(results['timestamps'], list) else list(results['timestamps'])
        timestamps_iso = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in timestamps]
        
        cursor = conn.cursor()
        cursor.execute("""INSERT INTO backtest_runs (timestamp, symbol, agent_id, agent_name, test_period, agent_return, buy_hold_return, outperformance, total_trades, final_value, trades, portfolio_history, portfolio_dates) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (datetime.now().isoformat(), request.symbol, None, strategy.name, f"{start_date.date()} to {end_date.date()}", float(results['total_return_pct']), float(buy_hold_return), float(results['total_return_pct'] - buy_hold_return), int(results['fills_received']), float(results['final_equity']), json.dumps([]), json.dumps(equity_curve), json.dumps(timestamps_iso)))
        conn.commit()
        return BacktestResponse(symbol=request.symbol, test_period=f"{start_date.date()} to {end_date.date()}", agent_return=results['total_return_pct'], buy_hold_return=buy_hold_return, outperformance=results['total_return_pct'] - buy_hold_return, total_trades=results['fills_received'], final_value=results['final_equity'], sharpe_ratio=results['sharpe_ratio'], sortino_ratio=results['sortino_ratio'], max_drawdown=results['max_drawdown_pct'], value_at_risk_95=results.get('value_at_risk_95'), value_at_risk_99=results.get('value_at_risk_99'), conditional_var_95=results.get('conditional_var_95'), conditional_var_99=results.get('conditional_var_99'), trades=[], portfolio_history=equity_curve, portfolio_dates=timestamps_iso)
    except Exception as e:
        traceback.print_exc(); raise HTTPException(status_code=500, detail=str(e))

@router.get("/symbols")
async def get_popular_symbols():
    return {"yfinance_symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "CRM", "SPY", "QQQ", "DIA", "IWM"]}

@router.get("/strategies")
async def get_available_strategies():
    return {"strategies": [{"id": "ma_cross", "name": "Moving Average Crossover", "description": "Classic trend-following strategy", "parameters": {"short_window": {"type": "int", "default": 20, "min": 5, "max": 100}, "long_window": {"type": "int", "default": 50, "min": 20, "max": 200}}}, {"id": "rsi", "name": "RSI Mean Reversion", "description": "Mean reversion strategy", "parameters": {"period": {"type": "int", "default": 14, "min": 5, "max": 30}, "oversold": {"type": "int", "default": 30, "min": 10, "max": 40}, "overbought": {"type": "int", "default": 70, "min": 60, "max": 90}}}, {"id": "momentum", "name": "Momentum Strategy", "description": "Momentum-based strategy", "parameters": {"lookback": {"type": "int", "default": 20, "min": 10, "max": 60}}}, {"id": "buy_hold", "name": "Buy & Hold", "description": "Benchmark strategy", "parameters": {}}]}

@router.post("/agents", response_model=StrategyResponse)
async def create_strategy(request: StrategyCreateRequest, conn: sqlite3.Connection = Depends(get_db)):
    try:
        new_strategy = strategy_manager.create_strategy(conn, name=request.name, strategy_type=request.type, parameters=request.parameters)
        return StrategyResponse(**new_strategy)
    except ValueError as e: raise HTTPException(status_code=400, detail=str(e))
    except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail="Failed to create strategy.")

@router.get("/agents", response_model=List[StrategyResponse])
async def get_strategies(conn: sqlite3.Connection = Depends(get_db)):
    try:
        strategies = strategy_manager.get_strategies(conn)
        return [StrategyResponse(**strategy) for strategy in strategies]
    except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail="Failed to retrieve strategies.")

@router.get("/backtest_runs", response_model=List[BacktestRunResponse])
async def get_backtest_runs(conn: sqlite3.Connection = Depends(get_db)):
    try:
        cursor = conn.cursor(); cursor.execute("SELECT * FROM backtest_runs ORDER BY timestamp DESC"); runs_data = cursor.fetchall()
        runs = []
        for row in runs_data:
            run = dict(row); run['timestamp'] = datetime.fromisoformat(run['timestamp'])
            run['trades'] = json.loads(run['trades']) if run['trades'] else []
            run['portfolio_history'] = json.loads(run['portfolio_history']) if run['portfolio_history'] else []
            run['portfolio_dates'] = json.loads(run['portfolio_dates']) if run['portfolio_dates'] else []
            runs.append(run)
        return [BacktestRunResponse(**run) for run in runs]
    except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail="Failed to retrieve backtest runs.")

@router.get("/backtest_runs/{run_id}", response_model=BacktestRunResponse)
async def get_backtest_run_details(run_id: int, conn: sqlite3.Connection = Depends(get_db)):
    try:
        cursor = conn.cursor(); cursor.execute("SELECT * FROM backtest_runs WHERE id = ?", (run_id,)); run_data = cursor.fetchone()
        if not run_data: raise HTTPException(status_code=404, detail=f"Backtest run with ID {run_id} not found.")
        run = dict(run_data); run['timestamp'] = datetime.fromisoformat(run['timestamp'])
        run['trades'] = json.loads(run['trades']) if run['trades'] else []
        run['portfolio_history'] = json.loads(run['portfolio_history']) if run['portfolio_history'] else []
        run['portfolio_dates'] = json.loads(run['portfolio_dates']) if run['portfolio_dates'] else []
        return BacktestRunResponse(**run)
    except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail="Failed to retrieve backtest run details.")

@router.get("/compare_strategies")
async def compare_strategies(symbol: str = "AAPL", start_date: Optional[str] = None, end_date: Optional[str] = None, conn: sqlite3.Connection = Depends(get_db)):
    """Compare all available strategies on the same symbol and timeframe"""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime.now() - timedelta(days=365)
        end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
        
        strategies_to_compare = [
            ("ma_cross", MovingAverageCrossStrategy([symbol], 20, 50), {}),
            ("rsi", RSIStrategy([symbol], 14, 30, 70), {}),
            ("momentum", MomentumStrategy([symbol], 20), {}),
            ("buy_hold", BuyAndHoldStrategy([symbol]), {})
        ]
        
        results = []
        for strategy_id, strategy, params in strategies_to_compare:
            backtester = Backtester([symbol], start, end, 100000.0, strategy)
            result = backtester.run()
            results.append({
                "strategy": strategy_id,
                "name": strategy.name,
                "total_return": result['total_return_pct'],
                "sharpe_ratio": result['sharpe_ratio'],
                "sortino_ratio": result['sortino_ratio'],
                "max_drawdown": result['max_drawdown_pct'],
                "total_trades": result['fills_received'],
                "final_value": result['final_equity']
            })
        
        return {"symbol": symbol, "period": f"{start.date()} to {end.date()}", "results": results}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
