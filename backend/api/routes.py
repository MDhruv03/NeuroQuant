from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from datetime import datetime
import traceback

from ..services.market_data import MarketDataProvider
from ..services.trading_agent import BacktestEngine

router = APIRouter()

# Global instances
data_provider = MarketDataProvider()

class BacktestRequest(BaseModel):
    symbol: str
    train_split: float = 0.7

class BacktestResponse(BaseModel):
    symbol: str
    test_period: str
    agent_return: float
    buy_hold_return: float
    outperformance: float
    total_trades: int
    final_value: float
    trades: List[Dict]

@router.get("/")
async def root():
    return {"message": "NeuroQuant MVP is running!", "version": "1.0.0"}

@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """Run backtest for a given symbol"""
    try:
        backtest_engine = BacktestEngine(data_provider)
        
        results = backtest_engine.run_backtest(request.symbol, request.train_split)
        return BacktestResponse(**results)
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/symbols")
async def get_popular_symbols():
    """Get list of popular trading symbols"""
    return {
        "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "CRM"]
    }