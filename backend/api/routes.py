from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
import traceback
import sqlite3
import json
import pandas as pd
import io
import asyncio

from backend.services.market_data import MarketDataProvider
from backend.services.trading_agent import BacktestEngine
from backend.services.agent_manager import AgentManager
from backend.services.portfolio_manager import PortfolioManager, MultiSymbolBacktester
from backend.services.strategy_optimizer import StrategyOptimizer, GeneticOptimizer
from backend.services.live_trading import LiveTradingSimulator, simple_momentum_agent, AlertRule
from backend.api import analytics_routes
from database.database import get_db

# Create router with /api prefix
router = APIRouter(prefix="/api")

# Include analytics routes (already has /api/analytics prefix)
router.include_router(analytics_routes.router, prefix="")

# Global instances
data_provider = MarketDataProvider()
agent_manager = AgentManager() # Initialize AgentManager

# Live trading simulators
live_simulators: Dict[str, LiveTradingSimulator] = {}
simulator_tasks: Dict[str, asyncio.Task] = {}

class BacktestRequest(BaseModel):
    symbol: str
    train_split: float = 0.7
    agent_id: Optional[int] = None # Optional agent ID
    data_source: str = "yfinance" # New: "yfinance" or "custom"
    custom_dataset_id: Optional[int] = None # New: ID of custom dataset if data_source is "custom"

class BacktestResponse(BaseModel):
    symbol: str
    test_period: str
    agent_return: float
    buy_hold_return: float
    outperformance: float
    total_trades: int
    final_value: float
    trades: List[Dict]
    portfolio_history: List[float]
    portfolio_dates: List[str]

class AgentCreateRequest(BaseModel):
    name: str
    type: str # e.g., "DQN", "PPO", "IndicatorBased"
    parameters: Dict # JSON string of parameters

class AgentResponse(BaseModel):
    id: int
    name: str
    type: str
    parameters: Dict

class BacktestRunResponse(BaseModel):
    id: int
    timestamp: datetime
    symbol: str
    agent_id: Optional[int]
    agent_name: Optional[str]
    test_period: str
    agent_return: float
    buy_hold_return: float
    outperformance: float
    total_trades: int
    final_value: float
    trades: List[Dict]
    portfolio_history: List[float]
    portfolio_dates: List[str]

class CustomDatasetResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None

@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest, conn: sqlite3.Connection = Depends(get_db)):
    """Run backtest for a given symbol and save results"""
    try:
        agent_config = None
        agent_name = "Default Agent" # Default name if no agent_id is provided
        if request.agent_id:
            agent_config = agent_manager.get_agent_by_id(conn, request.agent_id)
            if not agent_config:
                raise HTTPException(status_code=404, detail=f"Agent with ID {request.agent_id} not found.")
            agent_name = agent_config['name']
            print(f"Using agent: {agent_name} (Type: {agent_config['type']})")

        # Handle data source
        if request.data_source == "yfinance":
            data = data_provider.fetch_stock_data(request.symbol)
        elif request.data_source == "custom":
            if not request.custom_dataset_id:
                raise HTTPException(status_code=400, detail="custom_dataset_id is required for custom data source.")
            
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM custom_datasets WHERE id = ?", (request.custom_dataset_id,))
            dataset_row = cursor.fetchone()
            if not dataset_row:
                raise HTTPException(status_code=404, detail=f"Custom dataset with ID {request.custom_dataset_id} not found.")
            
            csv_data = dataset_row['data']
            # Assuming CSV has 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'
            data = pd.read_csv(io.StringIO(csv_data), index_col='Date', parse_dates=True)
            # Ensure column names are consistent with yfinance output
            data.columns = [col.capitalize() for col in data.columns]
            # For custom data, symbol will be the dataset name
            request.symbol = f"Custom: {dataset_row['name']}" 
        else:
            raise HTTPException(status_code=400, detail="Invalid data_source. Must be 'yfinance' or 'custom'.")

        # Continue with backtest logic
        data['Return'] = data['Close'].pct_change()
        data.dropna(inplace=True)
        data = data_provider.calculate_technical_indicators(data) # Use data_provider's method

        backtest_engine = BacktestEngine(data_provider) # Pass data_provider
        results = backtest_engine.run_backtest(request.symbol, request.train_split, agent_config=agent_config)

        # Save backtest results to database
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO backtest_runs (
                timestamp, symbol, agent_id, agent_name, test_period,
                agent_return, buy_hold_return, outperformance, total_trades,
                final_value, trades, portfolio_history, portfolio_dates
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                results['symbol'],
                request.agent_id,
                agent_name,
                results['test_period'],
                results['agent_return'],
                results['buy_hold_return'],
                results['outperformance'],
                results['total_trades'],
                results['final_value'],
                json.dumps(results['trades']),
                json.dumps(results['portfolio_history']),
                json.dumps(results['portfolio_dates'])
            )
        )
        conn.commit()

        return BacktestResponse(**results)
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/symbols")
async def get_popular_symbols(conn: sqlite3.Connection = Depends(get_db)): # Inject conn
    """Get list of popular trading symbols and custom datasets"""
    yfinance_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "CRM"]
    
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM custom_datasets")
    custom_datasets = cursor.fetchall()
    
    custom_symbol_options = [{"id": row['id'], "name": f"Custom: {row['name']}"} for row in custom_datasets]
    
    return {
        "yfinance_symbols": yfinance_symbols,
        "custom_datasets": custom_symbol_options
    }

@router.post("/agents", response_model=AgentResponse)
async def create_agent(agent_request: AgentCreateRequest, conn: sqlite3.Connection = Depends(get_db)):
    """Create a new trading agent"""
    try:
        new_agent = agent_manager.create_agent(
            conn, # Pass the connection
            name=agent_request.name,
            agent_type=agent_request.type,
            parameters=agent_request.parameters
        )
        return AgentResponse(**new_agent)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to create agent.")

@router.get("/agents", response_model=List[AgentResponse])
async def get_agents(conn: sqlite3.Connection = Depends(get_db)):
    """Get a list of all registered trading agents"""
    try:
        agents = agent_manager.get_agents(conn) # Pass the connection
        return [AgentResponse(**agent) for agent in agents]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to retrieve agents.")

@router.get("/backtest_runs", response_model=List[BacktestRunResponse])
async def get_backtest_runs(conn: sqlite3.Connection = Depends(get_db)):
    """Get a list of all past backtest runs"""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM backtest_runs ORDER BY timestamp DESC")
        runs_data = cursor.fetchall()
        
        runs = []
        for row in runs_data:
            run = dict(row)
            run['timestamp'] = datetime.fromisoformat(run['timestamp'])
            run['trades'] = json.loads(run['trades']) if run['trades'] else []
            run['portfolio_history'] = json.loads(run['portfolio_history']) if run['portfolio_history'] else []
            run['portfolio_dates'] = json.loads(run['portfolio_dates']) if run['portfolio_dates'] else []
            runs.append(run)
        return [BacktestRunResponse(**run) for run in runs]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to retrieve backtest runs.")

@router.get("/backtest_runs/{run_id}", response_model=BacktestRunResponse)
async def get_backtest_run_details(run_id: int, conn: sqlite3.Connection = Depends(get_db)):
    """Get details of a specific backtest run"""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM backtest_runs WHERE id = ?", (run_id,))
        run_data = cursor.fetchone()
        
        if not run_data:
            raise HTTPException(status_code=404, detail=f"Backtest run with ID {run_id} not found.")
        
        run = dict(run_data)
        run['timestamp'] = datetime.fromisoformat(run['timestamp'])
        run['trades'] = json.loads(run['trades']) if run['trades'] else []
        run['portfolio_history'] = json.loads(run['portfolio_history']) if run['portfolio_history'] else []
        run['portfolio_dates'] = json.loads(run['portfolio_dates']) if run['portfolio_dates'] else []
        
        return BacktestRunResponse(**run)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to retrieve backtest run details.")

@router.post("/upload_dataset")
async def upload_dataset(
    name: str,
    file: UploadFile = File(...),
    description: Optional[str] = None,
    conn: sqlite3.Connection = Depends(get_db)
):
    """Upload a custom CSV dataset for backtesting."""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
        
        contents = await file.read()
        csv_data = contents.decode('utf-8')

        # Basic validation: check if it's a valid CSV for stock data
        # (e.g., has 'Date', 'Open', 'High', 'Low', 'Close', 'Volume' columns)
        try:
            df = pd.read_csv(io.StringIO(csv_data))
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                raise HTTPException(status_code=400, detail=f"CSV must contain columns: {', '.join(required_cols)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format or missing required columns: {e}")

        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO custom_datasets (name, description, data) VALUES (?, ?, ?)",
                (name, description, csv_data)
            )
            conn.commit()
            return {"message": f"Dataset '{name}' uploaded successfully!", "id": cursor.lastrowid}
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=400, detail=f"Dataset with name '{name}' already exists.")
    except HTTPException as e:
        raise e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {e}")

@router.get("/custom_datasets", response_model=List[CustomDatasetResponse])
async def get_custom_datasets(conn: sqlite3.Connection = Depends(get_db)):
    """Get a list of all uploaded custom datasets."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, description FROM custom_datasets")
        datasets = cursor.fetchall()
        return [CustomDatasetResponse(**dict(row)) for row in datasets]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to retrieve custom datasets: {e}")
