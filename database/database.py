import sqlite3
import json
from config import config

DATABASE_URL = config.database.URL if hasattr(config.database, 'URL') else "./database/neuroquant.db"

def create_db_and_tables():
    with sqlite3.connect(DATABASE_URL) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL,
                parameters TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                agent_id INTEGER,
                agent_name TEXT,
                test_period TEXT NOT NULL,
                agent_return REAL NOT NULL,
                buy_hold_return REAL NOT NULL,
                outperformance REAL NOT NULL,
                total_trades INTEGER NOT NULL,
                final_value REAL NOT NULL,
                trades TEXT, -- Stored as JSON string
                portfolio_history TEXT, -- Stored as JSON string
                portfolio_dates TEXT, -- Stored as JSON string
                FOREIGN KEY (agent_id) REFERENCES agents(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS custom_datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                data TEXT NOT NULL -- Store CSV content as text
            )
        """)
        conn.commit()

        # Add default strategies if they don't exist
        default_strategies = [
            {"name": "MA Crossover (20/50)", "type": "ma_cross", "parameters": {"short_window": 20, "long_window": 50}},
            {"name": "RSI Mean Reversion", "type": "rsi", "parameters": {"period": 14, "oversold": 30, "overbought": 70}},
            {"name": "Momentum (20 days)", "type": "momentum", "parameters": {"lookback": 20}},
            {"name": "Buy & Hold Benchmark", "type": "buy_hold", "parameters": {}}
        ]

        for strategy_data in default_strategies:
            try:
                cursor.execute(
                    "INSERT INTO agents (name, type, parameters) VALUES (?, ?, ?)",
                    (strategy_data["name"], strategy_data["type"], json.dumps(strategy_data["parameters"]))
                )
                conn.commit()
                print(f"Default strategy '{strategy_data['name']}' added.")
            except sqlite3.IntegrityError:
                print(f"Default strategy '{strategy_data['name']}' already exists, skipping.")
            except Exception as e:
                print(f"Error adding default strategy '{strategy_data['name']}': {e}")

def get_db():
    conn = sqlite3.connect(DATABASE_URL, check_same_thread=False)
    conn.row_factory = sqlite3.Row # This allows accessing columns by name
    try:
        yield conn
    finally:
        conn.close()

# Call this function to initialize the database when the application starts
create_db_and_tables()
