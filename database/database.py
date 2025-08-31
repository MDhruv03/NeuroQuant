import sqlite3
import json # Import json for parameters

DATABASE_URL = "./database/neuroquant.db"

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

        # Add default agents if they don't exist
        default_agents = [
            {"name": "DQN Agent (Default)", "type": "DQN", "parameters": {"learning_rate": 0.001, "gamma": 0.99}},
            {"name": "Indicator-Based (SMA Cross)", "type": "IndicatorBased", "parameters": {"short_sma": 20, "long_sma": 50}},
            {"name": "Random Agent", "type": "Random", "parameters": {}}
        ]

        for agent_data in default_agents:
            try:
                cursor.execute(
                    "INSERT INTO agents (name, type, parameters) VALUES (?, ?, ?)",
                    (agent_data["name"], agent_data["type"], json.dumps(agent_data["parameters"]))
                )
                conn.commit()
                print(f"Default agent '{agent_data['name']}' added.")
            except sqlite3.IntegrityError:
                print(f"Default agent '{agent_data['name']}' already exists, skipping.")
            except Exception as e:
                print(f"Error adding default agent '{agent_data['name']}': {e}")

def get_db():
    conn = sqlite3.connect(DATABASE_URL, check_same_thread=False)
    conn.row_factory = sqlite3.Row # This allows accessing columns by name
    try:
        yield conn
    finally:
        conn.close()

# Call this function to initialize the database when the application starts
create_db_and_tables()
