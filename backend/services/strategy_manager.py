import sqlite3
import json
from typing import List, Dict

from database.database import DATABASE_URL, get_db

class StrategyManager:
    """Strategy management for trading strategies"""
    
    def __init__(self):
        """Initialize the strategy manager"""
        pass

    def create_strategy(self, conn: sqlite3.Connection, name: str, strategy_type: str, parameters: Dict) -> Dict:
        """Create a new trading strategy"""
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO agents (name, type, parameters) VALUES (?, ?, ?)",
                (name, strategy_type, json.dumps(parameters))
            )
            conn.commit()
            return {"id": cursor.lastrowid, "name": name, "type": strategy_type, "parameters": parameters}
        except sqlite3.IntegrityError:
            raise ValueError(f"Strategy with name '{name}' already exists.")

    def get_strategies(self, conn: sqlite3.Connection) -> List[Dict]:
        """Get all trading strategies"""
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, type, parameters FROM agents")
        strategies_data = cursor.fetchall()
        
        strategies = []
        for row in strategies_data:
            strategy = dict(row)
            strategy['parameters'] = json.loads(strategy['parameters'])
            strategies.append(strategy)
        return strategies

    def get_strategy_by_id(self, conn: sqlite3.Connection, strategy_id: int) -> Dict:
        """Get a specific strategy by ID"""
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, type, parameters FROM agents WHERE id = ?", (strategy_id,))
        strategy_data = cursor.fetchone()
        if strategy_data:
            strategy = dict(strategy_data)
            strategy['parameters'] = json.loads(strategy['parameters'])
            return strategy
        return None
