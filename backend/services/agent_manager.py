import sqlite3
import json
from typing import List, Dict

from database.database import DATABASE_URL, get_db

class AgentManager:
    def __init__(self):
        pass

    def create_agent(self, conn: sqlite3.Connection, name: str, agent_type: str, parameters: Dict) -> Dict:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO agents (name, type, parameters) VALUES (?, ?, ?)",
                (name, agent_type, json.dumps(parameters))
            )
            conn.commit()
            return {"id": cursor.lastrowid, "name": name, "type": agent_type, "parameters": parameters}
        except sqlite3.IntegrityError:
            raise ValueError(f"Agent with name '{name}' already exists.")

    def get_agents(self, conn: sqlite3.Connection) -> List[Dict]:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, type, parameters FROM agents")
        agents_data = cursor.fetchall()
        
        agents = []
        for row in agents_data:
            agent = dict(row)
            agent['parameters'] = json.loads(agent['parameters'])
            agents.append(agent)
        return agents

    def get_agent_by_id(self, conn: sqlite3.Connection, agent_id: int) -> Dict:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, type, parameters FROM agents WHERE id = ?", (agent_id,))
        agent_data = cursor.fetchone()
        if agent_data:
            agent = dict(agent_data)
            agent['parameters'] = json.loads(agent['parameters'])
            return agent
        return None
