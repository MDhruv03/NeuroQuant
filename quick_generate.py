"""
Quick data generation - uses only fast agents
"""
import requests
import time
from datetime import datetime, timedelta

API_BASE = "http://localhost:8000/api"

# Symbols to test
symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "JPM", "BAC", "DIS"]

# Use only fast agents (Indicator-Based and Random)
fast_agents = ["Indicator-Based (SMA Cross)", "Random Agent"]

# Shorter date ranges for speed
end_date = datetime.now()
date_ranges = [
    (end_date - timedelta(days=90), end_date),   # 3 months
    (end_date - timedelta(days=180), end_date),  # 6 months
]

print("🚀 Quick backtest generation (fast agents only)...\n")

successful = 0
failed = 0
total = len(symbols) * len(fast_agents) * len(date_ranges)
current = 0

for symbol in symbols:
    for agent in fast_agents:
        for start, end in date_ranges:
            current += 1
            try:
                payload = {
                    "symbol": symbol,
                    "agent_name": agent,
                    "start_date": start.strftime("%Y-%m-%d"),
                    "end_date": end.strftime("%Y-%m-%d")
                }
                
                print(f"[{current}/{total}] {symbol} + {agent.split('(')[0].strip()}... ", end='', flush=True)
                
                response = requests.post(f"{API_BASE}/backtest", json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
                        ret = data['agent_return']*100
                        color = "✓" if ret > 0 else "✗"
                        print(f"{color} {ret:+.1f}% ({data['total_trades']} trades)")
                        successful += 1
                    else:
                        print(f"✗ {data.get('message', 'Failed')}")
                        failed += 1
                else:
                    print(f"✗ HTTP {response.status_code}")
                    failed += 1
                
                time.sleep(0.3)
                
            except Exception as e:
                print(f"✗ {str(e)[:50]}")
                failed += 1

print(f"\n{'='*60}")
print(f"📊 {successful} successful, {failed} failed out of {total}")
print(f"{'='*60}")
