import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [symbols, setSymbols] = useState([]);
  const [selectedSymbol, setSelectedSymbol] = useState('');
  const [trainSplit, setTrainSplit] = useState(0.7);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Fetch symbols on component mount
    fetch('http://localhost:8000/symbols')
      .then(response => response.json())
      .then(data => {
        setSymbols(data.symbols);
        setSelectedSymbol(data.symbols[0]);
      })
      .catch(error => console.error('Error fetching symbols:', error));
  }, []);

  const runBacktest = () => {
    setLoading(true);
    setResults(null);
    setError(null);

    fetch('http://localhost:8000/backtest', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        symbol: selectedSymbol,
        train_split: trainSplit,
      }),
    })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        setResults(data);
        setLoading(false);
      })
      .catch(error => {
        setError(error.message);
        setLoading(false);
      });
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>NeuroQuant</h1>
        <p>AI-Powered Trading Agent Dashboard</p>
      </header>
      <div className="container">
        <div className="control-panel">
          <h2>Run Backtest</h2>
          <div className="form-group">
            <label htmlFor="symbol">Stock Symbol</label>
            <select
              id="symbol"
              value={selectedSymbol}
              onChange={e => setSelectedSymbol(e.target.value)}
            >
              {symbols.map(symbol => (
                <option key={symbol} value={symbol}>
                  {symbol}
                </option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label htmlFor="trainSplit">Training Split</label>
            <input
              type="number"
              id="trainSplit"
              min="0.1"
              max="0.9"
              step="0.1"
              value={trainSplit}
              onChange={e => setTrainSplit(parseFloat(e.target.value))}
            />
          </div>
          <button className="btn" onClick={runBacktest} disabled={loading}>
            {loading ? 'Running...' : 'Start Backtest'}
          </button>
        </div>
        {loading && <div className="loading">Running backtest...</div>}
        {error && <div className="error">Error: {error}</div>}
        {results && (
          <div className="results">
            <h2>Backtest Results</h2>
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-value">{results.agent_return.toFixed(2)}%</div>
                <div className="metric-label">Agent Return</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{results.buy_hold_return.toFixed(2)}%</div>
                <div className="metric-label">Buy & Hold Return</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{results.outperformance.toFixed(2)}%</div>
                <div className="metric-label">Outperformance</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{results.total_trades}</div>
                <div className="metric-label">Total Trades</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;