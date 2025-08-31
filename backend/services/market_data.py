import numpy as np
import pandas as pd
import yfinance as yf
import talib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class MarketDataProvider:
    """Handles market data fetching and preprocessing"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def fetch_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using TA-Lib"""
        data = df.copy()
        
        # Price-based indicators
        data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
        data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
        data['EMA_12'] = talib.EMA(data['Close'], timeperiod=12)
        data['EMA_26'] = talib.EMA(data['Close'], timeperiod=26)
        
        # Momentum indicators
        data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
        data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'])
        data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
        data['STOCH_k'], data['STOCH_d'] = talib.STOCH(data['High'], data['Low'], data['Close'])
        
        # Volatility indicators
        data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['Close'])
        data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'])
        
        # Volume indicators
        data['Volume_SMA'] = talib.SMA(data['Volume'], timeperiod=20)
        data['OBV'] = talib.OBV(data['Close'], data['Volume'])
        
        # Calculate derived features
        data['Price_to_SMA20'] = data['Close'] / data['SMA_20']
        data['Price_to_SMA50'] = data['Close'] / data['SMA_50']
        data['BB_position'] = (data['Close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'])
        data['Volume_ratio'] = data['Volume'] / data['Volume_SMA']
        
        # Forward fill NaN values
        data = data.fillna(method='ffill').dropna()
        
        return data
    
    def get_mock_sentiment(self, symbol: str, date: str) -> float:
        """Mock sentiment analysis - replace with real news scraping"""
        # For MVP, we'll use random sentiment with some trend correlation
        np.random.seed(hash(f"{symbol}_{date}") % 2**32)
        base_sentiment = np.random.normal(0, 0.3)
        return np.clip(base_sentiment, -1, 1)