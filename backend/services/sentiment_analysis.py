"""
Sentiment Analysis using FinBERT and news sources
"""
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from config import config
from utils.logging_config import setup_logger
from utils.exceptions import SentimentAnalysisError

logger = setup_logger(__name__)


class SentimentAnalyzer:
    """Sentiment analysis using FinBERT"""
    
    def __init__(self):
        self.enabled = config.sentiment.ENABLED
        self.model_name = config.sentiment.MODEL
        
        if self.enabled:
            try:
                logger.info(f"Loading sentiment model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Sentiment model loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load sentiment model: {e}")
                self.enabled = False
        
        self.news_api_key = config.sentiment.NEWS_API_KEY
        self.finnhub_api_key = config.sentiment.FINNHUB_API_KEY
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a text using FinBERT
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment scores {positive, negative, neutral}
        """
        if not self.enabled:
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34, "compound": 0.0}
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT labels: positive, negative, neutral
            scores = predictions[0].cpu().numpy()
            
            # Calculate compound score (positive - negative)
            compound = float(scores[0] - scores[1])
            
            return {
                "positive": float(scores[0]),
                "negative": float(scores[1]),
                "neutral": float(scores[2]),
                "compound": compound
            }
        
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            raise SentimentAnalysisError(f"Failed to analyze sentiment: {str(e)}")
    
    def fetch_news_newsapi(self, symbol: str, days_back: int = 7) -> List[Dict]:
        """
        Fetch news from NewsAPI
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
        
        Returns:
            List of news articles
        """
        if not self.news_api_key:
            logger.warning("NewsAPI key not configured")
            return []
        
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': symbol,
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en',
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data.get('articles', [])
        
        except Exception as e:
            logger.error(f"Failed to fetch news from NewsAPI: {e}")
            return []
    
    def fetch_news_finnhub(self, symbol: str, days_back: int = 7) -> List[Dict]:
        """
        Fetch news from Finnhub
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
        
        Returns:
            List of news articles
        """
        if not self.finnhub_api_key:
            logger.warning("Finnhub API key not configured")
            return []
        
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': symbol,
                'from': from_date,
                'to': to_date,
                'token': self.finnhub_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
        
        except Exception as e:
            logger.error(f"Failed to fetch news from Finnhub: {e}")
            return []
    
    def get_symbol_sentiment(self, symbol: str, days_back: int = 7) -> Dict[str, float]:
        """
        Get aggregated sentiment for a symbol based on recent news
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
        
        Returns:
            Aggregated sentiment scores
        """
        # Fetch news from available sources
        news_articles = []
        
        if self.finnhub_api_key:
            news_articles.extend(self.fetch_news_finnhub(symbol, days_back))
        
        if self.news_api_key:
            news_articles.extend(self.fetch_news_newsapi(symbol, days_back))
        
        if not news_articles:
            logger.warning(f"No news found for {symbol}, using neutral sentiment")
            return {
                "positive": 0.33,
                "negative": 0.33,
                "neutral": 0.34,
                "compound": 0.0,
                "article_count": 0
            }
        
        # Analyze sentiment for each article
        sentiments = []
        for article in news_articles[:20]:  # Limit to 20 most recent
            try:
                # Get headline and description
                text = article.get('headline') or article.get('title', '')
                description = article.get('summary') or article.get('description', '')
                full_text = f"{text} {description}"
                
                if full_text.strip():
                    sentiment = self.analyze_text(full_text)
                    sentiments.append(sentiment)
            
            except Exception as e:
                logger.debug(f"Failed to analyze article: {e}")
                continue
        
        if not sentiments:
            return {
                "positive": 0.33,
                "negative": 0.33,
                "neutral": 0.34,
                "compound": 0.0,
                "article_count": 0
            }
        
        # Aggregate sentiments
        avg_sentiment = {
            "positive": np.mean([s['positive'] for s in sentiments]),
            "negative": np.mean([s['negative'] for s in sentiments]),
            "neutral": np.mean([s['neutral'] for s in sentiments]),
            "compound": np.mean([s['compound'] for s in sentiments]),
            "article_count": len(sentiments)
        }
        
        logger.info(
            f"Sentiment for {symbol}: "
            f"Positive={avg_sentiment['positive']:.2f}, "
            f"Negative={avg_sentiment['negative']:.2f}, "
            f"Compound={avg_sentiment['compound']:.2f}, "
            f"Articles={avg_sentiment['article_count']}"
        )
        
        return avg_sentiment
    
    def get_mock_sentiment(self, symbol: str, date: str) -> float:
        """
        Generate mock sentiment for testing (when APIs not available)
        
        Args:
            symbol: Stock symbol
            date: Date string
        
        Returns:
            Mock sentiment score
        """
        np.random.seed(hash(f"{symbol}_{date}") % 2**32)
        base_sentiment = np.random.normal(0, 0.3)
        return float(np.clip(base_sentiment, -1, 1))
