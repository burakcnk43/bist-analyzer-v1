import logging
import pandas as pd
import yfinance as yf
from typing import List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Fast, rule-based sentiment analyzer for financial news.
    Optimized for on-device performance.
    """
    def __init__(self):
        # Professional financial lexicon (Turkish & English)
        self.positive_words = {
            'yükseliş', 'artış', 'kâr', 'pozitif', 'büyüme', 'rekor', 'anlaşma', 'ihale',
            'sipariş', 'yatırım', 'temettü', 'güçlü', 'beklenti üstü', 'buy', 'upgrade',
            'bullish', 'growth', 'profit', 'dividend', 'contract', 'win', 'record'
        }
        self.negative_words = {
            'düşüş', 'zarar', 'negatif', 'azalış', 'iptal', 'risk', 'kayıp', 'zayıf',
            'beklenti altı', 'sell', 'downgrade', 'bearish', 'loss', 'cancel', 'lawsuit',
            'penalty', 'bankruptcy', 'default'
        }

    def analyze(self, text: str) -> float:
        if not text: return 0.0
        text = text.lower()
        pos_count = sum(1 for word in self.positive_words if word in text)
        neg_count = sum(1 for word in self.negative_words if word in text)

        total = pos_count + neg_count
        if total == 0: return 0.0
        return (pos_count - neg_count) / total

class NewsDataProvider:
    def __init__(self):
        self.analyzer = SentimentAnalyzer()

    def fetch_news(self, symbol: str) -> List[Dict]:
        """
        Fetches latest news from yfinance and calculates sentiment.
        """
        logger.info(f"Fetching active news for {symbol}")
        try:
            ticker = yf.Ticker(symbol)
            raw_news = ticker.news
            processed = []

            for item in raw_news:
                content = item.get('content', {})
                title = content.get('title', '')
                summary = content.get('summary', '')
                full_text = f"{title} {summary}"

                sentiment = self.analyzer.analyze(full_text)

                processed.append({
                    "symbol": symbol,
                    "timestamp": content.get('pubDate', datetime.now().isoformat()),
                    "title": title,
                    "sentiment": sentiment,
                    "source": item.get('provider', {}).get('displayName', 'Yahoo'),
                    "type": "news"
                })
            return processed
        except Exception as e:
            logger.error(f"News fetch error for {symbol}: {e}")
            return []

def get_news_sentiment_features(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Returns a daily sentiment score dataframe.
    """
    provider = NewsDataProvider()
    news = provider.fetch_news(symbol)
    if not news:
        return pd.DataFrame()

    df = pd.DataFrame(news)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily = df.groupby('date')['sentiment'].mean().to_frame('news_sentiment')
    return daily
