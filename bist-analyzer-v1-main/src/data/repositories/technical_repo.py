# src/data/repositories/technical_repo.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, date, timedelta
import logging
import aiohttp
import asyncio

from src.domain.interfaces.repository import ITechnicalRepository

logger = logging.getLogger(__name__)


class TechnicalIndicatorCalculator:
    """Teknik gösterge hesaplama motoru."""
    
    @staticmethod
    def calculate_rsi(close_prices: pd.Series, period: int = 14) -> float:
        """Relative Strength Index"""
        if len(close_prices) < period + 1:
            return 50.0
        delta = close_prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.tail(period).mean()
        avg_loss = loss.tail(period).mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 1)
    
    @staticmethod
    def calculate_macd(close_prices: pd.Series) -> Dict[str, float]:
        """MACD (12, 26, 9)"""
        if len(close_prices) < 35:
            return {"macd_line": 0, "signal_line": 0, "histogram": 0}
        ema_12 = close_prices.ewm(span=12, adjust=False).mean()
        ema_26 = close_prices.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            "macd_line": round(macd_line.iloc[-1], 4),
            "signal_line": round(signal_line.iloc[-1], 4),
            "histogram": round(histogram.iloc[-1], 4)
        }
    
    @staticmethod
    def calculate_bollinger_bands(close_prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """Bollinger Bantları (20,2)"""
        if len(close_prices) < period:
            return {"upper": 0, "middle": 0, "lower": 0, "position": 50}
        sma = close_prices.rolling(window=period).mean()
        std = close_prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        current = close_prices.iloc[-1]
        position = ((current - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) * 100) if upper.iloc[-1] != lower.iloc[-1] else 50
        return {
            "upper": round(upper.iloc[-1], 2),
            "middle": round(sma.iloc[-1], 2),
            "lower": round(lower.iloc[-1], 2),
            "position": round(position, 1)
        }
    
    @staticmethod
    def detect_trend(df: pd.DataFrame, short_ma: int = 20, long_ma: int = 50) -> str:
        """Trend yönü tespiti"""
        if len(df) < long_ma:
            return "neutral"
        sma_short = df["close"].rolling(window=short_ma).mean().iloc[-1]
        sma_long = df["close"].rolling(window=long_ma).mean().iloc[-1]
        current_price = df["close"].iloc[-1]
        if current_price > sma_short > sma_long:
            return "bullish"
        elif current_price < sma_short < sma_long:
            return "bearish"
        else:
            return "neutral"


class TechnicalRepository(ITechnicalRepository):
    """Teknik veri deposu - Yahoo Finance'ten OHLCV çeker, göstergeleri hesaplar."""
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._calculator = TechnicalIndicatorCalculator()
        self._ohlcv_cache: Dict[str, pd.DataFrame] = {}
        self._indicator_cache: Dict[str, Dict] = {}
    
    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"User-Agent": "BIST-AI-Analyzer/1.0"}
            )
        return self._session
    
    async def get_ohlcv(self, ticker: str, start_date: date = None, end_date: date = None, interval: str = "1d") -> pd.DataFrame:
        """OHLCV verisi"""
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=365)
        
        cache_key = f"{ticker}:{start_date}:{end_date}:{interval}"
        if cache_key in self._ohlcv_cache:
            return self._ohlcv_cache[cache_key]
        
        symbol = f"{ticker}.IS"
        period1 = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        period2 = int(datetime.combine(end_date, datetime.max.time()).timestamp())
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"period1": period1, "period2": period2, "interval": interval}
        
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return pd.DataFrame()
                data = await response.json()
                chart = data.get("chart", {}).get("result", [{}])[0]
                if not chart:
                    return pd.DataFrame()
                timestamps = chart.get("timestamp", [])
                quotes = chart.get("indicators", {}).get("quote", [{}])[0]
                if not timestamps:
                    return pd.DataFrame()
                df = pd.DataFrame({
                    "timestamp": pd.to_datetime(timestamps, unit="s"),
                    "open": quotes.get("open", []),
                    "high": quotes.get("high", []),
                    "low": quotes.get("low", []),
                    "close": quotes.get("close", []),
                    "volume": quotes.get("volume", [])
                })
                df = df.dropna(subset=["close"])
                df = df.set_index("timestamp")
                self._ohlcv_cache[cache_key] = df
                return df
        except Exception as e:
            logger.error(f"OHLCV hatası ({ticker}): {e}")
            return pd.DataFrame()
    
    async def get_indicators(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """Tüm teknik göstergeleri hesapla"""
        cache_key = f"{ticker}:indicators:{period}"
        if cache_key in self._indicator_cache:
            return self._indicator_cache[cache_key]
        
        end_date = date.today()
        if period == "1mo":
            start_date = end_date - timedelta(days=30)
        elif period == "3mo":
            start_date = end_date - timedelta(days=90)
        elif period == "6mo":
            start_date = end_date - timedelta(days=180)
        else:
            start_date = end_date - timedelta(days=365)
        
        df = await self.get_ohlcv(ticker, start_date, end_date)
        
        if df.empty:
            return {"error": "Veri bulunamadı", "rsi_14": 50, "trend": "neutral"}
        
        close = df["close"]
        current_price = close.iloc[-1]
        
        indicators = {
            "current_price": round(current_price, 2),
            "rsi_14": self._calculator.calculate_rsi(close),
            "macd": self._calculator.calculate_macd(close),
            "bollinger": self._calculator.calculate_bollinger_bands(close),
            "trend": self._calculator.detect_trend(df),
            "volume_avg_20d": round(df["volume"].tail(20).mean(), 0) if len(df) >= 20 else 0,
            "price_change_1d": round(((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100, 2) if len(close) >= 2 else 0,
        }
        
        self._indicator_cache[cache_key] = indicators
        return indicators
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()