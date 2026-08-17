# src/data/sources/realtime_price_feed.py
import asyncio
import aiohttp
import json
import logging
from typing import Dict, Optional, Callable, Set
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PriceTick:
    """Anlık fiyat verisi"""
    ticker: str
    price: float
    change_pct: float
    volume: float
    bid: float
    ask: float
    high: float
    low: float
    timestamp: datetime
    source: str


class RealTimePriceFeed:
    """
    GERÇEK ZAMANLI fiyat beslemesi.
    Yahoo Finance REST API ile 15 saniyede bir günceller.
    """
    
    def __init__(self):
        self._prices: Dict[str, PriceTick] = {}
        self._subscribers: Dict[str, Set[Callable]] = {}
        self._running = False
        self._rest_session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self):
        """Lazy REST session"""
        if self._rest_session is None or self._rest_session.closed:
            self._rest_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5),
                headers={"User-Agent": "BIST-AI-Analyzer/1.0"}
            )
        return self._rest_session
    
    async def start(self, tickers: list = None):
        """Fiyat beslemesini başlat"""
        self._running = True
        
        if tickers is None:
            tickers = ["THYAO", "GARAN", "AKBNK", "BIMAS", "ASELS", "TUPRS"]
        
        asyncio.create_task(self._poll_rest_prices(tickers))
        logger.info(f"Gerçek zamanlı fiyat beslemesi başladı: {len(tickers)} hisse")
    
    async def stop(self):
        """Fiyat beslemesini durdur"""
        self._running = False
        if self._rest_session and not self._rest_session.closed:
            await self._rest_session.close()
    
    async def _poll_rest_prices(self, tickers: list):
        """REST API ile fiyat polling (her 15 saniye)"""
        while self._running:
            try:
                symbols = ",".join(f"{t}.IS" for t in tickers)
                url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbols}"
                
                session = await self._get_session()
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        quotes = data.get("quoteResponse", {}).get("result", [])
                        
                        for quote in quotes:
                            ticker = quote.get("symbol", "").replace(".IS", "")
                            if ticker:
                                tick = PriceTick(
                                    ticker=ticker,
                                    price=quote.get("regularMarketPrice", 0),
                                    change_pct=quote.get("regularMarketChangePercent", 0),
                                    volume=quote.get("regularMarketVolume", 0),
                                    bid=quote.get("bid", 0),
                                    ask=quote.get("ask", 0),
                                    high=quote.get("regularMarketDayHigh", 0),
                                    low=quote.get("regularMarketDayLow", 0),
                                    timestamp=datetime.now(),
                                    source="rest"
                                )
                                self._prices[ticker] = tick
                                await self._notify_subscribers(ticker, tick)
            
            except Exception as e:
                logger.warning(f"REST polling hatası: {e}")
            
            await asyncio.sleep(15)
    
    async def get_price(self, ticker: str) -> Optional[PriceTick]:
        """Güncel fiyatı getir"""
        return self._prices.get(ticker.upper())
    
    async def get_all_prices(self) -> Dict[str, PriceTick]:
        """Tüm güncel fiyatları getir"""
        return self._prices.copy()
    
    def subscribe(self, ticker: str, callback: Callable):
        """Fiyat güncellemelerine abone ol"""
        ticker = ticker.upper()
        if ticker not in self._subscribers:
            self._subscribers[ticker] = set()
        self._subscribers[ticker].add(callback)
    
    async def _notify_subscribers(self, ticker: str, tick: PriceTick):
        """Abonelere bildirim gönder"""
        if ticker in self._subscribers:
            for callback in self._subscribers[ticker]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(tick)
                    else:
                        callback(tick)
                except Exception as e:
                    logger.error(f"Callback hatası ({ticker}): {e}")


# SINGLETON
price_feed = RealTimePriceFeed()