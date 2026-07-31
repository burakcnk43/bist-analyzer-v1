# src/data/pipeline/fast_data_pipeline.py
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import aiohttp

logger = logging.getLogger(__name__)


class FastDataPipeline:
    """
    YÜKSEK PERFORMANSLI veri işleme boru hattı.
    Paralel veri çekme, akıllı önbellekleme, öncelikli güncelleme.
    """
    
    def __init__(self, kap_client, redis_manager, price_feed, technical_repo):
        self.kap = kap_client
        self.redis = redis_manager
        self.prices = price_feed
        self.technical = technical_repo
        self._hot_stocks: List[str] = []
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"User-Agent": "BIST-AI-Analyzer/1.0"}
            )
        return self._session
    
    async def start(self):
        """Veri boru hattını başlat"""
        asyncio.create_task(self._periodic_full_update())
        logger.info("Hızlı veri boru hattı başlatıldı")
    
    async def get_stock_data(self, ticker: str, force_refresh: bool = False) -> Dict:
        """TEK HİSSE için tüm verileri hızlıca getir."""
        ticker = ticker.upper()
        
        # Memory cache kontrolü
        if not force_refresh:
            cached = await self.redis.get(f"stock_data:{ticker}")
            if cached:
                return cached
        
        # Paralel veri çekme
        tasks = [
            self._get_price_data(ticker),
            self._get_technical_data(ticker)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        price_info = results[0] if not isinstance(results[0], Exception) else {}
        technicals = results[1] if not isinstance(results[1], Exception) else {}
        
        stock_data = {
            "ticker": ticker,
            **price_info,
            "technicals": technicals,
            "last_updated": datetime.now().isoformat()
        }
        
        # Cache'e yaz (5 dakika)
        await self.redis.set(f"stock_data:{ticker}", stock_data, ttl=300)
        
        # Hot stock takibi
        self._track_hot_stock(ticker)
        
        return stock_data
    
    async def get_bulk_data(self, tickers: List[str]) -> Dict[str, Dict]:
        """TOPLU veri çekme - paralel isteklerle maksimum hız."""
        results = {}
        missing = []
        
        # Önce cache kontrolü
        for ticker in tickers:
            cached = await self.redis.get(f"stock_data:{ticker}")
            if cached:
                results[ticker] = cached
            else:
                missing.append(ticker)
        
        # Toplu çek (5'li batch'ler)
        if missing:
            batch_size = 5
            for i in range(0, len(missing), batch_size):
                batch = missing[i:i+batch_size]
                tasks = [self.get_stock_data(t) for t in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for ticker, result in zip(batch, batch_results):
                    if not isinstance(result, Exception):
                        results[ticker] = result
                
                await asyncio.sleep(0.3)  # Rate limit
        
        return results
    
    async def get_all_bist100_data(self) -> Dict[str, Dict]:
        """BIST 100'ün tamamı için veri çek"""
        bist100 = await self.kap.get_bist100_list()
        tickers = [s["ticker"] for s in bist100]
        return await self.get_bulk_data(tickers)
    
    async def _get_price_data(self, ticker: str) -> Dict:
        """Güncel fiyat verisi (çoklu kaynak)"""
        # 1. Canlı fiyat feed'i
        tick = await self.prices.get_price(ticker)
        if tick and tick.price > 0:
            return {
                "current_price": tick.price,
                "change_pct": tick.change_pct,
                "volume": tick.volume,
                "day_high": tick.high,
                "day_low": tick.low,
                "price_source": tick.source
            }
        
        # 2. Yahoo REST API
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.IS?interval=1d&range=5d"
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    chart = data.get("chart", {}).get("result", [{}])[0]
                    meta = chart.get("meta", {})
                    return {
                        "current_price": meta.get("regularMarketPrice", 0),
                        "change_pct": meta.get("regularMarketChangePercent", 0),
                        "volume": meta.get("regularMarketVolume", 0),
                        "day_high": meta.get("regularMarketDayHigh", 0),
                        "day_low": meta.get("regularMarketDayLow", 0),
                        "price_source": "yahoo_rest"
                    }
        except Exception:
            pass
        
        return {"current_price": 0, "change_pct": 0, "price_source": "offline"}
    
    async def _get_technical_data(self, ticker: str) -> Dict:
        """Hızlı teknik gösterge (son 100 mum)"""
        try:
            indicators = await self.technical.get_indicators(ticker, "6mo")
            if "error" not in indicators:
                return indicators
        except Exception:
            pass
        
        return {"rsi_14": 50, "trend": "neutral", "error": "Teknik veri alınamadı"}
    
    def _track_hot_stock(self, ticker: str):
        """En çok bakılan hisseleri takip et"""
        if ticker not in self._hot_stocks:
            self._hot_stocks.append(ticker)
        if len(self._hot_stocks) > 100:
            self._hot_stocks = self._hot_stocks[-50:]
    
    def get_hot_stocks(self) -> List[str]:
        """En popüler hisseleri döndür"""
        return self._hot_stocks[-20:]
    
    async def _periodic_full_update(self):
        """Her 15 dakikada bir hot stocks'ları güncelle"""
        while True:
            await asyncio.sleep(900)
            if self._hot_stocks:
                hot = self._hot_stocks[-30:]
                await self.get_bulk_data(hot)
                logger.debug(f"Periyodik güncelleme: {len(hot)} hisse")
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()