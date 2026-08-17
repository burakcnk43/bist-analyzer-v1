# src/domain/interfaces/repository.py
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import pandas as pd


class IFinancialRepository(ABC):
    """Finansal veri erişimi için soyut temel sınıf."""
    
    @abstractmethod
    async def get_balance_sheet(self, ticker: str, year: int, quarter: int) -> Dict[str, Any]:
        """Bilanço verilerini getirir."""
        pass
    
    @abstractmethod
    async def get_income_statement(self, ticker: str, year: int, quarter: int) -> Dict[str, Any]:
        """Gelir tablosu verilerini getirir."""
        pass
    
    @abstractmethod
    async def get_cash_flow_statement(self, ticker: str, year: int, quarter: int) -> Dict[str, Any]:
        """Nakit akış tablosu verilerini getirir."""
        pass
    
    @abstractmethod
    async def get_financial_ratios(self, ticker: str) -> Dict[str, float]:
        """Temel finansal rasyoları getirir."""
        pass
    
    @abstractmethod
    async def get_historical_financials(self, ticker: str, years_back: int = 5) -> pd.DataFrame:
        """Çok yıllı finansal veri trendi."""
        pass


class ITechnicalRepository(ABC):
    """Teknik gösterge verileri için soyut sınıf."""
    
    @abstractmethod
    async def get_ohlcv(self, ticker: str, start_date: date, end_date: date, interval: str = "1d") -> pd.DataFrame:
        """OHLCV fiyat verisi."""
        pass
    
    @abstractmethod
    async def get_indicators(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """Tüm teknik göstergeler (RSI, MACD, Ichimoku, Bollinger, VWAP, Fibonacci)."""
        pass


class IBISTIndexRepository(ABC):
    """BIST 100 endeksi ve piyasa verileri."""
    
    @abstractmethod
    async def get_bist100_composition(self) -> List[Dict[str, Any]]:
        """Güncel BIST 100 hisse listesi."""
        pass
    
    @abstractmethod
    async def get_sector_distribution(self) -> Dict[str, float]:
        """BIST 100 sektör dağılımı."""
        pass
    
    @abstractmethod
    async def get_market_multipliers(self) -> Dict[str, float]:
        """Piyasa geneli çarpanlar (F/K, PD/DD, faiz)."""
        pass