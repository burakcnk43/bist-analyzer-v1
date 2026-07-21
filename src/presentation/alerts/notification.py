# src/presentation/alerts/notification.py
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class AlertThreshold:
    """Uyarı eşik değeri"""
    ticker: str
    metric: str
    operator: str
    value: float
    triggered_at: Optional[datetime] = None


class AlertManager:
    """Fiyat ve temel analiz eşik ihlali uyarı sistemi."""
    
    def __init__(self):
        self._thresholds: List[AlertThreshold] = []
        self._triggered: List[AlertThreshold] = []
    
    def add_threshold(self, ticker: str, metric: str, operator: str, value: float):
        """Yeni bir eşik değeri ekle"""
        threshold = AlertThreshold(
            ticker=ticker.upper(),
            metric=metric,
            operator=operator,
            value=value
        )
        self._thresholds.append(threshold)
        logger.info(f"Uyarı eklendi: {ticker} {metric} {operator} {value}")
    
    def remove_threshold(self, ticker: str, metric: str):
        """Eşik değerini kaldır"""
        self._thresholds = [
            t for t in self._thresholds
            if not (t.ticker == ticker.upper() and t.metric == metric)
        ]
    
    def check_thresholds(self, ticker: str, current_data: Dict[str, float]) -> List[AlertThreshold]:
        """Tüm eşik değerlerini kontrol et"""
        triggered = []
        
        for threshold in self._thresholds:
            if threshold.ticker != ticker.upper():
                continue
            
            current_value = current_data.get(threshold.metric)
            if current_value is None:
                continue
            
            is_triggered = False
            if threshold.operator == "above":
                is_triggered = current_value > threshold.value
            elif threshold.operator == "below":
                is_triggered = current_value < threshold.value
            
            if is_triggered and threshold not in self._triggered:
                threshold.triggered_at = datetime.now()
                self._triggered.append(threshold)
                triggered.append(threshold)
                logger.warning(f"⚠️ UYARI: {ticker} {threshold.metric} {threshold.operator} {threshold.value}")
            elif not is_triggered and threshold in self._triggered:
                self._triggered.remove(threshold)
        
        return triggered
    
    def get_active_alerts(self) -> List[AlertThreshold]:
        """Aktif uyarıları döndür"""
        return self._triggered.copy()


class PortfolioRiskManager:
    """Portföy risk analitiği."""
    
    def __init__(self):
        self._portfolio: Dict[str, Dict] = {}
    
    def add_position(self, ticker: str, shares: int, avg_cost: float, sector: str):
        """Portföye pozisyon ekle"""
        self._portfolio[ticker.upper()] = {
            "shares": shares,
            "avg_cost": avg_cost,
            "sector": sector
        }
    
    def calculate_sector_exposure(self) -> Dict[str, float]:
        """Sektör bazlı portföy ağırlığı (%)"""
        total_value = sum(p["shares"] * p["avg_cost"] for p in self._portfolio.values())
        if total_value == 0:
            return {}
        
        exposure = {}
        for ticker, position in self._portfolio.items():
            sector = position["sector"]
            value = position["shares"] * position["avg_cost"]
            weight = (value / total_value) * 100
            exposure[sector] = exposure.get(sector, 0) + weight
        
        return exposure
    
    def concentration_risk(self) -> Dict[str, any]:
        """Yoğunlaşma riski analizi"""
        exposure = self.calculate_sector_exposure()
        
        if not exposure:
            return {"risk_level": "N/A", "warning": "Portföy boş"}
        
        max_sector = max(exposure, key=exposure.get)
        max_weight = exposure[max_sector]
        
        if max_weight > 30:
            risk_level = "YÜKSEK"
            warning = f"{max_sector} sektöründe %{max_weight:.1f} yoğunlaşma"
        elif max_weight > 20:
            risk_level = "ORTA"
            warning = f"{max_sector} sektöründe %{max_weight:.1f} yoğunlaşma"
        else:
            risk_level = "DÜŞÜK"
            warning = "Sektör dağılımı dengeli"
        
        return {
            "risk_level": risk_level,
            "warning": warning,
            "max_sector": max_sector,
            "max_weight": round(max_weight, 1),
            "exposure": exposure
        }