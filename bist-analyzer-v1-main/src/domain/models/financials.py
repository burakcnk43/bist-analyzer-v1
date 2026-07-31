# src/domain/models/financials.py
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class NetCurrentAssetValue:
    """Graham Net Current Asset Value (Net-Net) Hesaplama Modeli"""
    ticker: str
    donem: str
    donen_varliklar: float
    kisa_vadeli_yukumlulukler: float
    uzun_vadeli_yukumlulukler: float
    hisse_sayisi: int
    
    @property
    def ncv(self) -> float:
        """NCV = Dönen Varlıklar - (KVYK + UVYK)"""
        return self.donen_varliklar - (self.kisa_vadeli_yukumlulukler + self.uzun_vadeli_yukumlulukler)
    
    @property
    def ncv_per_share(self) -> float:
        """Hisse başına net çalışma sermayesi."""
        if self.hisse_sayisi <= 0:
            return 0.0
        return self.ncv / self.hisse_sayisi
    
    def margin_of_safety(self, market_price: float) -> float:
        """Güvenlik marjı = (NCV/Hisse - Piyasa Fiyatı) / Piyasa Fiyatı * 100"""
        if market_price <= 0:
            return 0.0
        return ((self.ncv_per_share - market_price) / market_price) * 100


@dataclass(frozen=True)
class FinancialScoreCard:
    """Finansal sağlık skor kartı (0-10 ⭐)"""
    ticker: str
    oz_sermaye_buyumesi: float
    nakit_akisi_pozitif: bool
    ncv_iskonto_orani: float
    roe: float
    fk_orani: float
    pd_dd: float
    sektor: str
    
    def score_roe(self) -> int:
        """ROE: 15+ iyi, 20+ çok iyi, 30+ mükemmel"""
        if self.roe >= 30:
            return 10
        elif self.roe >= 20:
            return 8
        elif self.roe >= 15:
            return 6
        elif self.roe >= 10:
            return 4
        elif self.roe > 0:
            return 2
        else:
            return 0
    
    def score_fk(self) -> int:
        """F/K sektör ayarlı: 5-15 cazip, 15-25 normal, 25+ pahalı"""
        if 5 <= self.fk_orani <= 15:
            return 9
        elif 15 < self.fk_orani <= 25:
            return 6
        elif 25 < self.fk_orani <= 35:
            return 3
        elif self.fk_orani > 35:
            return 1
        else:
            return 0