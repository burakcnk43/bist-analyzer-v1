# src/domain/services/graham_valuation.py
import math
from typing import Dict, Optional
import logging

from src.domain.models.financials import NetCurrentAssetValue

logger = logging.getLogger(__name__)


class GrahamValuationService:
    """
    Benjamin Graham Değer Yatırımı Metodolojisi.
    
    NCV = Dönen Varlıklar - (KVYK + UVYK)
    NCV/Hisse = NCV / Hisse Sayısı
    
    Graham Kriteri: Hisse Fiyatı < NCV/Hisse * 0.67 ise derin değer fırsatı.
    Hata Toleransı: %0.01 (99.99% doğruluk)
    """
    
    def __init__(self):
        self._ncv_cache: Dict[str, NetCurrentAssetValue] = {}
    
    def calculate_ncv(
        self,
        ticker: str,
        donen_varliklar: float,
        kisa_vadeli_yukumlulukler: float,
        uzun_vadeli_yukumlulukler: float,
        hisse_sayisi: int,
        donem: str = "2025/1"
    ) -> NetCurrentAssetValue:
        """Graham Net-Net Değer Hesaplaması"""
        
        if hisse_sayisi <= 0:
            raise ValueError(f"Hisse sayısı sıfır veya negatif: {ticker}")
        
        ncv = NetCurrentAssetValue(
            ticker=ticker,
            donem=donem,
            donen_varliklar=donen_varliklar,
            kisa_vadeli_yukumlulukler=kisa_vadeli_yukumlulukler,
            uzun_vadeli_yukumlulukler=uzun_vadeli_yukumlulukler,
            hisse_sayisi=hisse_sayisi
        )
        
        logger.info(
            f"{ticker} NCV: {ncv.ncv:,.0f} TL | "
            f"Hisse Başı: {ncv.ncv_per_share:.2f} TL"
        )
        
        return ncv
    
    def margin_of_safety_analysis(
        self,
        ticker: str,
        current_price: float,
        donen_varliklar: float,
        kisa_vadeli_yukumlulukler: float,
        uzun_vadeli_yukumlulukler: float,
        hisse_sayisi: int
    ) -> Dict[str, any]:
        """Kapsamlı güvenlik marjı analizi"""
        
        ncv = self.calculate_ncv(
            ticker, donen_varliklar, kisa_vadeli_yukumlulukler,
            uzun_vadeli_yukumlulukler, hisse_sayisi
        )
        
        ncv_per_share = ncv.ncv_per_share
        
        # Güvenlik marjı hesaplama
        if current_price > 0:
            margin_pct = ((ncv_per_share - current_price) / current_price) * 100
        else:
            margin_pct = 0.0
        
        # Graham sınıflandırması
        buy_threshold = ncv_per_share * 0.67
        
        if current_price <= buy_threshold:
            recommendation = "Derin Değer Fırsatı"
        elif current_price <= ncv_per_share * 0.80:
            recommendation = "Önemli İskonto"
        elif current_price <= ncv_per_share:
            recommendation = "İskontolu"
        elif current_price <= ncv_per_share * 1.20:
            recommendation = "Adil Değer"
        else:
            recommendation = "Pahalı"
        
        # Hassasiyet kontrolü (%99.99 doğruluk)
        if ncv_per_share > 0:
            calculated_ratio = current_price / ncv_per_share
            if math.isclose(calculated_ratio, 0.67, rel_tol=0.0001):
                logger.warning(
                    f"{ticker}: Fiyat/NCV oranı kritik eşikte (%67.00±%0.01). "
                    "Manuel doğrulama önerilir."
                )
        
        return {
            "ticker": ticker,
            "ncv": ncv.ncv,
            "ncv_per_share": round(ncv_per_share, 2),
            "current_price": current_price,
            "margin_of_safety_pct": round(margin_pct, 2),
            "graham_recommendation": recommendation,
            "buy_threshold": round(buy_threshold, 2),
            "price_to_ncv_ratio": round(
                current_price / ncv_per_share if ncv_per_share > 0 else float('inf'), 2
            )
        }