# src/domain/services/ratio_analyzer.py
import math
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RatioAnalyzer:
    """
    Finansal Rasyo Analiz Motoru.
    
    Kritik Metrikler:
    - ROE (Özkaynak Kârlılığı): %15+ İyi, %20+ Çok İyi, %30+ Mükemmel
    - F/K (Fiyat/Kazanç): 5-15 Cazip, 15-25 Normal, 25+ Pahalı
    - PD/DD: <1.0 İskontolu, 1.0-2.0 Adil, >2.0 Primli
    - Cari Oran: 1.5+ Sağlıklı
    - Borç/Özsermaye: <1.0 İdeal
    
    Matematiksel Hassasiyet: Tüm oranlar 2 ondalık basamakla hesaplanır.
    """
    
    def calculate_roe(self, net_kar: float, oz_sermaye: float) -> float:
        """
        Özkaynak Kârlılığı (Return on Equity).
        ROE = (Net Kâr / Öz Sermaye) * 100
        
        Değerlendirme:
        - %30+ : Mükemmel (10/10)
        - %20-30: Çok İyi (8/10)
        - %15-20: İyi (6/10)
        - %10-15: Orta (4/10)
        - %0-10: Zayıf (2/10)
        - Negatif: Riskli (0/10)
        """
        if oz_sermaye == 0:
            return 0.0
        
        roe = (net_kar / oz_sermaye) * 100
        return round(roe, 2)
    
    def score_roe(self, roe: float) -> int:
        """ROE'ye göre 0-10 arası puan"""
        if roe >= 30:
            return 10
        elif roe >= 20:
            return 8
        elif roe >= 15:
            return 6
        elif roe >= 10:
            return 4
        elif roe > 0:
            return 2
        else:
            return 0
    
    def calculate_pe_ratio(self, market_price: float, net_kar: float, hisse_sayisi: int) -> float:
        """
        Fiyat/Kazanç Oranı (Price/Earnings).
        F/K = Piyasa Fiyatı / Hisse Başı Kâr
        
        Sektör Ayarlı Değerlendirme:
        - 5-15: Cazip (9/10)
        - 15-25: Normal (6/10)
        - 25-35: Pahalı (3/10)
        - 35+: Çok Pahalı (1/10)
        - Negatif: Zarar (0/10)
        """
        if hisse_sayisi <= 0:
            return 0.0
        
        eps = net_kar / hisse_sayisi
        
        if eps <= 0:
            return float('inf')
        
        pe_ratio = market_price / eps
        return round(pe_ratio, 2)
    
    def score_pe(self, pe_ratio: float) -> int:
        """F/K'ya göre 0-10 arası puan"""
        if pe_ratio == float('inf') or pe_ratio <= 0:
            return 0
        elif 5 <= pe_ratio <= 15:
            return 9
        elif 15 < pe_ratio <= 25:
            return 6
        elif 25 < pe_ratio <= 35:
            return 3
        else:
            return 1
    
    def calculate_pb_ratio(self, market_price: float, oz_sermaye: float, hisse_sayisi: int) -> float:
        """
        Piyasa Değeri / Defter Değeri (PD/DD).
        PD/DD = Piyasa Fiyatı / (Öz Sermaye / Hisse Sayısı)
        """
        if hisse_sayisi <= 0:
            return 0.0
        
        defter_degeri = oz_sermaye / hisse_sayisi
        
        if defter_degeri <= 0:
            return float('inf')
        
        pb = market_price / defter_degeri
        return round(pb, 2)
    
    def score_pb(self, pb_ratio: float) -> int:
        """PD/DD'ye göre 0-10 arası puan"""
        if pb_ratio == float('inf') or pb_ratio <= 0:
            return 0
        elif pb_ratio <= 1.0:
            return 9
        elif pb_ratio <= 2.0:
            return 6
        elif pb_ratio <= 3.0:
            return 3
        else:
            return 1
    
    def calculate_current_ratio(self, donen_varliklar: float, kisa_vadeli_yukumlulukler: float) -> float:
        """
        Cari Oran = Dönen Varlıklar / Kısa Vadeli Yükümlülükler.
        
        - 2.0+: Çok Güçlü
        - 1.5-2.0: Güçlü
        - 1.0-1.5: Yeterli
        - <1.0: Likidite Riski
        """
        if kisa_vadeli_yukumlulukler == 0:
            return float('inf') if donen_varliklar > 0 else 0.0
        
        return round(donen_varliklar / kisa_vadeli_yukumlulukler, 2)
    
    def calculate_debt_to_equity(
        self,
        kisa_vadeli_yukumlulukler: float,
        uzun_vadeli_yukumlulukler: float,
        oz_sermaye: float
    ) -> float:
        """
        Borç/Özsermaye Oranı.
        
        - <0.5: Düşük Kaldıraç (İdeal)
        - 0.5-1.0: Orta Kaldıraç
        - 1.0-2.0: Yüksek Kaldıraç
        - >2.0: Riskli
        """
        toplam_borc = kisa_vadeli_yukumlulukler + uzun_vadeli_yukumlulukler
        
        if oz_sermaye == 0:
            return float('inf') if toplam_borc > 0 else 0.0
        
        return round(toplam_borc / oz_sermaye, 2)
    
    def get_roe_label(self, roe: float) -> str:
        """ROE seviyesine göre etiket"""
        if roe >= 30:
            return "Mükemmel"
        elif roe >= 20:
            return "Çok İyi"
        elif roe >= 15:
            return "İyi"
        elif roe >= 10:
            return "Orta"
        elif roe > 0:
            return "Zayıf"
        else:
            return "Riskli"
    
    def get_pe_label(self, pe_ratio: float) -> str:
        """F/K seviyesine göre etiket"""
        if pe_ratio == float('inf') or pe_ratio <= 0:
            return "Zarar"
        elif 5 <= pe_ratio <= 15:
            return "Cazip"
        elif 15 < pe_ratio <= 25:
            return "Normal"
        elif 25 < pe_ratio <= 35:
            return "Pahalı"
        else:
            return "Çok Pahalı"