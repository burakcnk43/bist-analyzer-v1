# src/domain/services/confidence_engine.py
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from src.domain.services.graham_valuation import GrahamValuationService
from src.domain.services.ratio_analyzer import RatioAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceScore:
    """0-100 Güven Skoru Modeli"""
    ticker: str
    total_score: float
    breakdown: Dict[str, float]
    stars: Dict[str, int]
    short_term_rating: int
    long_term_rating: int
    strengths: List[str]
    risks: List[str]
    commentary: str
    timestamp: str


class ConfidenceEngine:
    """
    Çok Faktörlü Güven Skorlama Motoru.
    
    Ağırlıklar (Toplam 100):
    - Finansal Sağlık (Öz Sermaye, Nakit Akışı): 25 puan
    - Değerleme (NCV, F/K, PD/DD): 30 puan
    - Kârlılık (ROE): 20 puan
    - Büyüme (Öz Sermaye Trendi): 15 puan
    - Teknik Göstergeler (RSI, Trend): 10 puan
    
    Hiçbir sinyal "AL/SAT" içermez. Sadece güven skoru ve yorum.
    """
    
    WEIGHT_FINANCIAL_HEALTH = 0.25
    WEIGHT_VALUATION = 0.30
    WEIGHT_PROFITABILITY = 0.20
    WEIGHT_GROWTH = 0.15
    WEIGHT_TECHNICAL = 0.10
    
    def __init__(self):
        self.graham = GrahamValuationService()
        self.ratios = RatioAnalyzer()
    
    def analyze(
        self,
        ticker: str,
        current_price: float,
        donen_varliklar: float,
        kisa_vadeli_yukumlulukler: float,
        uzun_vadeli_yukumlulukler: float,
        oz_sermaye: float,
        net_kar: float,
        hisse_sayisi: int,
        isletme_nakit_akisi: float,
        sector: str = "Genel",
        rsi: float = 50,
        trend: str = "neutral"
    ) -> ConfidenceScore:
        """Tam kapsamlı güven skoru analizi."""
        
        breakdown = {}
        stars = {}
        strengths = []
        risks = []
        
        # 1. FİNANSAL SAĞLIK (25 puan)
        health_score = self._score_financial_health(
            oz_sermaye, isletme_nakit_akisi
        )
        breakdown["Finansal Sağlık"] = health_score
        
        if oz_sermaye > 0:
            stars["Öz Sermaye"] = 8
            strengths.append("Öz sermaye pozitif")
        else:
            stars["Öz Sermaye"] = 0
            risks.append("Öz sermaye negatif")
        
        if isletme_nakit_akisi > 0:
            stars["Nakit Akışı"] = 8
            strengths.append("İşletme nakit akışı pozitif")
        else:
            stars["Nakit Akışı"] = 2
            risks.append("İşletme nakit akışı negatif")
        
        # 2. DEĞERLEME (30 puan)
        val_score = self._score_valuation(
            ticker, current_price, donen_varliklar,
            kisa_vadeli_yukumlulukler, uzun_vadeli_yukumlulukler,
            hisse_sayisi, net_kar, oz_sermaye
        )
        breakdown["Değerleme"] = val_score
        
        # NCV analizi
        ncv_analysis = self.graham.margin_of_safety_analysis(
            ticker, current_price, donen_varliklar,
            kisa_vadeli_yukumlulukler, uzun_vadeli_yukumlulukler,
            hisse_sayisi
        )
        margin = ncv_analysis["margin_of_safety_pct"]
        
        if margin > 20:
            stars["NCV Değeri"] = 9
            strengths.append(f"NCV'ye göre %{margin:.1f} iskontolu")
        elif margin > 0:
            stars["NCV Değeri"] = 7
        elif margin > -20:
            stars["NCV Değeri"] = 4
        else:
            stars["NCV Değeri"] = 2
            risks.append(f"NCV'ye göre %{abs(margin):.1f} primli")
        
        # F/K skoru
        pe = self.ratios.calculate_pe_ratio(current_price, net_kar, hisse_sayisi)
        pe_score = self.ratios.score_pe(pe)
        stars["F/K Oranı"] = pe_score
        
        if pe_score >= 9:
            strengths.append(f"F/K oranı cazip seviyede: {pe:.1f}")
        elif pe_score <= 3:
            risks.append(f"F/K oranı yüksek: {pe:.1f}")
        
        # 3. KÂRLILIK (20 puan)
        roe = self.ratios.calculate_roe(net_kar, oz_sermaye)
        roe_score = self.ratios.score_roe(roe)
        prof_score = (roe_score / 10) * 100 * self.WEIGHT_PROFITABILITY
        breakdown["Kârlılık"] = prof_score
        stars["ROE"] = roe_score
        
        if roe >= 20:
            strengths.append(f"Yüksek özkaynak kârlılığı: %{roe:.1f}")
        elif roe < 10:
            risks.append(f"Düşük özkaynak kârlılığı: %{roe:.1f}")
        
        # 4. BÜYÜME (15 puan) - Basitleştirilmiş
        growth_score = 10 if oz_sermaye > 0 and net_kar > 0 else 5
        breakdown["Büyüme"] = (growth_score / 10) * 100 * self.WEIGHT_GROWTH
        stars["Büyüme"] = growth_score
        
        # 5. TEKNİK (10 puan)
        tech_score = self._score_technical(rsi, trend)
        breakdown["Teknik"] = tech_score
        stars["Teknik"] = self._to_star_rating(rsi_score=7 if 40 <= rsi <= 60 else 5)
        
        # TOPLAM SKOR
        total_score = sum(breakdown.values())
        total_score = max(0.0, min(100.0, total_score))
        
        # KISA/UZUN VADE
        short_term = 3 if trend == "bullish" else 2 if trend == "neutral" else 1
        long_term = 4 if total_score > 60 else 3 if total_score > 40 else 2
        
        # YORUM
        commentary = self._generate_commentary(
            ticker, total_score, strengths, risks, sector
        )
        
        return ConfidenceScore(
            ticker=ticker,
            total_score=round(total_score, 2),
            breakdown=breakdown,
            stars=stars,
            short_term_rating=short_term,
            long_term_rating=long_term,
            strengths=strengths,
            risks=risks,
            commentary=commentary,
            timestamp=datetime.now().isoformat()
        )
    
    def _score_financial_health(self, oz_sermaye: float, nakit_akisi: float) -> float:
        """Finansal sağlık skoru"""
        equity_score = 8 if oz_sermaye > 0 else 2
        cash_score = 8 if nakit_akisi > 0 else 2
        return ((equity_score * 0.6 + cash_score * 0.4) / 10) * 100 * self.WEIGHT_FINANCIAL_HEALTH
    
    def _score_valuation(
        self, ticker: str, current_price: float,
        donen_varliklar: float, kvyk: float, uvyk: float,
        hisse_sayisi: int, net_kar: float, oz_sermaye: float
    ) -> float:
        """Değerleme skoru"""
        # NCV
        ncv = self.graham.margin_of_safety_analysis(
            ticker, current_price, donen_varliklar, kvyk, uvyk, hisse_sayisi
        )
        margin = ncv["margin_of_safety_pct"]
        ncv_score = 9 if margin > 20 else 7 if margin > 0 else 4 if margin > -20 else 2
        
        # F/K
        pe = self.ratios.calculate_pe_ratio(current_price, net_kar, hisse_sayisi)
        pe_score = self.ratios.score_pe(pe)
        
        # PD/DD
        pb = self.ratios.calculate_pb_ratio(current_price, oz_sermaye, hisse_sayisi)
        pb_score = self.ratios.score_pb(pb)
        
        return ((ncv_score * 0.4 + pe_score * 0.4 + pb_score * 0.2) / 10) * 100 * self.WEIGHT_VALUATION
    
    def _score_technical(self, rsi: float, trend: str) -> float:
        """Teknik skor"""
        if 40 <= rsi <= 60:
            rsi_score = 7
        elif 30 <= rsi < 40 or 60 < rsi <= 70:
            rsi_score = 5
        else:
            rsi_score = 3
        
        trend_score = 8 if trend == "bullish" else 5 if trend == "neutral" else 3
        
        tech_avg = (rsi_score + trend_score) / 2
        return (tech_avg / 10) * 100 * self.WEIGHT_TECHNICAL
    
    def _generate_commentary(
        self, ticker: str, total_score: float,
        strengths: List[str], risks: List[str], sector: str
    ) -> str:
        """İnsan benzeri yorum"""
        if total_score >= 80:
            comment = f"{ticker}, {sector} sektöründe yüksek güven skoruna sahip."
        elif total_score >= 60:
            comment = f"{ticker} için orta-üst seviye güven skoru."
        elif total_score >= 40:
            comment = f"{ticker} ortalama bir profil çiziyor."
        elif total_score >= 20:
            comment = f"{ticker} için temkinli yaklaşım önerilir."
        else:
            comment = f"{ticker} yüksek risk profili gösteriyor."
        
        if strengths:
            comment += f" ✅ {'; '.join(strengths[:2])}."
        if risks:
            comment += f" ⚠️ {'; '.join(risks[:2])}."
        
        return comment
    
    @staticmethod
    def _to_star_rating(rsi_score: int = 5) -> int:
        """0-10 arası yıldız derecelendirme"""
        return min(10, max(0, rsi_score))