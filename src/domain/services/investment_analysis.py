"""Explainable multi-factor analysis for the BCBIST V2 interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd

from .market_analysis import TechnicalSummary
from .news_analysis import NewsAnalysis, analyze_news


@dataclass(frozen=True)
class InvestmentAnalysis:
    total_score: int
    categories: dict[str, int]
    recommendation: str
    supporting_factors: list[str]
    risks: list[str]
    change_conditions: list[str]
    news: NewsAnalysis
    thesis: "InvestmentThesis"


@dataclass(frozen=True)
class InvestmentThesis:
    attractive_because: str
    strengths: list[str]
    weaknesses: list[str]
    biggest_risks: list[str]
    invalidation: list[str]
    horizon: str
    investor_profile: str


def _latest(frame: pd.DataFrame, labels: list[str]) -> float | None:
    if frame is None or frame.empty:
        return None
    for label in labels:
        if label in frame.index:
            values = frame.loc[label].dropna()
            if not values.empty:
                return float(values.iloc[0])
    return None


def build_investment_analysis(data: dict[str, Any], technical: TechnicalSummary) -> InvestmentAnalysis:
    income, balance, cashflow, info = data.get("income"), data.get("balance"), data.get("cashflow"), data.get("info", {})
    revenue = _latest(income, ["Total Revenue", "Operating Revenue"])
    profit = _latest(income, ["Net Income", "Net Income Common Stockholders"])
    debt = _latest(balance, ["Total Debt", "Long Term Debt And Capital Lease Obligation"])
    equity = _latest(balance, ["Stockholders Equity", "Total Stockholder Equity"])
    cashflow_value = _latest(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    news = analyze_news(data.get("news"))
    strengths: list[str] = []
    risks: list[str] = []

    financial = 45
    if profit is not None and profit > 0: financial += 20; strengths.append("Son erişilebilir dönemde net kâr pozitif.")
    elif profit is not None: financial -= 20; risks.append("Son erişilebilir dönemde net zarar görülüyor.")
    if cashflow_value is not None and cashflow_value > 0: financial += 15; strengths.append("Faaliyet nakit akışı pozitif.")
    elif cashflow_value is not None: financial -= 15; risks.append("Faaliyet nakit akışı negatif.")
    if debt is not None and equity not in (None, 0):
        if debt / equity < 1: financial += 10
        elif debt / equity > 2: financial -= 15; risks.append("Borç / öz kaynak oranı yüksek.")

    technical_score = 45
    if technical.trend == "Yukarı yönlü": technical_score += 25; strengths.append("Fiyat kısa ve orta vadeli ortalamaların üzerinde.")
    elif technical.trend == "Aşağı yönlü": technical_score -= 25; risks.append("Fiyat hareketli ortalamaların altında.")
    if technical.rsi is not None and 50 <= technical.rsi <= 68: technical_score += 15
    elif technical.rsi is not None and technical.rsi >= 75: technical_score -= 15; risks.append("RSI aşırı alım bölgesinde.")
    if technical.macd is not None and technical.macd_signal is not None and technical.macd > technical.macd_signal: technical_score += 10

    momentum = 50
    if technical.momentum_20d is not None:
        momentum += 25 if technical.momentum_20d > 5 else 10 if technical.momentum_20d > 0 else -20 if technical.momentum_20d < -5 else -8
    if technical.volume_ratio is not None and technical.volume_ratio >= 1.2: momentum += 10

    risk = 60
    history = data.get("history", pd.DataFrame())
    if history is not None and not history.empty:
        volatility = history["Close"].pct_change().dropna().tail(60).std() * (252 ** .5) * 100
        if volatility > 55: risk -= 25; risks.append("Tarihsel volatilite yüksek.")
        elif volatility < 30: risk += 10

    growth = 45
    if revenue not in (None, 0) and profit is not None:
        margin = profit / revenue
        if margin >= .10: growth += 25; strengths.append("Net kâr marjı çift haneli.")
        elif margin < 0: growth -= 25
    if cashflow_value is not None and cashflow_value > 0: growth += 10

    valuation = 50
    pe, pb = info.get("trailingPE"), info.get("priceToBook")
    if isinstance(pe, (int, float)) and pe > 0:
        valuation += 15 if pe < 12 else -15 if pe > 30 else 0
    if isinstance(pb, (int, float)) and pb > 0:
        valuation += 10 if pb < 2 else -10 if pb > 6 else 0
    if pe is None and pb is None: risks.append("Değerleme çarpanları sağlanmadı.")

    categories = {"Finansal durum": financial, "Teknik analiz": technical_score, "Haber etkisi": news.score, "Momentum": momentum, "Risk": risk, "Büyüme potansiyeli": growth, "Değerleme": valuation}
    categories = {key: max(0, min(100, int(value))) for key, value in categories.items()}
    weights = {"Finansal durum": .20, "Teknik analiz": .18, "Haber etkisi": .10, "Momentum": .12, "Risk": .15, "Büyüme potansiyeli": .15, "Değerleme": .10}
    total = round(sum(categories[key] * weights[key] for key in categories))
    # Weighted voting plus wide bands reduces churn from a single indicator.
    positive_votes = sum(categories[key] >= 60 for key in ("Finansal durum", "Teknik analiz", "Haber etkisi", "Momentum", "Değerleme", "Risk"))
    negative_votes = sum(categories[key] < 40 for key in ("Finansal durum", "Teknik analiz", "Haber etkisi", "Momentum", "Değerleme", "Risk"))
    recommendation = "AL / Pozitif izleme" if total >= 70 and positive_votes >= 4 and negative_votes <= 1 else "SAT / Temkinli izleme" if total <= 35 and negative_votes >= 3 else "TUT / Nötr izleme"
    conditions = ["Teknik trend iki ardışık değerlendirmede aşağı yönlü kalırsa.", "Finansal tablo veya haber akışı temel varsayımları bozarsa.", "Volatilite artıp risk skoru anlamlı biçimde düşerse."]
    final_strengths = strengths or ["Belirgin destekleyici veri sınırlı."]
    final_risks = risks or ["Öne çıkan hesaplanmış risk bulunmadı; bu risksiz olduğu anlamına gelmez."]
    horizon = "Orta-uzun vade" if categories["Finansal durum"] >= 60 and categories["Büyüme potansiyeli"] >= 60 else "Kısa-orta vade"
    profile = "Conservative" if categories["Risk"] >= 70 and total >= 55 else "Aggressive" if categories["Risk"] < 45 else "Balanced"
    thesis = InvestmentThesis(
        attractive_because=f"Genel skor {total}/100; en güçlü katkılar: {', '.join(sorted(categories, key=categories.get, reverse=True)[:2])}.",
        strengths=final_strengths,
        weaknesses=[risk for risk in final_risks if "volatilite" not in risk.lower()] or final_risks[:1],
        biggest_risks=final_risks,
        invalidation=conditions,
        horizon=horizon,
        investor_profile=profile,
    )
    return InvestmentAnalysis(total, categories, recommendation, final_strengths, final_risks, conditions, news, thesis)
