"""Transparent, provider-agnostic analysis of available news headlines.

This is deliberately not a price prediction.  It turns the news metadata
already supplied by the data provider into explainable context and is designed
to be replaced by an LLM/news API adapter later without changing the UI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


POSITIVE = ("kar", "kâr", "büyü", "ihale", "sözleşme", "yatırım", "temettü", "artış", "onay", "rekor", "buyback")
NEGATIVE = ("zarar", "ceza", "soruştur", "dava", "borç", "düşüş", "iptal", "risk", "uyarı", "downgrade")
TRUSTED_SOURCES = ("kap", "reuters", "bloomberg", "aa", "anadolu", "foreks")


@dataclass(frozen=True)
class NewsAnalysis:
    score: int
    confidence: int
    sentiment: str
    summary: str
    short_term: str
    long_term: str
    positive_factors: list[str]
    negative_factors: list[str]
    why_it_matters: list[str]


def _field(item: dict[str, Any], name: str, default: Any = "") -> Any:
    content = item.get("content", item)
    return content.get(name, item.get(name, default)) if isinstance(content, dict) else default


def analyze_news(items: list[dict[str, Any]] | None) -> NewsAnalysis:
    items = items or []
    if not items:
        return NewsAnalysis(50, 15, "Veri yok", "Güncel haber kaydı bulunamadı.", "Haber etkisi ölçülemedi.", "Haber etkisi ölçülemedi.", [], [], ["Haber olmadığı için skor nötr tutuldu; bu olumlu bir sinyal değildir."])
    positive = negative = trusted = 0
    reasons: list[str] = []
    positive_factors: list[str] = []
    negative_factors: list[str] = []
    for item in items[:8]:
        title = str(_field(item, "title", ""))
        provider = _field(item, "provider", {})
        provider_name = str(provider.get("displayName", "") if isinstance(provider, dict) else provider).lower()
        text = title.lower()
        pos = sum(term in text for term in POSITIVE)
        neg = sum(term in text for term in NEGATIVE)
        positive += pos
        negative += neg
        if any(source in provider_name for source in TRUSTED_SOURCES):
            trusted += 1
        if pos or neg:
            explanation = f"{title}: {'olumlu' if pos > neg else 'olumsuz'} olası etki; başlık tek başına kesin sonuç değildir."
            reasons.append(explanation)
            (positive_factors if pos > neg else negative_factors).append(title)
    score = max(0, min(100, 50 + (positive - negative) * 10))
    confidence = min(85, 25 + len(items[:8]) * 5 + trusted * 5)
    if positive > negative:
        sentiment, direction = "Olumlu eğilim", "Kısa vadede ilgi ve volatiliteyi artırabilir."
    elif negative > positive:
        sentiment, direction = "Olumsuz eğilim", "Kısa vadede belirsizlik ve satış baskısı oluşturabilir."
    else:
        sentiment, direction = "Nötr / karışık", "Başlıklar yön konusunda ortak bir sinyal vermiyor."
    summary = f"Son {min(8, len(items))} haber başlığında {sentiment.lower()} görüldü."
    long_term = "Kalıcı etki için açıklamanın finansallara, sözleşmeye veya düzenleyici sonuca yansıması izlenmelidir."
    return NewsAnalysis(score, confidence, sentiment, summary, direction, long_term, positive_factors, negative_factors, reasons or ["Başlıklarda tanımlı bir etki anahtarı bulunmadı."])
