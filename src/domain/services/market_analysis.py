"""Doğrulanabilir piyasa verisini teknik analiz özetine dönüştüren yardımcılar.

Bu modül yatırım tavsiyesi üretmez. Yorumlar yalnızca hesaplanan göstergelerin
durumunu açıklar; veri yoksa sonuç üretmek yerine bunu açıkça bildirir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TechnicalSummary:
    trend: str
    rsi: float | None
    macd: float | None
    macd_signal: float | None
    sma_20: float | None
    sma_50: float | None
    ema_20: float | None
    bollinger_upper: float | None
    bollinger_lower: float | None
    momentum_20d: float | None
    volume_ratio: float | None
    atr_14: float | None
    support: float | None
    resistance: float | None
    explanations: list[str]


def _number(value: Any) -> float | None:
    """Convert a pandas/numpy value to float without leaking NaN to the UI."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def _last(series: pd.Series) -> float | None:
    return _number(series.iloc[-1]) if not series.empty else None


def calculate_technicals(history: pd.DataFrame) -> TechnicalSummary:
    """Calculate commonly used indicators from an OHLCV dataframe.

    The calculation needs at least 60 daily bars for a complete summary. Partial
    data remains useful, but unavailable indicators are explicitly returned as
    ``None`` instead of being guessed.
    """
    required = {"Close", "High", "Low", "Volume"}
    if history is None or history.empty or not required.issubset(history.columns):
        return TechnicalSummary("Veri yetersiz", None, None, None, None, None, None,
                                None, None, None, None, None, None, None,
                                ["Teknik analiz için yeterli fiyat verisi alınamadı."])

    frame = history.copy().dropna(subset=["Close"])
    close = frame["Close"].astype(float)
    high = frame["High"].astype(float)
    low = frame["Low"].astype(float)
    volume = frame["Volume"].astype(float)

    delta = close.diff()
    # Wilder smoothing is the conventional RSI calculation and reacts more
    # naturally than a simple rolling average when new prices arrive.
    gains = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = gains / losses.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    # Flat loss/gain windows are meaningful edge cases: a window containing
    # gains but no losses has RSI 100; the inverse has RSI 0.
    rsi = rsi.mask((losses == 0) & (gains > 0), 100.0)
    rsi = rsi.mask((gains == 0) & (losses > 0), 0.0)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    sma_20_series = close.rolling(20, min_periods=20).mean()
    sma_50_series = close.rolling(50, min_periods=50).mean()
    ema_20_series = close.ewm(span=20, adjust=False).mean()
    rolling_std = close.rolling(20, min_periods=20).std()
    upper_band = sma_20_series + (2 * rolling_std)
    lower_band = sma_20_series - (2 * rolling_std)
    momentum = close.pct_change(20) * 100
    volume_average = volume.rolling(20, min_periods=20).mean()
    volume_ratio = volume / volume_average.replace(0, np.nan)
    previous_close = close.shift(1)
    true_range = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
    atr_14 = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    price = _last(close)
    last_sma20, last_sma50 = _last(sma_20_series), _last(sma_50_series)
    if price is None or last_sma20 is None:
        trend = "Veri yetersiz"
    elif last_sma50 is not None and price > last_sma20 > last_sma50:
        trend = "Yukarı yönlü"
    elif last_sma50 is not None and price < last_sma20 < last_sma50:
        trend = "Aşağı yönlü"
    else:
        trend = "Yatay / karışık"

    last_rsi = _last(rsi)
    last_macd, last_signal = _last(macd_line), _last(macd_signal)
    notes: list[str] = [f"Fiyatın hareketli ortalamalara göre görünümü: {trend}."]
    if last_rsi is not None:
        if last_rsi >= 70:
            notes.append(f"RSI {last_rsi:.1f}: aşırı alım eşiği üzerindedir; tek başına satış sinyali değildir.")
        elif last_rsi <= 30:
            notes.append(f"RSI {last_rsi:.1f}: aşırı satım eşiği altındadır; tek başına alım sinyali değildir.")
        else:
            notes.append(f"RSI {last_rsi:.1f}: nötr bölgede yer alıyor.")
    if last_macd is not None and last_signal is not None:
        direction = "üzerinde" if last_macd > last_signal else "altında"
        notes.append(f"MACD, sinyal çizgisinin {direction}; momentum bu göstergeye göre izlenmelidir.")
    last_volume_ratio = _last(volume_ratio)
    if last_volume_ratio is not None:
        notes.append(f"Son gün hacmi, 20 günlük ortalamanın {last_volume_ratio:.2f} katı.")

    # 20-day levels are a more actionable short/medium-term reference than an
    # old 60-day extreme, while the UI clearly labels them as observation levels.
    lookback = min(20, len(frame))
    return TechnicalSummary(
        trend=trend,
        rsi=last_rsi,
        macd=last_macd,
        macd_signal=last_signal,
        sma_20=last_sma20,
        sma_50=last_sma50,
        ema_20=_last(ema_20_series),
        bollinger_upper=_last(upper_band),
        bollinger_lower=_last(lower_band),
        momentum_20d=_last(momentum),
        volume_ratio=last_volume_ratio,
        atr_14=_last(atr_14),
        support=_number(low.tail(lookback).min()),
        resistance=_number(high.tail(lookback).max()),
        explanations=notes,
    )


def score_opportunity(summary: TechnicalSummary) -> tuple[int, list[str]]:
    """Return a transparent watch-list score, not an investment recommendation."""
    score = 0
    reasons: list[str] = []
    if summary.trend == "Yukarı yönlü":
        score += 35
        reasons.append("Fiyat kısa ve orta vadeli ortalamaların üzerinde.")
    if summary.rsi is not None and 55 <= summary.rsi < 68:
        score += 25
        reasons.append("RSI pozitif bölgede ve aşırı alım eşiğinin altında.")
    elif summary.rsi is not None and summary.rsi >= 75:
        score -= 10
        reasons.append("RSI çok yüksek; kısa vadeli aşırı alım riski puanı düşürdü.")
    if summary.macd is not None and summary.macd_signal is not None and summary.macd > summary.macd_signal:
        score += 20
        reasons.append("MACD sinyal çizgisinin üzerinde.")
    if summary.volume_ratio is not None and summary.volume_ratio >= 1.2:
        score += 20
        reasons.append("Hacim 20 günlük ortalamanın belirgin üzerinde.")
    return max(0, min(100, score)), reasons
