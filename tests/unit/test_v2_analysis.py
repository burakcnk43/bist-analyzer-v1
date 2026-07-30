import pandas as pd

from src.data.bist_universe import BIST_30_TICKERS, BIST_100_TICKERS, BIST_TICKERS
from src.domain.services.investment_analysis import build_investment_analysis
from src.domain.services.market_analysis import calculate_technicals
from src.domain.services.news_analysis import analyze_news


def history() -> pd.DataFrame:
    close = pd.Series(range(100, 180), dtype=float)
    return pd.DataFrame({"Close": close, "High": close + 2, "Low": close - 2, "Volume": 1_000_000})


def test_market_scopes_are_explicit_subsets():
    assert len(BIST_30_TICKERS) == 30
    assert len(BIST_100_TICKERS) == 100
    assert set(BIST_30_TICKERS).issubset(BIST_TICKERS)
    assert set(BIST_100_TICKERS).issubset(BIST_TICKERS)


def test_news_analysis_explains_positive_headline():
    result = analyze_news([{"title": "Şirket yeni ihale sözleşmesi ve rekor kâr açıkladı", "publisher": "Reuters"}])
    assert result.score > 50
    assert result.confidence > 25
    assert result.why_it_matters


def test_multifactor_score_returns_all_categories():
    prices = history()
    technical = calculate_technicals(prices)
    data = {
        "history": prices,
        "info": {"trailingPE": 10, "priceToBook": 1.5},
        "income": pd.DataFrame({"latest": [1_000.0, 150.0]}, index=["Total Revenue", "Net Income"]),
        "balance": pd.DataFrame({"latest": [100.0, 500.0]}, index=["Total Debt", "Stockholders Equity"]),
        "cashflow": pd.DataFrame({"latest": [120.0]}, index=["Operating Cash Flow"]),
        "news": [{"title": "Yeni yatırım sözleşmesi açıklandı", "publisher": "Reuters"}],
    }
    result = build_investment_analysis(data, technical)
    assert 0 <= result.total_score <= 100
    assert set(result.categories) == {"Finansal durum", "Teknik analiz", "Haber etkisi", "Momentum", "Risk", "Büyüme potansiyeli", "Değerleme"}
    assert result.recommendation in {"AL / Pozitif izleme", "TUT / Nötr izleme", "SAT / Temkinli izleme"}
    assert result.thesis.investor_profile in {"Conservative", "Balanced", "Aggressive"}


def test_recommendation_is_stable_for_small_price_move():
    prices = history()
    data = {"history": prices, "info": {}, "income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame(), "news": []}
    first = build_investment_analysis(data, calculate_technicals(prices))
    moved = prices.copy()
    moved.loc[moved.index[-1], "Close"] *= 1.002
    second = build_investment_analysis({**data, "history": moved}, calculate_technicals(moved))
    assert first.recommendation == second.recommendation
