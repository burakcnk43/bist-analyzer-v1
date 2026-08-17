"""BCBIST AI V2 — doğrulanabilir veri odaklı Streamlit arayüzü.

Çalıştırma: streamlit run src/presentation/dashboard/v2_app.py
"""

from __future__ import annotations


import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domain.services.market_analysis import calculate_technicals, score_opportunity

st.set_page_config(page_title="BCBIST AI", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

from src.data.news import get_news

from src.data.bist_universe import (
    BIST30_TICKERS,
    BIST100_TICKERS,
    BIST_TICKERS,
)
def analyze_financial_quality(symbol: str) -> dict:
    """
    Uzun vadeli finansal kalite puanlama motoru.
    0-100 arası puan üretir.
    """

    try:
        ticker = yf.Ticker(f"{symbol}.IS")

        financials = ticker.financials
        balance = ticker.balance_sheet
        cashflow = ticker.cashflow

        score = 50
        criteria = []


        # -------------------------
        # 1) KÂRLILIK
        # -------------------------

        try:
            net_income = financials.loc["Net Income"].iloc[0]
            revenue = financials.loc["Total Revenue"].iloc[0]

            margin = net_income / revenue

            if margin > 0.15:
                score += 15
                criteria.append("Güçlü net kâr marjı")
            elif margin > 0:
                score += 8
                criteria.append("Pozitif net kâr")
            else:
                score -= 15
                criteria.append("Negatif kârlılık")

        except:
            criteria.append("Kârlılık verisi alınamadı")


        # -------------------------
        # 2) BORÇLULUK
        # -------------------------

        try:
            debt = balance.loc["Total Debt"].iloc[0]
            equity = balance.loc["Stockholders Equity"].iloc[0]

            debt_ratio = debt / equity

            if debt_ratio < 0.5:
                score += 15
                criteria.append("Düşük borçluluk")
            elif debt_ratio < 1.5:
                score += 5
                criteria.append("Kontrollü borç seviyesi")
            else:
                score -= 10
                criteria.append("Yüksek borçluluk")

        except:
            criteria.append("Borç verisi alınamadı")


        # -------------------------
        # 3) GELİR BÜYÜMESİ
        # -------------------------

        try:
            revenue_now = financials.loc["Total Revenue"].iloc[0]
            revenue_old = financials.loc["Total Revenue"].iloc[-1]

            growth = (revenue_now - revenue_old) / abs(revenue_old)

            if growth > 0.20:
                score += 10
                criteria.append("Güçlü gelir büyümesi")
            elif growth > 0:
                score += 5
                criteria.append("Pozitif gelir büyümesi")
            else:
                score -= 5
                criteria.append("Gelir büyümesi zayıf")

        except:
            criteria.append("Büyüme verisi alınamadı")


        # -------------------------
        # 4) NAKİT AKIŞI
        # -------------------------

        try:
            cash = cashflow.loc["Operating Cash Flow"].iloc[0]

            if cash > 0:
                score += 10
                criteria.append("Pozitif operasyonel nakit akışı")
            else:
                score -= 10
                criteria.append("Negatif nakit akışı")

        except:
            criteria.append("Nakit akışı verisi alınamadı")


        # sınırlandırma
        score = max(0, min(score, 100))


        return {
            "score": score,
            "criteria": criteria
        }


    except Exception as e:

        return {
            "score": None,
            "criteria": [str(e)]
        }
KNOWN_STOCKS = {
    "ASELS": {"name": "Aselsan Elektronik Sanayi ve Ticaret A.Ş.", "sector": "Savunma"},
    "THYAO": {"name": "Türk Hava Yolları A.O.", "sector": "Ulaştırma"},
    "ISCTR": {"name": "Türkiye İş Bankası A.Ş. (C)", "sector": "Bankacılık"},
    "KCHOL": {"name": "Koç Holding A.Ş.", "sector": "Holding"},
    "AKBNK": {"name": "Akbank T.A.Ş.", "sector": "Bankacılık"},
    "BIMAS": {"name": "BİM Birleşik Mağazalar A.Ş.", "sector": "Perakende"},
    "EREGL": {"name": "Ereğli Demir ve Çelik Fabrikaları T.A.Ş.", "sector": "Metal"},
    "FROTO": {"name": "Ford Otomotiv Sanayi A.Ş.", "sector": "Otomotiv"},
    "TCELL": {"name": "Turkcell İletişim Hizmetleri A.Ş.", "sector": "İletişim"},
    "TUPRS": {"name": "Tüpraş Türkiye Petrol Rafinerileri A.Ş.", "sector": "Petrol"},
}

# Her sembol aramada kullanılabilir. Şirket adı/sekörü veri sağlayıcısından
# alınır; burada yalnızca V2'nin bilinen ilk sembolleri için yerel yedek vardır.
STOCKS = {symbol: {"name": symbol, "sector": "Diğer"} for symbol in BIST_TICKERS}
STOCKS.update(KNOWN_STOCKS)


st.markdown("""
<style>
  .stApp { background: radial-gradient(circle at 5% 0%, #163b62 0%, transparent 32%), radial-gradient(circle at 95% 10%, #38205e 0%, transparent 28%), #07111f; color: #e5edf7; }
  .brand { font-size: 2.5rem; font-weight: 800; letter-spacing: -.07rem; margin-bottom: .2rem; color: #f2f8ff; }
  .muted { color: #9fb0c4; }
  .card { background: linear-gradient(145deg, rgba(19,43,70,.95), rgba(12,24,42,.95)); border: 1px solid #2d5c88; border-radius: 18px; padding: 1.1rem 1.25rem; min-height: 110px; box-shadow: 0 12px 30px rgba(0,0,0,.18); }
  .eyebrow { font-size: .75rem; color: #79d8ff; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
  .disclaimer { background: #172334; border-left: 3px solid #eab308; padding: .85rem 1rem; border-radius: 8px; color: #d9e4f0; }
  .source { color: #9fb0c4; font-size: .85rem; }
  .stMetric { background: rgba(13, 31, 52, .9); border: 1px solid #28547f; border-radius: 14px; padding: .6rem; }
  .stButton > button { border-radius: 12px; border: 1px solid #3c82b7; background: linear-gradient(110deg, #0f6fae, #5c3bb1); color: white; font-weight: 700; min-height: 2.65rem; }
  .stButton > button:hover { border-color: #74d4ff; box-shadow: 0 0 18px rgba(92, 180, 255, .35); }
  .candidate { background: linear-gradient(120deg, #0d5b75, #27316d); border: 1px solid #4bd2ea; border-radius: 16px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)


def tr_number(value: Any, suffix: str = "", decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "Veri yok"
    return f"{float(value):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".") + suffix


def compact_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Veri yok"
    value = float(value)
    for threshold, symbol in ((1_000_000_000, " mlr"), (1_000_000, " mn"), (1_000, " bin")):
        if abs(value) >= threshold:
            return tr_number(value / threshold, symbol, 1)
    return tr_number(value, "", 0)


def latest_value(frame: pd.DataFrame, labels: list[str]) -> float | None:
    if frame is None or frame.empty:
        return None
    for label in labels:
        if label in frame.index:
            values = frame.loc[label].dropna()
            if not values.empty:
                return float(values.iloc[0])
    return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock(symbol: str) -> dict[str, Any]:
    ticker = yf.Ticker(f"{symbol}.IS")
    history = ticker.history(period="1y", auto_adjust=True)
    if history.empty:
        raise ValueError(f"{symbol} için fiyat verisi alınamadı.")
    try:
        info = ticker.info
    except Exception:
        info = {}
    try:
        income = ticker.financials
        balance = ticker.balance_sheet
        cashflow = ticker.cashflow
    except Exception:
        income, balance, cashflow = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    try:
        news = ticker.news or []
    except Exception:
        news = []
    return {"history": history, "info": info, "income": income, "balance": balance, "cashflow": cashflow, "news": news}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_price_histories(symbols: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Fetch many symbols in one request for the fast technical screener.

    The daily screener does not need company profiles, statements or news. Using
    yfinance's batch endpoint here avoids a costly per-symbol request sequence.
    """
    provider_symbols = [f"{symbol}.IS" for symbol in symbols]
    downloaded = yf.download(provider_symbols, period="1y", auto_adjust=True, group_by="ticker", threads=True, progress=False)
    if downloaded.empty:
        return {}
    histories: dict[str, pd.DataFrame] = {}
    if isinstance(downloaded.columns, pd.MultiIndex):
        for symbol, provider_symbol in zip(symbols, provider_symbols):
            if provider_symbol in downloaded.columns.get_level_values(0):
                history = downloaded[provider_symbol].dropna(how="all")
                if not history.empty:
                    histories[symbol] = history
    elif len(symbols) == 1:
        histories[symbols[0]] = downloaded.dropna(how="all")
    return histories


def price_chart(history: pd.DataFrame, symbol: str) -> None:
    view = history.tail(180).copy()
    close = view["Close"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.index, y=close, name="Kapanış", line={"color": "#54b7ff", "width": 2}))
    fig.add_trace(go.Scatter(x=view.index, y=close.rolling(20).mean(), name="SMA 20", line={"color": "#f6c453", "width": 1.5}))
    fig.add_trace(go.Scatter(x=view.index, y=close.rolling(50).mean(), name="SMA 50", line={"color": "#ca8cff", "width": 1.5}))
    fig.update_layout(title=f"{symbol} · Son 180 işlem günü", height=360, margin={"l": 0, "r": 0, "t": 42, "b": 0},
                      paper_bgcolor="#0d1b2d", plot_bgcolor="#0d1b2d", font={"color": "#dbeafe"}, legend={"orientation": "h"})
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#1d3653")
    st.plotly_chart(fig, use_container_width=True)


def render_disclaimer() -> None:
    st.markdown("<div class='disclaimer'><strong>Yatırım tavsiyesi değildir.</strong> Bu ekrandaki veriler bilgi ve araştırma amaçlıdır. Teknik göstergeler geçmiş fiyatlardan hesaplanır; gelecekteki performansı garanti etmez.</div>", unsafe_allow_html=True)


def render_home() -> None:
    st.markdown("<div class='brand'>BCBIST AI</div><div class='muted'>Borsa İstanbul için şeffaf, veri odaklı analiz asistanı</div>", unsafe_allow_html=True)
    render_disclaimer()
    st.write("")
    left, mid, right = st.columns(3)
    with left:
        st.markdown("<div class='card'><div class='eyebrow'>01 · Tek hisse analizi</div><h3>Veriyi tek yerde görün</h3><span class='muted'>Fiyat, teknik göstergeler, finansal özet ve mevcut haber bağlantıları.</span></div>", unsafe_allow_html=True)
    with mid:
        st.markdown("<div class='card'><div class='eyebrow'>02 · Şeffaf yorum</div><h3>Veri ile yorumu ayırın</h3><span class='muted'>Hesaplanan metrikler, yorumlardan ayrı gösterilir; eksik veri açıkça belirtilir.</span></div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'><div class='eyebrow'>03 · İzleme listeleri</div><h3>Fırsatları kuralla tarayın</h3><span class='muted'>Günlük tarama, uzun vadeli kalite görünümü ve portföy analizi aynı temel üzerinde çalışır.</span></div>", unsafe_allow_html=True)
    st.write("")
    st.subheader("Nasıl çalışır?")
    st.write("Sol menüden istediğiniz analiz ekranını açın. Kaynak, güncelleme zamanı ve ulaşılamayan alanlar sonuçta görünür.")


def render_stock_analysis() -> None:
    st.title("Tek Hisse Analizi")
    st.caption(f"{len(STOCKS)} sembollük BIST tarama evreninden hisse seçin veya kutuya kod yazarak arayın.")
    col_input, col_action = st.columns([4, 1])
    with col_input:
        current_symbol = st.session_state.get("symbol", "ASELS")
        default_index = sorted(STOCKS).index(current_symbol) if current_symbol in STOCKS else 0
        raw_symbol = st.selectbox(
            "Hisse kodu",
            options=sorted(STOCKS),
            index=default_index,
            format_func=lambda symbol: f"{symbol} — {STOCKS[symbol]['name']}",
        )
    with col_action:
        st.write("")
        run = st.button("Analiz oluştur", type="primary", use_container_width=True)
    if run:
        st.session_state.symbol = raw_symbol
        st.session_state.analysis_requested = True
    if not st.session_state.get("analysis_requested"):
        st.info("Analiz için bir hisse kodu girip “Analiz oluştur” düğmesine basın.")
        return
    symbol = st.session_state.get("symbol", raw_symbol)
    if not symbol:
        st.warning("Lütfen geçerli bir BIST kodu girin.")
        return
    try:
        with st.spinner(f"{symbol} verisi doğrulanıyor ve göstergeler hesaplanıyor..."):
            data = fetch_stock(symbol)
    except Exception as exc:
        st.error(f"Veri alınamadı: {exc}")
        st.caption("Piyasa kapalı olabilir, sembol hatalı olabilir veya veri sağlayıcısı geçici olarak erişilemez olabilir.")
        return

    history, info = data["history"], data["info"]
    summary = calculate_technicals(history)
    latest = float(history["Close"].iloc[-1])
    previous = float(history["Close"].iloc[-2]) if len(history) > 1 else latest
    change = ((latest / previous) - 1) * 100 if previous else 0
    configured = STOCKS.get(symbol, {})
    company_name = info.get("longName") or configured.get("name") or symbol
    sector = info.get("sector") or configured.get("sector") or "Veri sağlayıcısında yok"

    st.subheader(company_name)
    st.caption(f"Sembol: {symbol}.IS · Sektör: {sector} · Son fiyat çubuğu: {history.index[-1].strftime('%d.%m.%Y')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Son kapanış", tr_number(latest, " TL"), f"{change:+.2f}%")
    c2.metric("Piyasa değeri", compact_number(info.get("marketCap")))
    c3.metric("İşlem hacmi", compact_number(history["Volume"].iloc[-1]))
    c4.metric("Trend", summary.trend)
    st.caption("Kaynak: Yahoo Finance. Fiyatlar gecikmeli olabilir; işlem öncesinde resmi/veri sağlayıcısı ekranından kontrol edin.")

    price_chart(history, symbol)
    technical_tab, financial_tab, news_tab, conclusion_tab = st.tabs(["Teknik Analiz", "Finansal Analiz", "Haberler", "Genel Değerlendirme"])
    with technical_tab:
        cols = st.columns(4)
        cols[0].metric("RSI (14)", tr_number(summary.rsi, "", 1))
        cols[1].metric("MACD", tr_number(summary.macd, "", 2))
        cols[2].metric("Momentum (20g)", tr_number(summary.momentum_20d, "%", 1))
        cols[3].metric("Hacim / Ort. (20g)", tr_number(summary.volume_ratio, "x", 2))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("SMA 20", tr_number(summary.sma_20, " TL"))
        c2.metric("SMA 50", tr_number(summary.sma_50, " TL"))
        c3.metric("Destek (20g)", tr_number(summary.support, " TL"))
        c4.metric("Direnç (20g)", tr_number(summary.resistance, " TL"))
        st.caption(f"ATR (14): {tr_number(summary.atr_14, ' TL')} · ATR, son dönemdeki ortalama günlük fiyat hareket aralığını gösterir.")
        st.subheader("Gösterge açıklamaları")
        for explanation in summary.explanations:
            st.write("• " + explanation)
        st.caption("Destek ve direnç, son en fazla 20 işlem gününün düşük/yüksek değerleridir; kesin fiyat seviyesi değildir.")
    with financial_tab:
        income, balance, cashflow = data["income"], data["balance"], data["cashflow"]
        revenue = latest_value(income, ["Total Revenue", "Operating Revenue"])
        net_income = latest_value(income, ["Net Income", "Net Income Common Stockholders"])
        ebitda = latest_value(income, ["EBITDA", "Normalized EBITDA"])
        total_debt = latest_value(balance, ["Total Debt", "Long Term Debt And Capital Lease Obligation"])
        equity = latest_value(balance, ["Stockholders Equity", "Total Stockholder Equity"])
        operating_cashflow = latest_value(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        f1, f2, f3 = st.columns(3)
        f1.metric("Gelir", compact_number(revenue))
        f2.metric("Net kâr", compact_number(net_income))
        f3.metric("FAVÖK", compact_number(ebitda))
        f4, f5, f6 = st.columns(3)
        f4.metric("Toplam borç", compact_number(total_debt))
        f5.metric("Öz kaynak", compact_number(equity))
        f6.metric("Faaliyet nakit akışı", compact_number(operating_cashflow))
        observations = []
        if revenue is not None and net_income is not None:
            observations.append(f"Hesaplanan net kâr marjı: %{(net_income / revenue) * 100:.1f}.")
        if total_debt is not None and equity not in (None, 0):
            observations.append(f"Borç / öz kaynak oranı: %{(total_debt / equity) * 100:.1f}.")
        if operating_cashflow is not None:
            observations.append("Faaliyet nakit akışı, şirketin ana faaliyetlerinden ürettiği nakdi gösterir.")
        if observations:
            st.subheader("Hesaplanmış finansal notlar")
            for observation in observations:
                st.write("• " + observation)
        else:
            st.info("Veri sağlayıcısı bu sembol için yeterli finansal tablo döndürmedi. Finansal yorum üretilmedi.")
        st.caption("Finansal dönemler şirket bazında farklılaşabilir. Karşılaştırma öncesinde dönem ve para birimini resmi finansal rapordan doğrulayın.")
    with news_tab:
        st.subheader("Sağlayıcının sunduğu güncel bağlantılar")
        news_items = data["news"]
        if not news_items:
            st.info("Bu sembol için veri sağlayıcısı haber bağlantısı döndürmedi. KAP bildirimleri ayrıca resmi KAP kaynağından doğrulanmalıdır.")
        for item in news_items[:8]:
            content = item.get("content", item)
            title = content.get("title") or item.get("title") or "Başlıksız içerik"
            provider = content.get("provider", {}).get("displayName") or content.get("publisher") or "Kaynak belirtilmemiş"
            url = content.get("canonicalUrl", {}).get("url") or content.get("clickThroughUrl", {}).get("url") or item.get("link")
            if url:
                st.markdown(f"- [{title}]({url})  \\n+  <span class='source'>Kaynak: {provider}</span>", unsafe_allow_html=True)
            else:
                st.write(f"• {title} — {provider}")
        st.caption("Haber metinleri model tarafından yorumlanmaz; bağlantılar kaynak kontrolü için gösterilir.")
    with conclusion_tab:
        score, reasons = score_opportunity(summary)
        st.subheader("Kural tabanlı teknik görünüm")
        st.write(f"İzleme puanı: **{score}/100**")
        if reasons:
            for reason in reasons:
                st.write("• " + reason)
        else:
            st.info("Tanımlı izleme kuralları şu anda belirgin bir teknik kesişim göstermiyor.")
        st.write("Bu puan, yalnızca ekrandaki teknik kuralların sayısal özeti olup alım, satım ya da hedef fiyat önerisi değildir.")
    render_disclaimer()
    st.caption(f"Son uygulama yenilemesi: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")


def risk_from_history(history: pd.DataFrame) -> str:
    """Classify historical volatility; it is not a forecast of risk."""
    returns = history["Close"].pct_change().dropna().tail(60)
    if returns.empty:
        return "Veri yetersiz"
    annualized_volatility = float(returns.std() * (252 ** 0.5) * 100)
    if annualized_volatility < 25:
        return "Düşük"
    if annualized_volatility < 45:
        return "Orta"
    return "Yüksek"


def render_daily_opportunities() -> None:
    st.title("Günlük Fırsatlar")
    st.write("Bu sayfa bir öneri listesi değildir. Seçili sembollerde, açıkça tanımlanmış teknik kuralların güncel durumunu gösterir.")
    scope = st.selectbox(
    "Tarama Evreni",
    [
        "BIST 30",
        "BIST 100",
        "Tüm BIST"
    ],
    index=1
)

    from src.data.bist_universe import (
                BIST30_TICKERS,
                BIST100_TICKERS,
                BIST_TICKERS,
    )

    if scope == "BIST 30":
        symbols = list(BIST30_TICKERS)
    elif scope == "BIST 100":
        symbols = list(BIST100_TICKERS)
    else:
        symbols = list(BIST_TICKERS)

        st.caption(f"Toplam {len(symbols)} hisse taranacak.")

        if not st.button("🔍 Günlük Taramayı Başlat", type="primary", use_container_width=True):
            st.info("Tarama başlatıldığında her sembolün fiyat verisi alınır ve aynı teknik kurallar uygulanır.")
            render_disclaimer()
            return
    rows: list[dict[str, Any]] = []
    if scope == "BIST 30":
        symbols = list(BIST30_TICKERS)
    elif scope == "BIST 100":
        symbols = list(BIST100_TICKERS)
    else:
        symbols = list(BIST_TICKERS)
    with st.spinner(f"{len(symbols)} sembolün fiyat verisi tek seferde alınıyor..."):
        histories = fetch_price_histories(tuple(symbols))
    progress = st.progress(0, text="Teknik göstergeler hesaplanıyor...")
    for index, symbol in enumerate(symbols, start=1):
        try:
            history = histories.get(symbol)
            if history is None or history.empty:
                raise ValueError("Fiyat verisi alınamadı")
            summary = calculate_technicals(history)
            score, reasons = score_opportunity(summary)
            price = float(history["Close"].iloc[-1])
            rows.append({
                "Sembol": symbol,
                "Şirket": STOCKS[symbol]["name"],
                "Puan": score,
                "Risk": risk_from_history(history),
                "Son Fiyat (TL)": price,
                "Trend": summary.trend,
                "Gerekçeler": reasons,
                "Destek": summary.support,
                "Direnç": summary.resistance,
            })
        except Exception as exc:
            rows.append({"Sembol": symbol, "Şirket": STOCKS[symbol]["name"], "Puan": None, "Risk": "Veri alınamadı", "Son Fiyat (TL)": None, "Trend": "Veri alınamadı", "Gerekçeler": [str(exc)], "Destek": None, "Direnç": None})
        progress.progress(index / len(symbols), text=f"{symbol} taranıyor ({index}/{len(symbols)})")
    progress.empty()

    valid_rows = [row for row in rows if row["Puan"] is not None]
    valid_rows.sort(key=lambda row: row["Puan"], reverse=True)
    if not valid_rows:
        st.error("Tarama için veri alınamadı. Lütfen daha sonra yeniden deneyin.")
        return
    table = pd.DataFrame(valid_rows)[["Sembol", "Puan", "Risk", "Son Fiyat (TL)", "Trend"]]
    top_candidates = valid_rows[:3]
    if top_candidates:
        st.subheader("Bugünün belirgin izleme adayları")
        candidate_columns = st.columns(len(top_candidates))
        for column, candidate in zip(candidate_columns, top_candidates):
            with column:
                st.markdown(f"<div class='candidate'><div class='eyebrow'>İZLEME ADAYI</div><h3>{candidate['Sembol']}</h3><b>{candidate['Puan']}/100</b> teknik puan<br><span class='muted'>Risk: {candidate['Risk']}</span></div>", unsafe_allow_html=True)
    top_candidates = valid_rows[:5]

    if top_candidates:
        st.subheader("🏆 En Güçlü Finansal Şirketler")

        cols = st.columns(len(top_candidates))

        for col, company in zip(cols, top_candidates):
            with col:
                st.metric(
                    company["Sembol"],
                    f"{company['Puan']}/100"
            )
    st.dataframe(table, use_container_width=True, hide_index=True, column_config={"Son Fiyat (TL)": st.column_config.NumberColumn(format="%.2f TL")})
    st.subheader("Adayların hesaplama gerekçesi")
    for row in valid_rows:
        with st.expander(f"{row['Sembol']} · {row['Puan']}/100 · {row['Risk']} volatilite"):
            st.write(row["Şirket"])
            if row["Gerekçeler"]:
                for reason in row["Gerekçeler"]:
                    st.write("• " + reason)
            else:
                st.write("Tanımlı kurallarda pozitif kesişim tespit edilmedi.")
            st.write(f"İzleme seviyeleri — destek: {tr_number(row['Destek'], ' TL')}, direnç: {tr_number(row['Direnç'], ' TL')}")
    st.caption("Puan; trend, RSI, MACD ve hacim kurallarının toplamıdır. Şirketin finansal kalitesini veya haber akışını henüz içermez.")
    render_disclaimer()


def calculate_quality(data: dict[str, Any]) -> tuple[int, list[str]]:
    """Produce a transparent, limited financial-quality score from available statements."""
    income, balance, cashflow = data["income"], data["balance"], data["cashflow"]
    revenue = latest_value(income, ["Total Revenue", "Operating Revenue"])
    net_income = latest_value(income, ["Net Income", "Net Income Common Stockholders"])
    debt = latest_value(balance, ["Total Debt", "Long Term Debt And Capital Lease Obligation"])
    equity = latest_value(balance, ["Stockholders Equity", "Total Stockholder Equity"])
    operating_cashflow = latest_value(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    score, reasons = 0, []
    if revenue not in (None, 0) and net_income is not None and net_income > 0:
        score += 30
        reasons.append("Son erişilebilir dönemde net kâr pozitif.")
    if revenue not in (None, 0) and net_income is not None and (net_income / revenue) >= 0.10:
        score += 20
        reasons.append("Hesaplanan net kâr marjı %10 veya üzeri.")
    if operating_cashflow is not None and operating_cashflow > 0:
        score += 25
        reasons.append("Faaliyetlerden nakit akışı pozitif.")
    if debt is not None and equity not in (None, 0) and debt / equity < 1:
        score += 25
        reasons.append("Borç / öz kaynak oranı 1'in altında.")
    return score, reasons
POSITIVE = [
    "yatırım",
    "anlaşma",
    "ihale",
    "kar",
    "kâr",
    "büyüme",
    "ihracat",
    "kapasite",
    "satın alma",
    "temettü",
    "rekor",
]

NEGATIVE = [
    "zarar",
    "ceza",
    "dava",
    "iflas",
    "bedelli",
    "borç",
    "düşüş",
    "iptal",
    "soruşturma",
    "kayıp",
]


def calculate_news_score(symbol):

    news = get_news(symbol)

    score = 0

    reasons = []

    for item in news:

        title = item["title"].lower()

        for word in POSITIVE:

            if word in title:

                score += 2

                reasons.append("🟢 " + item["title"])

                break

        for word in NEGATIVE:

            if word in title:

                score -= 2

                reasons.append("🔴 " + item["title"])

                break

    score = max(-10, min(10, score))

    return score, reasons

def render_long_term() -> None:
    st.title("🏛️ Uzun Vadeli")

    st.write(
        "Bu ekran, şirketlerin finansal tablolarını temel alarak uzun vadeli kalite puanı oluşturur."
    )

    scope = st.selectbox(
        "Tarama Evreni",
        [
            "BIST 30",
            "BIST 100",
            "Tüm BIST"
        ],
        index=1,
        key="long_term_scope"
    )

    from src.data.bist_universe import (
        BIST30_TICKERS,
        BIST100_TICKERS,
        BIST_TICKERS,
    )

    if scope == "BIST 30":
        symbols = list(BIST30_TICKERS)
    elif scope == "BIST 100":
        symbols = list(BIST100_TICKERS)
    else:
        symbols = list(BIST_TICKERS)

    st.caption(f"Toplam {len(symbols)} şirket analiz edilecek.")

    if not st.button(
        "🏛️ Finansal Taramayı Başlat",
        type="primary",
        use_container_width=True,
    ):
        st.info(
            "Tarama; şirketlerin finansal tabloları, borçluluk, kârlılık ve nakit akışı kriterlerini değerlendirir."
        )
        render_disclaimer()
        return

    rows = []

    with st.spinner("Finansal veriler taranıyor..."):
        for symbol in symbols:
            try:
                result = analyze_financial_quality(symbol)

                rows.append({
                    "Sembol": symbol,
                    "Sektör": STOCKS.get(symbol, {}).get("sector", "Bilinmiyor"),
                    "Kalite puanı":  result["score"],
                    "Kriterler": result["criteria"],
                })

            except Exception as exc:
                rows.append({
                    "Sembol": symbol,
                    "Sektör": STOCKS.get(symbol, {}).get("sector", "Bilinmiyor"),
                    "Kalite puanı": None,
                    "Kriterler": [str(exc)],
                })

    valid_rows = [
        row for row in rows
        if row["Kalite puanı"] is not None
    ]

    valid_rows.sort(
        key=lambda x: x["Kalite puanı"],
        reverse=True
    )

    if not valid_rows:
        st.warning("Finansal veri bulunamadı.")
        render_disclaimer()
        return


    st.dataframe(
        pd.DataFrame(valid_rows)[
            ["Sembol", "Sektör", "Kalite puanı"]
        ],
        use_container_width=True,
        hide_index=True
    )


    for row in valid_rows:
        with st.expander(
            f"{row['Sembol']} · finansal kalite puanı: {row['Kalite puanı']}/100"
        ):

            if row["Kriterler"]:
                for criterion in row["Kriterler"]:
                    st.write("• " + criterion)
            else:
                st.write(
                "Veri mevcut olsa da tanımlı kalite kuralları karşılanmadı veya tablo kalemleri eşleşmedi."
                )

          
st.caption("Puan, yalnızca erişilebilir son dönem tablolarına dayanır; büyüme, yönetim kalitesi, değerleme ve gelecek beklentileri içermez.")
render_disclaimer()


def render_portfolio_assistant() -> None:
    st.title("Portföy Asistanı")
    st.write("Portföyünüze ait adet ve maliyet bilgisini girin. Hesaplamalar yalnızca bu tarayıcı oturumunda işlenir; kalıcı olarak saklanmaz.")
    default_portfolio = pd.DataFrame([
        {"Sembol": "ASELS", "Adet": 0.0, "Ortalama Maliyet (TL)": 0.0},
        {"Sembol": "THYAO", "Adet": 0.0, "Ortalama Maliyet (TL)": 0.0},
    ])
    portfolio = st.data_editor(default_portfolio, num_rows="dynamic", use_container_width=True, hide_index=True,
                               column_config={"Sembol": st.column_config.TextColumn("Sembol", required=True), "Adet": st.column_config.NumberColumn(min_value=0.0), "Ortalama Maliyet (TL)": st.column_config.NumberColumn(min_value=0.0)})
    if not st.button("Portföyü analiz et", type="primary"):
        render_disclaimer()
        return
    positions = portfolio.copy()
    positions["Sembol"] = positions["Sembol"].astype(str).str.upper().str.strip().str.replace(".IS", "", regex=False)
    positions = positions[(positions["Sembol"] != "") & (positions["Adet"] > 0)]
    if positions.empty:
        st.warning("Analiz için en az bir sembol ve sıfırdan büyük adet girin.")
        return
    calculated = []
    with st.spinner("Güncel fiyatlar alınıyor..."):
        for _, position in positions.iterrows():
            try:
                data = fetch_stock(position["Sembol"])
                current_price = float(data["history"]["Close"].iloc[-1])
                value = current_price * float(position["Adet"])
                cost = float(position["Ortalama Maliyet (TL)"]) * float(position["Adet"])
                calculated.append({"Sembol": position["Sembol"], "Sektör": STOCKS.get(position["Sembol"], {}).get("sector", "Bilinmiyor"), "Güncel Değer": value, "Maliyet": cost, "Kâr/Zarar": value - cost})
            except Exception as exc:
                st.warning(f"{position['Sembol']} için güncel fiyat alınamadı: {exc}")
    if not calculated:
        st.error("Geçerli pozisyon için fiyat alınamadı.")
        return
    result = pd.DataFrame(calculated)
    total_value = float(result["Güncel Değer"].sum())
    result["Ağırlık (%)"] = result["Güncel Değer"] / total_value * 100
    total_cost = float(result["Maliyet"].sum())
    a, b, c = st.columns(3)
    a.metric("Güncel portföy değeri", tr_number(total_value, " TL"))
    b.metric("Toplam maliyet", tr_number(total_cost, " TL"))
    c.metric("Hesaplanan kâr/zarar", tr_number(total_value - total_cost, " TL"))
    st.subheader("Pozisyon dağılımı")
    st.dataframe(result[["Sembol", "Sektör", "Güncel Değer", "Ağırlık (%)", "Kâr/Zarar"]], use_container_width=True, hide_index=True,
                 column_config={"Güncel Değer": st.column_config.NumberColumn(format="%.2f TL"), "Ağırlık (%)": st.column_config.NumberColumn(format="%.1f%%"), "Kâr/Zarar": st.column_config.NumberColumn(format="%.2f TL")})
    sector_weights = result.groupby("Sektör", as_index=False)["Güncel Değer"].sum()
    sector_weights["Ağırlık (%)"] = sector_weights["Güncel Değer"] / total_value * 100
    st.subheader("Sektör dağılımı")
    st.bar_chart(sector_weights.set_index("Sektör")["Ağırlık (%)"])
    largest_position = result.loc[result["Ağırlık (%)"].idxmax()]
    if largest_position["Ağırlık (%)"] >= 35:
        st.warning(f"Yoğunlaşma notu: {largest_position['Sembol']} portföyün %{largest_position['Ağırlık (%)']:.1f}'ini oluşturuyor.")
    if len(sector_weights) == 1:
        st.warning("Sektör çeşitliliği notu: portföy yalnızca tek sektörde yoğunlaşmış görünüyor.")
    st.caption("Bu ekran pozisyon büyüklüklerini açıklar; varlık alım/satımı veya yeniden dengeleme önerisi vermez.")
    render_disclaimer()


if "page" not in st.session_state:
    st.session_state.page = "Ana Sayfa"

nav_items = [
    ("🏠 Ana Sayfa", "Ana Sayfa"),
    ("📊 Tek Hisse", "Tek Hisse Analizi"),
    ("⚡ Günlük Fırsatlar", "Günlük Fırsatlar"),
    ("🏛️ Uzun Vadeli", "Uzun Vadeli"),
    ("💼 Portföy", "Portföy Asistanı"),
]
nav_columns = st.columns(len(nav_items))
for column, (label, target) in zip(nav_columns, nav_items):
    with column:
        button_type = "primary" if st.session_state.page == target else "secondary"
        if st.button(label, key=f"nav_{target}", use_container_width=True, type=button_type):
            st.session_state.page = target
            st.rerun()
st.caption(f"BCBIST V2 · {len(STOCKS)} sembol · Veri odaklı analiz · Yatırım tavsiyesi değildir")
st.divider()
page = st.session_state.page

if page == "Ana Sayfa":
    render_home()
elif page == "Tek Hisse Analizi":
    render_stock_analysis()
elif page == "Günlük Fırsatlar":
    render_daily_opportunities()
elif page == "Uzun Vadeli":
    render_long_term()
else:
    render_portfolio_assistant()
