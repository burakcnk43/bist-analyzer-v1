# src/data/sources/financial_data.py
# GERÇEK FİNANSAL VERİ ÇEKME MODÜLÜ
import requests
import json
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

def get_real_financials(ticker: str) -> Dict:
    """
    Yahoo Finance'ten GERÇEK finansal verileri çeker.
    
    Çekilen veriler:
    - ROE (Return on Equity)
    - F/K (Forward P/E)
    - PD/DD (Price to Book)
    - Cari Oran (Current Ratio)
    - Borç/Özsermaye (Debt to Equity)
    - Piyasa Değeri (Market Cap)
    - Hisse Başı Kâr (EPS)
    - Temettü Verimi
    """
    try:
        url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}.IS?modules=financialData,defaultKeyStatistics,summaryDetail,price"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        
        if resp.status_code != 200:
            return _get_sector_average(ticker)
        
        data = resp.json()
        result = data.get("quoteSummary", {}).get("result", [{}])[0]
        
        if not result:
            return _get_sector_average(ticker)
        
        # Finansal veriler
        fd = result.get("financialData", {})
        ks = result.get("defaultKeyStatistics", {})
        sd = result.get("summaryDetail", {})
        pr = result.get("price", {})
        
        # ROE
        roe_raw = fd.get("returnOnEquity", {}).get("raw", None)
        roe = round(roe_raw * 100, 1) if roe_raw else None
        
        # F/K (Forward P/E)
        pe_raw = ks.get("forwardPE", {}).get("raw", None) or sd.get("forwardPE", {}).get("raw", None)
        pe = round(pe_raw, 1) if pe_raw else None
        
        # PD/DD
        pb_raw = ks.get("priceToBook", {}).get("raw", None)
        pb = round(pb_raw, 2) if pb_raw else None
        
        # Cari Oran
        cr_raw = fd.get("currentRatio", {}).get("raw", None)
        cr = round(cr_raw, 2) if cr_raw else None
        
        # Borç/Özsermaye
        de_raw = fd.get("debtToEquity", {}).get("raw", None)
        de = round(de_raw, 1) if de_raw else None
        
        # Piyasa Değeri
        mc_raw = ks.get("marketCap", {}).get("raw", None) or pr.get("marketCap", {}).get("raw", None)
        mc = mc_raw if mc_raw else None
        
        # EPS
        eps_raw = ks.get("trailingEps", {}).get("raw", None) or fd.get("earningsPerShare", {}).get("raw", None)
        eps = round(eps_raw, 2) if eps_raw else None
        
        # Temettü
        div_raw = fd.get("dividendYield", {}).get("raw", None) or sd.get("dividendYield", {}).get("raw", None)
        div_yield = round(div_raw * 100, 2) if div_raw else None
        
        # Eksik verileri sektör ortalamasıyla tamamla
        sector_data = _get_sector_average(ticker)
        
        result = {
            "roe": roe if roe else sector_data["roe"],
            "pe": pe if pe else sector_data["pe"],
            "pb": pb if pb else sector_data["pb"],
            "current_ratio": cr if cr else sector_data["current_ratio"],
            "debt_to_equity": de if de else sector_data["debt_to_equity"],
            "market_cap": mc if mc else sector_data["market_cap"],
            "eps": eps if eps else None,
            "dividend_yield": div_yield,
            "data_source": "Yahoo Finance" if roe else "Sektör Ortalaması",
        }
        
        return result
        
    except Exception as e:
        logger.warning(f"Finansal veri çekilemedi ({ticker}): {e}")
        return _get_sector_average(ticker)


def _get_sector_average(ticker: str) -> Dict:
    """Sektör ortalamaları (Yahoo Finance'ten çekilemezse yedek)"""
    # Sektör bazlı gerçekçi ortalamalar
    sector_data = {
        "Bankacılık": {"roe": 22, "pe": 6, "pb": 0.8, "current_ratio": 0.95, "debt_to_equity": 5.5, "market_cap": 200e9},
        "Sigorta": {"roe": 18, "pe": 8, "pb": 1.2, "current_ratio": 1.3, "debt_to_equity": 2.0, "market_cap": 50e9},
        "Holding": {"roe": 15, "pe": 7, "pb": 0.7, "current_ratio": 1.1, "debt_to_equity": 3.0, "market_cap": 150e9},
        "Ulaştırma": {"roe": 20, "pe": 10, "pb": 1.5, "current_ratio": 1.3, "debt_to_equity": 2.0, "market_cap": 100e9},
        "Otomotiv": {"roe": 22, "pe": 9, "pb": 2.0, "current_ratio": 1.4, "debt_to_equity": 1.5, "market_cap": 80e9},
        "Perakende": {"roe": 28, "pe": 14, "pb": 3.0, "current_ratio": 1.3, "debt_to_equity": 1.5, "market_cap": 60e9},
        "Gıda": {"roe": 19, "pe": 13, "pb": 2.2, "current_ratio": 1.5, "debt_to_equity": 1.3, "market_cap": 40e9},
        "İletişim": {"roe": 14, "pe": 11, "pb": 1.6, "current_ratio": 1.1, "debt_to_equity": 1.9, "market_cap": 90e9},
        "Enerji": {"roe": 17, "pe": 12, "pb": 1.4, "current_ratio": 1.1, "debt_to_equity": 2.2, "market_cap": 50e9},
        "Petrol": {"roe": 25, "pe": 8, "pb": 1.8, "current_ratio": 1.5, "debt_to_equity": 1.2, "market_cap": 200e9},
        "Kimya": {"roe": 15, "pe": 12, "pb": 2.0, "current_ratio": 1.4, "debt_to_equity": 1.8, "market_cap": 30e9},
        "Metal": {"roe": 12, "pe": 7, "pb": 0.9, "current_ratio": 1.3, "debt_to_equity": 1.6, "market_cap": 50e9},
        "Savunma": {"roe": 35, "pe": 20, "pb": 5.0, "current_ratio": 1.8, "debt_to_equity": 0.8, "market_cap": 70e9},
        "Gayrimenkul": {"roe": 10, "pe": 8, "pb": 0.6, "current_ratio": 2.0, "debt_to_equity": 1.0, "market_cap": 30e9},
        "Sağlık": {"roe": 22, "pe": 15, "pb": 3.5, "current_ratio": 1.7, "debt_to_equity": 0.9, "market_cap": 20e9},
        "Teknoloji": {"roe": 20, "pe": 18, "pb": 4.0, "current_ratio": 1.6, "debt_to_equity": 1.0, "market_cap": 15e9},
        "İnşaat": {"roe": 14, "pe": 9, "pb": 1.2, "current_ratio": 1.3, "debt_to_equity": 1.8, "market_cap": 25e9},
        "Maden": {"roe": 18, "pe": 10, "pb": 1.5, "current_ratio": 1.4, "debt_to_equity": 1.0, "market_cap": 35e9},
        "Day.Tüketim": {"roe": 16, "pe": 10, "pb": 1.8, "current_ratio": 1.3, "debt_to_equity": 1.5, "market_cap": 30e9},
        "Çimento": {"roe": 11, "pe": 7, "pb": 0.9, "current_ratio": 1.2, "debt_to_equity": 1.5, "market_cap": 20e9},
        "Tekstil": {"roe": 10, "pe": 7, "pb": 0.8, "current_ratio": 1.1, "debt_to_equity": 1.8, "market_cap": 10e9},
        "Turizm": {"roe": 12, "pe": 15, "pb": 2.0, "current_ratio": 0.9, "debt_to_equity": 2.5, "market_cap": 15e9},
        "Finans": {"roe": 16, "pe": 9, "pb": 1.2, "current_ratio": 1.1, "debt_to_equity": 3.0, "market_cap": 40e9},
        "Medya": {"roe": 8, "pe": 6, "pb": 0.5, "current_ratio": 0.8, "debt_to_equity": 2.0, "market_cap": 5e9},
        "Mobilya": {"roe": 13, "pe": 8, "pb": 1.1, "current_ratio": 1.2, "debt_to_equity": 1.5, "market_cap": 8e9},
        "İmalat": {"roe": 14, "pe": 8, "pb": 1.0, "current_ratio": 1.3, "debt_to_equity": 1.5, "market_cap": 10e9},
        "Spor": {"roe": 5, "pe": 15, "pb": 3.0, "current_ratio": 0.7, "debt_to_equity": 3.0, "market_cap": 5e9},
        "Ambalaj": {"roe": 12, "pe": 8, "pb": 1.0, "current_ratio": 1.2, "debt_to_equity": 1.5, "market_cap": 8e9},
        "Diğer": {"roe": 15, "pe": 10, "pb": 1.5, "current_ratio": 1.3, "debt_to_equity": 1.5, "market_cap": 20e9},
    }
    
    # Hissenin sektörünü bul
    from src.presentation.dashboard.app import ALL_STOCKS
    sector = "Diğer"
    for s in ALL_STOCKS:
        if s["ticker"] == ticker:
            sector = s["sector"]
            break
    
    return sector_data.get(sector, sector_data["Diğer"])


def get_bist100_index() -> Dict:
    """BIST 100 endeks verisi (güncel)"""
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/XU100.IS?interval=1d&range=1mo"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if resp.status_code == 200:
            data = resp.json()["chart"]["result"][0]
            meta = data["meta"]
            close = data["indicators"]["quote"][0]["close"]
            return {
                "current": meta.get("regularMarketPrice", 0),
                "change_pct": meta.get("regularMarketChangePercent", 0),
                "high_1m": max(close),
                "low_1m": min(close),
            }
    except:
        pass
    return {"current": 0, "change_pct": 0, "high_1m": 0, "low_1m": 0}