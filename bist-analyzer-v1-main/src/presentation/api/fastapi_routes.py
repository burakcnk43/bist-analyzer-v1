# src/presentation/api/fastapi_routes.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)

# ========== Modeller ==========

class TickerRequest(BaseModel):
    ticker: str = Field(..., min_length=3, max_length=6)

class AnalysisResponse(BaseModel):
    ticker: str
    total_score: float
    stars: Dict[str, int]
    short_term_rating: int
    long_term_rating: int
    strengths: List[str]
    risks: List[str]
    commentary: str
    current_price: float
    roe: float
    pe_ratio: float
    ncv_per_share: float
    margin_of_safety_pct: float
    rsi: float
    trend: str
    timestamp: str

class BIST100Response(BaseModel):
    timestamp: str
    stock_count: int
    stocks: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    stocks_tracked: int

# ========== FastAPI App ==========

app = FastAPI(
    title="BIST AI ANALYZER PRO API",
    description="Çok Faktörlü BIST 100 Hisse Analiz ve Karar Destek Sistemi",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global servisler (run.py'da başlatılır)
pipeline = None
kap_client = None
price_feed = None
confidence_engine = None
ratio_analyzer = None


@app.on_event("startup")
async def startup():
    """Başlangıçta servisleri başlat"""
    global pipeline, kap_client, price_feed, confidence_engine, ratio_analyzer
    
    from src.data.sources.kap_client import KAPClient
    from src.data.sources.realtime_price_feed import RealTimePriceFeed
    from src.data.cache.redis_manager import RedisManager
    from src.data.repositories.technical_repo import TechnicalRepository
    from src.data.pipeline.fast_data_pipeline import FastDataPipeline
    from src.domain.services.confidence_engine import ConfidenceEngine
    from src.domain.services.ratio_analyzer import RatioAnalyzer
    
    kap_client = KAPClient()
    await kap_client.__aenter__()
    
    price_feed = RealTimePriceFeed()
    
    redis = RedisManager()
    technical = TechnicalRepository()
    
    pipeline = FastDataPipeline(kap_client, redis, price_feed, technical)
    await pipeline.start()
    
    confidence_engine = ConfidenceEngine()
    ratio_analyzer = RatioAnalyzer()
    
    # BIST 100 hisselerini izlemeye başla
    bist100 = await kap_client.get_bist100_list()
    tickers = [s["ticker"] for s in bist100[:30]]
    await price_feed.start(tickers)
    
    logger.info("API servisleri başlatıldı")


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Sistem sağlık kontrolü"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "stocks_tracked": len(price_feed._prices) if price_feed else 0
    }


@app.get("/api/bist100", response_model=BIST100Response)
async def get_bist100():
    """BIST 100 hisse listesi"""
    if not kap_client:
        raise HTTPException(status_code=503, detail="Servisler başlatılmadı")
    
    stocks = await kap_client.get_bist100_list()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "stock_count": len(stocks),
        "stocks": stocks
    }


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_stock(request: TickerRequest):
    """Hisse için tam kapsamlı analiz"""
    if not pipeline or not confidence_engine:
        raise HTTPException(status_code=503, detail="Servisler başlatılmadı")
    
    try:
        # Veriyi çek
        data = await pipeline.get_stock_data(request.ticker)
        
        if not data or data.get("current_price", 0) <= 0:
            raise HTTPException(status_code=404, detail=f"{request.ticker} için veri bulunamadı")
        
        current_price = data.get("current_price", 0)
        technicals = data.get("technicals", {})
        
        # BIST 100 listesinden sektör bul
        bist100 = await kap_client.get_bist100_list()
        sector = "Genel"
        for s in bist100:
            if s["ticker"] == request.ticker:
                sector = s.get("sector", "Genel")
                break
        
        # Örnek finansal veriler (gerçek KAP'tan çekilecek)
        # Şimdilik simüle edilmiş değerler
        donen_varliklar = 500_000_000_000
        kvyk = 200_000_000_000
        uvyk = 150_000_000_000
        oz_sermaye = 450_000_000_000
        net_kar = 128_000_000_000
        hisse_sayisi = 1_000_000_000
        nakit_akisi = 75_000_000_000
        
        # Analiz çalıştır
        score = confidence_engine.analyze(
            ticker=request.ticker,
            current_price=current_price,
            donen_varliklar=donen_varliklar,
            kisa_vadeli_yukumlulukler=kvyk,
            uzun_vadeli_yukumlulukler=uvyk,
            oz_sermaye=oz_sermaye,
            net_kar=net_kar,
            hisse_sayisi=hisse_sayisi,
            isletme_nakit_akisi=nakit_akisi,
            sector=sector,
            rsi=technicals.get("rsi_14", 50),
            trend=technicals.get("trend", "neutral")
        )
        
        roe = ratio_analyzer.calculate_roe(net_kar, oz_sermaye)
        pe = ratio_analyzer.calculate_pe_ratio(current_price, net_kar, hisse_sayisi)
        
        from src.domain.services.graham_valuation import GrahamValuationService
        graham = GrahamValuationService()
        ncv_data = graham.margin_of_safety_analysis(
            request.ticker, current_price, donen_varliklar,
            kvyk, uvyk, hisse_sayisi
        )
        
        return AnalysisResponse(
            ticker=score.ticker,
            total_score=score.total_score,
            stars=score.stars,
            short_term_rating=score.short_term_rating,
            long_term_rating=score.long_term_rating,
            strengths=score.strengths,
            risks=score.risks,
            commentary=score.commentary,
            current_price=current_price,
            roe=roe,
            pe_ratio=pe if pe != float('inf') else 0,
            ncv_per_share=ncv_data["ncv_per_share"],
            margin_of_safety_pct=ncv_data["margin_of_safety_pct"],
            rsi=technicals.get("rsi_14", 50),
            trend=technicals.get("trend", "neutral"),
            timestamp=score.timestamp
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analiz hatası ({request.ticker}): {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prices")
async def get_all_prices():
    """Tüm güncel fiyatlar"""
    if not price_feed:
        raise HTTPException(status_code=503, detail="Fiyat beslemesi başlatılmadı")
    
    prices = await price_feed.get_all_prices()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "count": len(prices),
        "prices": {
            ticker: {
                "price": tick.price,
                "change_pct": tick.change_pct,
                "volume": tick.volume,
                "source": tick.source
            }
            for ticker, tick in prices.items()
        }
    }


@app.get("/api/hot-stocks")
async def get_hot_stocks():
    """En popüler hisseler"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline başlatılmadı")
    
    hot = pipeline.get_hot_stocks()
    data = await pipeline.get_bulk_data(hot)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "stocks": [
            {
                "ticker": t,
                "price": data[t].get("current_price", 0) if t in data else 0,
                "change_pct": data[t].get("change_pct", 0) if t in data else 0,
            }
            for t in hot if t in data
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")