# run.py
# BIST AI ANALYZER PRO - ANA ÇALIŞTIRMA DOSYASI
import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def main():
    print("=" * 60)
    print("  BIST AI ANALYZER PRO - GERÇEK VERİ SİSTEMİ")
    print(f"  Başlangıç: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print("=" * 60)
    
    from src.data.sources.kap_client import KAPClient, KAPConfig
    from src.data.cache.redis_manager import RedisManager
    from src.data.sources.realtime_price_feed import RealTimePriceFeed
    from src.data.repositories.technical_repo import TechnicalRepository
    from src.data.pipeline.fast_data_pipeline import FastDataPipeline
    
    print("\n[1/4] Veri kaynakları başlatılıyor...")
    
    kap_config = KAPConfig(timeout=15, max_retries=3)
    kap = KAPClient(kap_config)
    await kap.__aenter__()
    print("   ✓ KAP Client hazır")
    
    redis = RedisManager()
    print("   ✓ Redis Cache hazır")
    
    price_feed = RealTimePriceFeed()
    print("   ✓ Fiyat beslemesi hazır")
    
    technical = TechnicalRepository()
    print("   ✓ Teknik analiz hazır")
    
    # Pipeline başlat
    print("\n[2/4] Veri boru hattı başlatılıyor...")
    pipeline = FastDataPipeline(kap, redis, price_feed, technical)
    await pipeline.start()
    
    # BIST 100 listesini çek
    bist100 = await kap.get_bist100_list()
    tickers = [s["ticker"] for s in bist100[:30]]
    print(f"   ✓ {len(tickers)} hisse yüklendi")
    
    # Fiyat beslemesini başlat
    print("\n[3/4] Gerçek zamanlı fiyat beslemesi başlatılıyor...")
    await price_feed.start(tickers)
    print("   ✓ Fiyat beslemesi aktif")
    
    # İlk veri çekme
    print("\n[4/4] İlk veri çekiliyor...")
    data = await pipeline.get_bulk_data(tickers[:10])
    print(f"   ✓ {len(data)} hisse verisi çekildi")
    
    print("\n" + "=" * 60)
    print("  SİSTEM HAZIR!")
    print("  Dashboard: streamlit run src/presentation/dashboard/app.py")
    print("  API: uvicorn src.presentation.api.fastapi_routes:app --port 8000")
    print("=" * 60)
    
    # Örnek hisse göster
    for ticker in tickers[:5]:
        if ticker in data:
            d = data[ticker]
            print(f"  {ticker}: {d.get('current_price', 0):.2f} TL | RSI: {d.get('technicals', {}).get('rsi_14', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(main())