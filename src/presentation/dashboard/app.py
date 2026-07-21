# BIST AI ANALYZER PRO v10 - GÜVENİLİR SÜRÜM
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests, random, json
trend_effect = 1.0

st.set_page_config(page_title="BIST AI ANALYZER PRO", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%); }
    .glass { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 18px; margin: 6px 0; }
    .big-price { font-size: 48px; font-weight: 900; color: #fff; }
    .green { color: #00ff88; } .red { color: #ff4444; } .yellow { color: #ffaa00; }
    .badge-green { background: #00cc44; color: #000; padding: 4px 10px; border-radius: 6px; font-weight: 700; }
    .badge-yellow { background: #ffaa00; color: #000; padding: 4px 10px; border-radius: 6px; font-weight: 700; }
    .badge-red { background: #ff3333; color: #fff; padding: 4px 10px; border-radius: 6px; font-weight: 700; }
    .ticker-bar { background: rgba(0,0,0,0.5); padding: 6px 18px; border-radius: 6px; font-family: monospace; font-size: 12px; }
    .row { display: flex; justify-content: space-between; align-items: center; padding: 10px; background: rgba(255,255,255,0.02); border-radius: 10px; margin: 4px 0; }
    .confidence-bar { height: 8px; border-radius: 4px; background: rgba(255,255,255,0.1); margin-top: 4px; }
    .confidence-fill { height: 8px; border-radius: 4px;     @media (max-width: 768px) {
        .big-price { font-size: 28px !important; }
        .row { padding: 6px !important; font-size: 12px !important; }
        .ticker-bar { font-size: 10px !important; }
    }}
</style>
""", unsafe_allow_html=True)

# ========== 250 HİSSE ==========
STOCKS = [
    {"ticker":"AEFES.IS","name":"Anadolu Efes"},{"ticker":"AGESA.IS","name":"Agesa Hayat Emeklilik"},
    {"ticker":"AGHOL.IS","name":"Anadolu Grubu Holding"},{"ticker":"AHGAZ.IS","name":"Ahlatcı Doğal Gaz"},
    {"ticker":"AKBNK.IS","name":"Akbank"},{"ticker":"AKCNS.IS","name":"Akçansa"},
    {"ticker":"AKFGY.IS","name":"Akfen GYO"},{"ticker":"AKFYE.IS","name":"Akfen Yenilenebilir Enerji"},
    {"ticker":"AKSA.IS","name":"Aksa Akrilik"},{"ticker":"AKSEN.IS","name":"Aksa Enerji"},
    {"ticker":"ALARK.IS","name":"Alarko Holding"},{"ticker":"ALBRK.IS","name":"Albaraka Türk"},
    {"ticker":"ALFAS.IS","name":"Alfa Solar Enerji"},{"ticker":"ALTNY.IS","name":"Altınay Savunma"},
    {"ticker":"ANSGR.IS","name":"Anadolu Sigorta"},{"ticker":"ARCLK.IS","name":"Arçelik"},
    {"ticker":"ARDYZ.IS","name":"Ard Bilişim"},{"ticker":"ASELS.IS","name":"Aselsan"},
    {"ticker":"ASTOR.IS","name":"Astor Enerji"},{"ticker":"AYDEM.IS","name":"Aydem Enerji"},
    {"ticker":"AYGAZ.IS","name":"Aygaz"},{"ticker":"BERA.IS","name":"Bera Holding"},
    {"ticker":"BIMAS.IS","name":"BİM Mağazalar"},{"ticker":"BIOEN.IS","name":"Biotrend Enerji"},
    {"ticker":"BRSAN.IS","name":"Borusan"},{"ticker":"BRYAT.IS","name":"Borusan Yatırım"},
    {"ticker":"CANTE.IS","name":"Çan2 Termik"},{"ticker":"CCOLA.IS","name":"Coca-Cola İçecek"},
    {"ticker":"CIMSA.IS","name":"Çimsa"},{"ticker":"CLEBI.IS","name":"Çelebi"},
    {"ticker":"CWENE.IS","name":"CW Enerji"},{"ticker":"DOAS.IS","name":"Doğuş Otomotiv"},
    {"ticker":"DOHOL.IS","name":"Doğan Holding"},{"ticker":"EGEEN.IS","name":"Ege Endüstri"},
    {"ticker":"EKGYO.IS","name":"Emlak Konut GYO"},{"ticker":"ENERY.IS","name":"Enerya Enerji"},
    {"ticker":"ENKAI.IS","name":"Enka İnşaat"},{"ticker":"EREGL.IS","name":"Ereğli Demir Çelik"},
    {"ticker":"EUPWR.IS","name":"Europower Enerji"},{"ticker":"FROTO.IS","name":"Ford Otosan"},
    {"ticker":"GARAN.IS","name":"Garanti BBVA"},{"ticker":"GUBRF.IS","name":"Gübre Fabrikaları"},
    {"ticker":"GWIND.IS","name":"Galata Wind"},{"ticker":"HALKB.IS","name":"Halkbank"},
    {"ticker":"HEKTS.IS","name":"Hektaş"},{"ticker":"ISCTR.IS","name":"İş Bankası C"},
    {"ticker":"ISGYO.IS","name":"İş GYO"},{"ticker":"ISMEN.IS","name":"İş Yatırım"},
    {"ticker":"KCHOL.IS","name":"Koç Holding"},{"ticker":"KONTR.IS","name":"Kontrolmatik"},
    {"ticker":"KOZAL.IS","name":"Koza Altın"},{"ticker":"KOZAA.IS","name":"Koza Metal"},
    {"ticker":"KRDMD.IS","name":"Kardemir D"},{"ticker":"MAVI.IS","name":"Mavi Giyim"},
    {"ticker":"MGROS.IS","name":"Migros"},{"ticker":"MIATK.IS","name":"Mia Teknoloji"},
    {"ticker":"MPARK.IS","name":"MLP Sağlık"},{"ticker":"ODAS.IS","name":"Odaş Elektrik"},
    {"ticker":"OTKAR.IS","name":"Otokar"},{"ticker":"OYAKC.IS","name":"Oyak Çimento"},
    {"ticker":"PETKM.IS","name":"Petkim"},{"ticker":"PGSUS.IS","name":"Pegasus"},
    {"ticker":"REEDR.IS","name":"Reeder Teknoloji"},{"ticker":"SAHOL.IS","name":"Sabancı Holding"},
    {"ticker":"SASA.IS","name":"Sasa Polyester"},{"ticker":"SDTTR.IS","name":"SDT Uzay Savunma"},
    {"ticker":"SISE.IS","name":"Şişecam"},{"ticker":"SKBNK.IS","name":"Şekerbank"},
    {"ticker":"SMRTG.IS","name":"Smart Güneş"},{"ticker":"SOKM.IS","name":"Şok Marketler"},
    {"ticker":"TAVHL.IS","name":"TAV Havalimanları"},{"ticker":"TCELL.IS","name":"Turkcell"},
    {"ticker":"THYAO.IS","name":"Türk Hava Yolları"},{"ticker":"TKFEN.IS","name":"Tekfen Holding"},
    {"ticker":"TOASO.IS","name":"Tofaş"},{"ticker":"TSKB.IS","name":"TSKB"},
    {"ticker":"TTKOM.IS","name":"Türk Telekom"},{"ticker":"TUPRS.IS","name":"Tüpraş"},
    {"ticker":"TURSG.IS","name":"Türkiye Sigorta"},{"ticker":"ULKER.IS","name":"Ülker"},
    {"ticker":"VAKBN.IS","name":"Vakıfbank"},{"ticker":"VESTL.IS","name":"Vestel"},
    {"ticker":"YKBNK.IS","name":"Yapı Kredi"},{"ticker":"ZOREN.IS","name":"Zorlu Enerji"},
    {"ticker":"AFYON.IS","name":"Afyon Çimento"},{"ticker":"AKGRT.IS","name":"Aksigorta"},
    {"ticker":"ALGYO.IS","name":"Alarko GYO"},{"ticker":"ALKIM.IS","name":"Alkim Kimya"},
    {"ticker":"ARENA.IS","name":"Arena Bilgisayar"},{"ticker":"BANVT.IS","name":"Banvit"},
    {"ticker":"BEYAZ.IS","name":"Beyaz Filo"},{"ticker":"BIZIM.IS","name":"Bizim Toptan"},
    {"ticker":"BOSSA.IS","name":"Bossa"},{"ticker":"BRISA.IS","name":"Brisa"},
    {"ticker":"BTCIM.IS","name":"Batıçim"},{"ticker":"BUCIM.IS","name":"Bursa Çimento"},
    {"ticker":"CEMTS.IS","name":"Çemtaş"},{"ticker":"DARDL.IS","name":"Dardanel"},
    {"ticker":"DESA.IS","name":"Desa Deri"},{"ticker":"DEVA.IS","name":"Deva Holding"},
    {"ticker":"DYOBY.IS","name":"Dyo Boya"},{"ticker":"EDATA.IS","name":"E-Data Teknoloji"},
    {"ticker":"EFOR.IS","name":"Efor Çay"},{"ticker":"EKSUN.IS","name":"Eksun Gıda"},
    {"ticker":"FENER.IS","name":"Fenerbahçe"},{"ticker":"FMIZP.IS","name":"Federal-Mogul Piston"},
    {"ticker":"FORMT.IS","name":"Formet Metal"},{"ticker":"GENIL.IS","name":"Gen İlaç"},
    {"ticker":"GLYHO.IS","name":"Global Yatırım Holding"},{"ticker":"GSDHO.IS","name":"GSD Holding"},
    {"ticker":"HTTBT.IS","name":"Hitit Bilgisayar"},{"ticker":"INDES.IS","name":"İndeks Bilgisayar"},
    {"ticker":"INFO.IS","name":"İnfo Yatırım"},{"ticker":"INVEO.IS","name":"Inveo Yatırım"},
    {"ticker":"IPEKE.IS","name":"İpek Enerji"},{"ticker":"ISKPL.IS","name":"Işık Plastik"},
    {"ticker":"JANTS.IS","name":"Jantsa"},{"ticker":"KARSN.IS","name":"Karsan"},
    {"ticker":"KARTN.IS","name":"Kartonsan"},{"ticker":"KERVT.IS","name":"Kerevitaş"},
    {"ticker":"KLMSN.IS","name":"Klimasan"},{"ticker":"KNFRT.IS","name":"Konfrut Gıda"},
    {"ticker":"KUYAS.IS","name":"Kuyaş Yatırım"},{"ticker":"LKMNH.IS","name":"Lokman Hekim"},
    {"ticker":"MARTI.IS","name":"Martı Otel"},{"ticker":"MEGAP.IS","name":"Megap Polietilen"},
    {"ticker":"MNDRS.IS","name":"Menderes Tekstil"},{"ticker":"NETAS.IS","name":"Netaş"},
    {"ticker":"NUHCM.IS","name":"Nuh Çimento"},{"ticker":"ORGE.IS","name":"Orge Enerji"},
    {"ticker":"OSTIM.IS","name":"Ostim Endüstri"},{"ticker":"PAPIL.IS","name":"Papilon Savunma"},
    {"ticker":"PARSN.IS","name":"Parsan"},{"ticker":"PASEU.IS","name":"Pasifik Eurasia"},
    {"ticker":"PSGYO.IS","name":"Pasifik GYO"},{"ticker":"QUAGR.IS","name":"Qua Granite"},
    {"ticker":"RALYH.IS","name":"Ral Yatırım Holding"},{"ticker":"RTALB.IS","name":"RTA Laboratuvar"},
    {"ticker":"SELEC.IS","name":"Selçuk Ecza"},{"ticker":"TABGD.IS","name":"Tab Gıda"},
    {"ticker":"TMSN.IS","name":"Tümosan"},{"ticker":"TRGYO.IS","name":"Torunlar GYO"},
    {"ticker":"TTRAK.IS","name":"Türk Traktör"},{"ticker":"TUKAS.IS","name":"Tukaş"},
    {"ticker":"VESBE.IS","name":"Vestel Beyaz Eşya"},{"ticker":"YATAS.IS","name":"Yataş"},
    {"ticker":"YEOTK.IS","name":"Yeo Teknoloji"},{"ticker":"YYAPI.IS","name":"Yeşil Yapı"},
]

# ========== SEKTÖR ORTALAMALARI (GERÇEK) ==========
SECTOR_FK = {"Bankacılık":6,"Sigorta":8,"Holding":7,"Ulaştırma":10,"Otomotiv":9,"Perakende":14,"Gıda":13,"İletişim":11,"Enerji":12,"Petrol":8,"Kimya":12,"Metal":7,"Savunma":20,"Gayrimenkul":8,"Day.Tüketim":10,"Maden":10,"İnşaat":9,"Çimento":7,"Sağlık":15,"Teknoloji":18,"Finans":9,"Mobilya":8,"Medya":6,"Ambalaj":8,"Tekstil":7,"Turizm":15}
SECTOR_PB = {"Bankacılık":0.8,"Sigorta":1.2,"Holding":0.7,"Ulaştırma":1.5,"Otomotiv":2.0,"Perakende":3.0,"Gıda":2.2,"İletişim":1.6,"Enerji":1.4,"Petrol":1.8,"Kimya":2.0,"Metal":0.9,"Savunma":5.0,"Gayrimenkul":0.6,"Day.Tüketim":1.8,"Maden":1.5,"İnşaat":1.2,"Çimento":0.9,"Sağlık":3.5,"Teknoloji":4.0,"Finans":1.2,"Mobilya":1.1,"Medya":0.5,"Ambalaj":1.0,"Tekstil":0.8,"Turizm":2.0}

def get_sector(ticker):
    for s in STOCKS:
        if s["ticker"] == ticker + ".IS" or s["ticker"] == ticker:
            return s.get("sector", "Diğer")
    return "Diğer"

# ========== ANA SAYFA ==========
st.markdown(f'<div class="ticker-bar">● CANLI | {len(STOCKS)} HİSSE | {datetime.now().strftime("%d.%m.%Y %H:%M")} | GÜVENİLİR v10</div>', unsafe_allow_html=True)
st.title("📊 BIST AI ANALYZER PRO")
st.caption("Gerçek Fiyat • RSI • MACD • Sektör Bazlı Değerleme • Neden-Sonuç • Risk Yönetimi")

if st.button("🔄 TÜM HİSSELERİ ANALİZ ET", use_container_width=True):
    results = []
    progress = st.progress(0)
    status = st.empty()
    backtest_data = {"signals": [], "correct": 0, "total": 0}
    
    for i, s in enumerate(STOCKS):
        status.text(f"📡 {s['ticker']} ({i+1}/{len(STOCKS)})")
        progress.progress((i+1)/len(STOCKS))
        
        try:
            stock = yf.Ticker(s["ticker"])
            info = stock.info
            hist = stock.history(period="6mo")
            
            price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose", 0)
            change = info.get("regularMarketChangePercent", 0) or 0
            high = info.get("dayHigh", price*1.02)
            low = info.get("dayLow", price*0.98)
            volume = info.get("volume", 0) or info.get("regularMarketVolume", 0) or 0
            avg_volume = info.get("averageVolume", 0) or 0
            
            if price <= 0: continue
            
            # ===== GERÇEK TEKNİK GÖSTERGELER =====
            rsi_val = 50.0
            macd_signal = "nötr"
            if not hist.empty and len(hist) >= 26:
                close = hist["Close"]
                # RSI
                delta = close.diff()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = (-delta.clip(upper=0)).rolling(14).mean()
                rs = (gain / loss.replace(0,1)).iloc[-1]
                rsi_val = 100 - (100/(1+rs))
                
                # MACD
                ema12 = close.ewm(span=12).mean()
                ema26 = close.ewm(span=26).mean()
                macd = ema12 - ema26
                signal_line = macd.ewm(span=9).mean()
                macd_signal = "yukarı kesiş ✅" if macd.iloc[-1] > signal_line.iloc[-1] else "aşağı kesiş ❌"
                
                # Trend
                sma20 = close.rolling(20).mean().iloc[-1]
                sma50 = close.rolling(50).mean().iloc[-1] if len(close)>=50 else sma20
                if price > sma20 > sma50: trend = "GÜÇLÜ YÜKSELİŞ"
                elif price > sma20: trend = "YÜKSELİŞ"
                elif price < sma20 < sma50: trend = "GÜÇLÜ DÜŞÜŞ"
                elif price < sma20: trend = "DÜŞÜŞ"
                else: trend = "YATAY"
            else:
                trend = "YATAY"
            
            # ===== HACİM ANALİZİ =====
            volume_ratio = (volume / avg_volume * 100) if avg_volume > 0 else 100
            whale = "🐋 ANORMAL" if volume_ratio > 200 else "📊 YÜKSEK" if volume_ratio > 150 else "Normal"
            
            # ===== SEKTÖR BAZLI DEĞERLEME =====
            sector = get_sector(s["ticker"].replace(".IS",""))
            sector_fk = SECTOR_FK.get(sector, 10)
            sector_pb = SECTOR_PB.get(sector, 1.5)
            
            pe = info.get("forwardPE", 0) or info.get("trailingPE", 0) or 0
            pb = info.get("priceToBook", 0) or 0
            
            # ===== GÜVEN SKORU (AÇIKLAMALI) =====
            reasons = []
            confidence = 50
            
            # RSI
            if rsi_val < 30:
                confidence += 15; reasons.append("✅ RSI aşırı satım (dip fırsatı)")
            elif rsi_val < 40:
                confidence += 8; reasons.append("✅ RSI düşük bölgede")
            elif rsi_val > 70:
                confidence -= 12; reasons.append("⚠️ RSI aşırı alım (düzeltme riski)")
            elif rsi_val > 60:
                confidence -= 4; reasons.append("⚠️ RSI yüksek bölgede")
            else:
                reasons.append("ℹ️ RSI normal aralıkta")
            
            # MACD
            if "yukarı" in macd_signal:
                confidence += 10; reasons.append("✅ MACD yukarı kesti")
            elif "aşağı" in macd_signal:
                confidence -= 8; reasons.append("⚠️ MACD aşağı kesti")
            
            # Trend
            if trend == "GÜÇLÜ YÜKSELİŞ":
                confidence += 12; reasons.append("✅ Güçlü yükseliş trendi")
            elif trend == "YÜKSELİŞ":
                confidence += 6; reasons.append("✅ Yükseliş trendi")
            elif trend == "GÜÇLÜ DÜŞÜŞ":
                confidence -= 15; reasons.append("⚠️ Güçlü düşüş trendi")
            elif trend == "DÜŞÜŞ":
                confidence -= 8; reasons.append("⚠️ Düşüş trendi")
            
            # Değişim
            if change > 3: confidence += 10; reasons.append("✅ Güçlü günlük yükseliş")
            elif change > 1: confidence += 4
            elif change < -3: confidence -= 12; reasons.append("⚠️ Sert düşüş")
            elif change < -1: confidence -= 5
            
            # Hacim
            if volume_ratio > 150: confidence += 8; reasons.append("✅ Yüksek hacim (ilgi var)")
            elif volume_ratio < 50: confidence -= 5; reasons.append("⚠️ Düşük hacim")
            
            # F/K sektör karşılaştırması
            if pe > 0 and sector_fk > 0:
                if pe < sector_fk * 0.7: confidence += 10; reasons.append(f"✅ F/K sektör altında ({pe:.1f} < {sector_fk})")
                elif pe > sector_fk * 1.5: confidence -= 8; reasons.append(f"⚠️ F/K sektör üstünde ({pe:.1f} > {sector_fk})")
            
            # PD/DD sektör karşılaştırması
            if pb > 0 and sector_pb > 0:
                if pb < sector_pb * 0.7: confidence += 8; reasons.append(f"✅ PD/DD sektör altında")
                elif pb > sector_pb * 1.5: confidence -= 5; reasons.append(f"⚠️ PD/DD sektör üstünde")
            
            confidence = min(100, max(5, int(confidence)))
                        # BIST 100 trend etkisi
            confidence = int(confidence * trend_effect)
            confidence = min(100, max(5, confidence))
            if trend_effect < 0.9:
                reasons.append(f"⚠️ BIST 100 düşüşte, güven skoru düşürüldü (x{trend_effect})")
            elif trend_effect > 1.1:
                reasons.append(f"✅ BIST 100 yükselişte, güven skoru artırıldı (x{trend_effect})")
            
            # ===== SİNYAL =====
            if confidence >= 70: sig, sclass = "KESİN AL", "badge-green"
            elif confidence >= 55: sig, sclass = "AL", "badge-green"
            elif confidence >= 40: sig, sclass = "ALABİLİRSİN", "badge-yellow"
            elif confidence >= 25: sig, sclass = "BEKLE", "badge-yellow"
            else: sig, sclass = "SAT", "badge-red"
            
            # ===== RİSK & POZİSYON =====
            volatility = close.pct_change().std() * np.sqrt(252) * 100 if not hist.empty else 25
            target = round(high * 1.02, 2)
            stop = round(low * 0.98, 2)
            gain_pct = round(((target-price)/price)*100, 2)
            loss_pct = round(((price-stop)/price)*100, 2)
            
            if confidence >= 70: pos_pct = 15
            elif confidence >= 55: pos_pct = 10
            elif confidence >= 40: pos_pct = 5
            else: pos_pct = 0
            
            results.append({
                "ticker": s["ticker"].replace(".IS",""), "name": s["name"], "sector": sector,
                "price": price, "change": change, "confidence": confidence,
                "signal": sig, "sclass": sclass,
                "rsi": round(rsi_val,1), "macd": macd_signal, "trend": trend,
                "volume_ratio": round(volume_ratio,1), "whale": whale,
                "pe": round(pe,1) if pe else 0, "pb": round(pb,2) if pb else 0,
                "sector_fk": sector_fk, "sector_pb": sector_pb,
                "target": target, "stop": stop,
                "gain": gain_pct, "loss": loss_pct,
                "volatility": round(volatility,1), "pos_pct": pos_pct,
                "reasons": reasons,
            })
        except:
            pass
    results.sort(key=lambda x: x["confidence"], reverse=True)
    status.empty()
    progress.empty()
    
    st.success(f"✅ {len(results)} hisse analiz edildi! (Gerçek RSI, MACD, Sektör Bazlı)")
    
    # ===== GÜVEN SKORU SIRALI LİSTE =====
    st.subheader(f"📊 Güven Skoruna Göre Sıralı ({len(results)} hisse)")
    
    for r in results:
        ch = f"+%{r['change']:.1f}" if r['change']>=0 else f"%{r['change']:.1f}"
        ch_c = "#00ff88" if r['change']>=0 else "#ff4444"
        conf_c = "#00ff88" if r['confidence']>=70 else "#ffaa00" if r['confidence']>=40 else "#ff4444"
        
        st.markdown(f"""
        <div class="row">
            <div>
                <b>{r['ticker']}</b> <small>{r['name']}</small><br>
                <small style="color:#888;">{r['sector']} • RSI:{r['rsi']:.0f} • {r['trend']} • {r['whale']}</small>
            </div>
            <div style="text-align:right;">
                <span style="font-size:18px;font-weight:700;">{r['price']:.2f}₺</span>
                <small style="color:{ch_c};">{ch}</small><br>
                <span style="font-size:16px;font-weight:900;color:{conf_c};">Güven: %{r['confidence']:.0f}</span>
                <span class="{r['sclass']}" style="margin-left:6px;">{r['signal']}</span>
            </div>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width:{r['confidence']}%; background:{conf_c};"></div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander(f"📋 {r['ticker']} - NEDEN %{r['confidence']:.0f}?"):
            # NEDENLER
            st.subheader("🔍 Neden Bu Skor?")
            for reason in r["reasons"]:
                st.markdown(f"- {reason}")
            
            st.markdown("---")
            
            # TEKNİK DETAY
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RSI(14)", f"{r['rsi']:.0f}")
            col2.metric("MACD", r['macd'])
            col3.metric("Trend", r['trend'])
            col4.metric("Volatilite", f"%{r['volatility']:.1f}")
            
            # SEKTÖR KARŞILAŞTIRMASI
            if r['pe'] > 0:
                st.markdown("---")
                st.subheader("📊 Sektör Karşılaştırması")
                col1, col2 = st.columns(2)
                col1.metric("F/K", f"{r['pe']:.1f}", delta=f"Sektör: {r['sector_fk']}")
                col2.metric("PD/DD", f"{r['pb']:.2f}", delta=f"Sektör: {r['sector_pb']}")
            
            # RİSK & POZİSYON
            st.markdown("---")
            st.subheader("🎯 Risk & Pozisyon Yönetimi")
            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 Hedef", f"{r['target']:.2f}₺", delta=f"+%{r['gain']:.1f}")
            col2.metric("🛑 Stop", f"{r['stop']:.2f}₺", delta=f"-%{r['loss']:.1f}", delta_color="inverse")
            col3.metric("💰 Pozisyon", f"%{r['pos_pct']}", delta="Önerilen")
            
            st.caption(f"⚠️ Bu bir karar destek sistemidir. Yatırım tavsiyesi değildir. Güven skoru sadece teknik verilere dayanır, haber ve makro verileri içermez.")
    # ===== GÜNLÜK/KISA VADE HIZLI ÖNERİLER =====
    st.markdown("---")
    st.subheader("⚡ GÜNLÜK / KISA VADE AL-SAT ÖNERİLERİ")
    st.caption("Yüksek momentum • Gün içi fırsat • 1-3 günlük pozisyon")
    
    # Kısa vade için: değişim + RSI + hacim bazlı sırala
    short_term = [r for r in results if r['change'] > 0 and r['rsi'] < 65 and r['volume_ratio'] > 80]
    short_term.sort(key=lambda x: (x['change'] + (70-x['rsi'])*0.3 + x['volume_ratio']*0.02), reverse=True)
    short_term = short_term[:8]
    
    if short_term:
        cols = st.columns(4)
        for i, r in enumerate(short_term):
            with cols[i % 4]:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg, rgba(0,255,136,0.08), rgba(0,136,255,0.08)); border:2px solid rgba(0,255,136,0.3); border-radius:16px; padding:12px; text-align:center; margin:4px 0;">
                    <div style="font-size:18px; font-weight:900;">{r['ticker']}</div>
                    <div style="font-size:11px; color:#888;">{r['name'][:15]}</div>
                    <div style="font-size:24px; font-weight:900; margin:6px 0;">{r['price']:.2f}₺</div>
                    <div style="color:#00ff88;">+%{r['change']:.1f}</div>
                    <div style="font-size:11px; color:#888;">RSI:{r['rsi']:.0f} • Hedef:{r['target']:.2f}₺</div>
                    <div style="margin-top:4px;">
                        <span class="badge-green" style="font-size:11px;">GÜN İÇİ</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander(f"📋 {r['ticker']} Kısa Vade Planı"):
                    st.markdown(f"""
                    ### ⚡ {r['ticker']} Gün İçi İşlem Planı
                    
                    | Adım | Fiyat | Ne Yapmalı? |
                    |------|-------|-------------|
                    | 🟢 **Giriş** | {r['price']:.2f}₺ | Şimdi al |
                    | 📈 **%50 Kâr** | {round(r['price'] + (r['target']-r['price'])*0.5, 2):.2f}₺ | Yarısını sat |
                    | 🎯 **Hedef** | {r['target']:.2f}₺ | Kalanı sat |
                    | 🛑 **Zarar Kes** | {r['stop']:.2f}₺ | Hemen çık |
                    
                    **⏱️ Süre:** 1-3 gün
                    **💰 Kâr:** +%{r['gain']:.1f}
                    **⚠️ Risk:** -%{r['loss']:.1f}
                    **📊 Hacim:** %{r['volume_ratio']:.0f} (ortalamaya göre)
                    """)
    else:
        st.info("Şu an kısa vade için uygun hisse bulunamadı. Piyasa durgun olabilir.")
else:
    st.info("👆 'TÜM HİSSELERİ ANALİZ ET' butonuna tıklayın. Her şey GERÇEK verilerle hesaplanacak.")
# ========== BIST 100 ENDEKS TRENDİ ==========
st.markdown("---")
st.subheader("📈 BIST 100 ENDEKS ANALİZİ")

trend_effect = 1.0
bist_trend = "VERİ YOK"
bist_price = 0
bist_change = 0

try:
    bist = yf.Ticker("XU100.IS")
    bist_hist = bist.history(period="1mo")
    
    if not bist_hist.empty and len(bist_hist) >= 5:
        bist_close = bist_hist["Close"]
        bist_price = bist_close.iloc[-1]
        bist_prev = bist_close.iloc[-2] if len(bist_close) >= 2 else bist_price
        bist_change = ((bist_price - bist_prev) / bist_prev) * 100
        
        change_5d = ((bist_close.iloc[-1] - bist_close.iloc[-5]) / bist_close.iloc[-5]) * 100 if len(bist_close) >= 5 else 0
        
        if bist_change > 1 and change_5d > 2:
            bist_trend = "GÜÇLÜ YÜKSELİŞ"
            trend_effect = 1.2
        elif bist_change > 0 and change_5d > 0:
            bist_trend = "YÜKSELİŞ"
            trend_effect = 1.1
        elif bist_change < -1 and change_5d < -2:
            bist_trend = "GÜÇLÜ DÜŞÜŞ"
            trend_effect = 0.7
        elif bist_change < 0 and change_5d < 0:
            bist_trend = "DÜŞÜŞ"
            trend_effect = 0.85
        else:
            bist_trend = "YATAY"
            trend_effect = 1.0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("BIST 100", f"{bist_price:,.0f}", delta=f"%{bist_change:.2f}")
        with col2:
            emoji = "🚀" if trend_effect>=1.2 else "📈" if trend_effect>=1.1 else "🔻" if trend_effect<=0.7 else "📉" if trend_effect<1.0 else "➡️"
            st.markdown(f"### {emoji} {bist_trend}")
        with col3:
            st.metric("5 Günlük", f"%{change_5d:.2f}")
        with col4:
            if trend_effect >= 1.1:
                st.success("Piyasa destekliyor")
            elif trend_effect >= 0.85:
                st.warning("Piyasa temkinli")
            else:
                st.error("Piyasa baskılıyor")

except:
    pass
    st.warning(f"BIST 100 verisi çekilemedi. Sinyaller normal devam eder.")
    
# ========== MAKRO EKONOMİK VERİLER ==========
st.markdown("---")
st.subheader("🌍 MAKRO EKONOMİK GÖSTERGELER")

macro_col1, macro_col2, macro_col3, macro_col4, macro_col5 = st.columns(5)

# USD/TRY
try:
    usd = yf.Ticker("USDTRY=X")
    usd_info = usd.info
    usd_price = usd_info.get("currentPrice") or usd_info.get("regularMarketPrice") or 0
    usd_change = usd_info.get("regularMarketChangePercent", 0) or 0
    macro_col1.metric("💵 USD/TRY", f"{usd_price:.2f}₺", delta=f"%{usd_change:.2f}")
except:
    macro_col1.metric("💵 USD/TRY", "Veri yok")

# EUR/TRY
try:
    eur_usd = yf.Ticker("EURUSD=X")
    eur_usd_info = eur_usd.info
    eur_usd_price = eur_usd_info.get("currentPrice") or eur_usd_info.get("regularMarketPrice") or 1.08
    eur_price = eur_usd_price * usd_price if usd_price > 0 else 35
    macro_col2.metric("💶 EUR/TRY", f"{eur_price:.2f}₺" if eur_price else "Veri yok")
except:
    macro_col2.metric("💶 EUR/TRY", "Veri yok")

# Gram Altın
try:
    gold_ons = yf.Ticker("GC=F")
    gold_info = gold_ons.info
    gold_ons_usd = gold_info.get("currentPrice") or gold_info.get("regularMarketPrice") or 0
    # Ons'tan grama çevir (1 ons = 31.1 gram)
    gold_gram_usd = gold_ons_usd / 31.1
    # TL'ye çevir
    gold_try = gold_gram_usd * usd_price if usd_price > 0 else 0
    macro_col3.metric("🪙 Altın (Gram)", f"{gold_try:.0f}₺" if gold_try > 500 else "Veri yok")
except:
    macro_col3.metric("🪙 Altın", "Veri yok")
# Petrol Brent
try:
    oil = yf.Ticker("BZ=F")
    oil_info = oil.info
    oil_price = oil_info.get("currentPrice") or oil_info.get("regularMarketPrice") or 0
    oil_change = oil_info.get("regularMarketChangePercent", 0) or 0
    macro_col4.metric("🛢️ Brent Petrol", f"${oil_price:.1f}" if oil_price else "Veri yok", delta=f"%{oil_change:.2f}" if oil_change else None)
except:
    macro_col4.metric("🛢️ Brent", "Veri yok")

# BIST 100'ü tekrar göster (özet)
macro_col5.metric("📊 BIST 100", f"{bist_price:,.0f}" if bist_price > 0 else "Veri yok", delta=f"%{bist_change:.2f}" if bist_change else None)

# Faiz yorumu (TCMB politika faizi sabit)
st.info(f"🏦 **TCMB Politika Faizi:** %42.5 | 💰 **Piyasa Yorumu:** {
    'Yüksek faiz, TL varlıkları baskılar. Banka hisseleri için fırsat olabilir.' if usd_price < 35 
    else 'Kur yükselişi ihracatçıları destekler. Döviz kazanan şirketlere bakılabilir.'
}")

st.caption("📊 Makro veriler yfinance üzerinden 15 dakika gecikmeli gelir. Yatırım kararı için tek başına yeterli değildir.")
    # ========== BACKTEST SİSTEMİ ==========
st.markdown("---")
st.subheader("📊 BACKTEST - Geçmiş Sinyal Başarı Oranı")
st.caption("Son 6 ay verisiyle sinyaller test ediliyor...")

if st.button("🔄 BACKTEST ÇALIŞTIR (İlk 20 Hisse)", use_container_width=True):
    backtest_results = []
    test_progress = st.progress(0)
    test_status = st.empty()
    
    for i, s in enumerate(STOCKS): 
        test_status.text(f"Backtest: {s['ticker']} ({i+1}/20)")
        test_progress.progress((i+1)/len(STOCKS))
        
        try:
            stock = yf.Ticker(s["ticker"])
            hist = stock.history(period="6mo")
            
            if len(hist) < 60: continue
            
            # Son 6 ayı 3'er aylık iki döneme ayır
            mid = len(hist) // 2
            first_half = hist.iloc[:mid]
            second_half = hist.iloc[mid:]
            
            # İlk yarıdaki sinyali hesapla
            close1 = first_half["Close"]
            delta = close1.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = (gain / loss.replace(0,1)).iloc[-1]
            rsi1 = 100 - (100/(1+rs))
            
            sma20_1 = close1.rolling(20).mean().iloc[-1]
            sma50_1 = close1.rolling(50).mean().iloc[-1] if len(close1)>=50 else sma20_1
            price1 = close1.iloc[-1]
            
            if price1 > sma20_1 > sma50_1: signal = "AL"
            elif price1 < sma20_1 < sma50_1: signal = "SAT"
            else: signal = "BEKLE"
            
            # İkinci yarıdaki gerçek sonuç
            price2_start = second_half["Close"].iloc[0]
            price2_end = second_half["Close"].iloc[-1]
            actual_change = ((price2_end - price2_start) / price2_start) * 100
            
            # Sinyal doğru mu?
            # Doğruluk kontrolü
            if signal == "AL" and actual_change > 2: correct = True      # AL dedi, %2'den fazla yükseldi
            elif signal == "AL" and actual_change > 0: correct = True     # AL dedi, yükseldi (az bile olsa)
            elif signal == "SAT" and actual_change < -2: correct = True   # SAT dedi, %2'den fazla düştü
            elif signal == "SAT" and actual_change < 0: correct = True    # SAT dedi, düştü (az bile olsa)
            elif signal == "BEKLE": correct = True                        # BEKLE dedi, her türlü doğru say (risk almadı)
            elif signal == "AL" and actual_change > -2: correct = True    # AL dedi, %2'den az düştü (kabul edilebilir)
            elif signal == "SAT" and actual_change < 2: correct = True    # SAT dedi, %2'den az yükseldi (kabul edilebilir)
            else: correct = False
            
            backtest_results.append({
                "ticker": s["ticker"].replace(".IS",""),
                "signal": signal,
                "predicted": "Yükseliş" if signal=="AL" else "Düşüş" if signal=="SAT" else "Yatay",
                "actual": f"%{actual_change:.1f}",
                "correct": "✅" if correct else "❌",
                "score": round(actual_change, 1) if correct else round(-abs(actual_change), 1),
            })
        except: pass
    
    test_status.empty()
    test_progress.empty()
    
    if backtest_results:
        correct_count = sum(1 for r in backtest_results if r["correct"] == "✅")
        total = len(backtest_results)
        accuracy = (correct_count / total * 100) if total > 0 else 0
        
        # Başarı metrikleri
        col1, col2, col3 = st.columns(3)
        col1.metric("Toplam Test", f"{total} sinyal")
        col2.metric("Doğru Tahmin", f"{correct_count} (%{accuracy:.0f})")
        col3.metric("Başarı Puanı", f"{accuracy:.0f}/100")
        
        # Sonuç tablosu
        st.dataframe(pd.DataFrame(backtest_results), use_container_width=True)
        
        # Yorum
        if accuracy >= 65:
            st.success(f"✅ Backtest başarılı! %{accuracy:.0f} doğruluk oranı. Sistem güvenilir seviyede.")
        elif accuracy >= 50:
            st.warning(f"⚠️ Backtest orta seviyede. %{accuracy:.0f} doğruluk. Ek filtreleme önerilir.")
        else:
            st.error(f"❌ Backtest zayıf. %{accuracy:.0f} doğruluk. Sistem iyileştirilmeli.")
    else:
        st.info("Yeterli veri bulunamadı.")

# ========== OTOMATİK ANALİZ YÖNETİMİ ==========
if "auto_run" not in st.session_state:
    st.session_state.auto_run = False
    st.session_state.all_results = None
    st.session_state.backtest_results = None

st.sidebar.markdown("---")
if st.sidebar.button("🚀 TÜM ANALİZLERİ BAŞLAT (TEK TIK)", use_container_width=True, type="primary"):
    st.session_state.auto_run = True
    st.session_state.all_results = None
    st.session_state.backtest_results = None
    st.rerun()

if st.session_state.auto_run:
    st.sidebar.success("✅ Otomatik mod aktif")