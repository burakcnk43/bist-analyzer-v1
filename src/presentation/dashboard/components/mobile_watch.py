# src/presentation/dashboard/components/mobile_watch.py
import streamlit as st
from datetime import datetime


def render_mobile_watch_view():
    """Mobil uyumlu kompakt izleme paneli."""
    
    st.markdown("""
    <style>
        .watch-card {
            background: linear-gradient(135deg, rgba(0,0,0,0.8), rgba(20,20,40,0.9));
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 15px;
            margin: 8px 0;
            backdrop-filter: blur(10px);
        }
        .watch-ticker {
            font-size: 18px;
            font-weight: 700;
            color: #ccc;
            letter-spacing: 1px;
        }
        .watch-price {
            font-size: 42px;
            font-weight: 900;
            color: #fff;
            font-family: 'Courier New', monospace;
        }
        .watch-change-positive { color: #00ff88; font-size: 16px; font-weight: 600; }
        .watch-change-negative { color: #ff4444; font-size: 16px; font-weight: 600; }
        .watch-metric-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 8px;
            margin-top: 10px;
        }
        .watch-metric {
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            padding: 8px;
            text-align: center;
        }
        .watch-metric-value { font-size: 14px; font-weight: 700; color: #00ff88; }
        .watch-metric-label { font-size: 10px; color: #888; text-transform: uppercase; }
        .watch-alert-dot {
            width: 8px; height: 8px; border-radius: 50%;
            display: inline-block; margin-right: 5px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; }
        }
        .watch-alert-active { background: #ff4444; }
        .watch-alert-inactive { background: #00ff88; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("## ⌚ Canlı İzleme")
    st.caption(f"Son güncelleme: {datetime.now().strftime('%H:%M:%S')} • Otomatik yenileme: 15s")
    
    watchlist = [
        {"ticker": "THYAO", "price": 285.50, "change": 2.35, "score": 72, "rsi": 52, "volume": "125M", "alert": True},
        {"ticker": "BIMAS", "price": 385.25, "change": -0.85, "score": 68, "rsi": 45, "volume": "89M", "alert": False},
        {"ticker": "ASELS", "price": 62.30, "change": 1.20, "score": 85, "rsi": 58, "volume": "210M", "alert": True},
        {"ticker": "GARAN", "price": 92.40, "change": -1.50, "score": 55, "rsi": 38, "volume": "340M", "alert": False},
    ]
    
    for stock in watchlist:
        change_class = "watch-change-positive" if stock["change"] >= 0 else "watch-change-negative"
        change_sign = "+" if stock["change"] >= 0 else ""
        
        st.markdown(f"""
        <div class="watch-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="watch-ticker">{stock['ticker']}</span>
                    <span class="{'watch-alert-dot watch-alert-active' if stock['alert'] else 'watch-alert-dot watch-alert-inactive'}"></span>
                </div>
                <div style="text-align: right;">
                    <div class="watch-price">{stock['price']:.2f} ₺</div>
                    <div class="{change_class}">{change_sign}{stock['change']:.2f}%</div>
                </div>
            </div>
            <div class="watch-metric-grid">
                <div class="watch-metric">
                    <div class="watch-metric-label">Skor</div>
                    <div class="watch-metric-value">{stock['score']}</div>
                </div>
                <div class="watch-metric">
                    <div class="watch-metric-label">RSI</div>
                    <div class="watch-metric-value">{stock['rsi']}</div>
                </div>
                <div class="watch-metric">
                    <div class="watch-metric-label">Hacim</div>
                    <div class="watch-metric-value">{stock['volume']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)