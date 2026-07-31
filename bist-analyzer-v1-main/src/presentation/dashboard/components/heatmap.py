# src/presentation/dashboard/components/heatmap.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def render_sector_heatmap():
    """Sektör ısı haritası görselleştirmesi."""
    
    st.markdown("## 🗺️ Sektör Isı Haritası")
    st.markdown("*BIST 100 sektör dağılımı ve performans analizi*")
    
    sectors = [
        {"name": "Bankacılık", "score": 72, "change": 1.5, "count": 8, "weight": 22.5},
        {"name": "Ulaştırma", "score": 68, "change": 2.1, "count": 4, "weight": 10.8},
        {"name": "Savunma", "score": 85, "change": 3.2, "count": 2, "weight": 5.4},
        {"name": "Perakende", "score": 55, "change": -0.5, "count": 5, "weight": 7.5},
        {"name": "Kimya", "score": 42, "change": -1.8, "count": 6, "weight": 4.7},
        {"name": "Metal Ana", "score": 38, "change": -2.3, "count": 4, "weight": 6.8},
        {"name": "Petrol", "score": 62, "change": 0.8, "count": 2, "weight": 8.2},
        {"name": "Holding", "score": 58, "change": 0.3, "count": 3, "weight": 6.2},
    ]
    
    # Üst metrikler
    col1, col2, col3, col4 = st.columns(4)
    best = max(sectors, key=lambda x: x["score"])
    worst = min(sectors, key=lambda x: x["score"])
    
    col1.metric("En Güçlü Sektör", best["name"], delta=f"Skor: {best['score']}")
    col2.metric("En Zayıf Sektör", worst["name"], delta=f"Skor: {worst['score']}", delta_color="inverse")
    col3.metric("Ortalama Skor", f"{np.mean([s['score'] for s in sectors]):.0f}/100")
    col4.metric("Pozitif Sektör", f"{sum(1 for s in sectors if s['change']>0)}/{len(sectors)}")
    
    # Treemap
    colors = ['#00ff88' if s['score']>=70 else '#ffaa00' if s['score']>=50 else '#ff4444' for s in sectors]
    
    fig = go.Figure(go.Treemap(
        labels=[f"{s['name']}<br>({s['count']} hisse)" for s in sectors],
        parents=[""] * len(sectors),
        values=[s['weight'] for s in sectors],
        textinfo="label+value",
        texttemplate="%{label}<br>%{value:.1f}%",
        marker=dict(colors=colors, line=dict(width=2, color='#1a1a2e')),
    ))
    fig.update_layout(
        margin=dict(t=20, l=10, r=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccc', size=12),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detay tablosu
    st.markdown("### 📊 Sektör Detayları")
    df = pd.DataFrame(sectors)
    df.columns = ["Sektör", "Skor", "Değişim %", "Hisse", "Ağırlık %"]
    st.dataframe(df, use_container_width=True, hide_index=True)