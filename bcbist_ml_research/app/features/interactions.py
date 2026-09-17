import pandas as pd
import numpy as np

def add_elite_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates high-order interaction features to discover hidden alpha patterns.
    """
    df = df.copy()

    # 1. Price-Volume-Volatility Synergy
    if 'relative_volume' in df.columns and 'dist_sma_20' in df.columns:
        df['pv_synergy'] = df['relative_volume'] * df['dist_sma_20']

    if 'bb_pct' in df.columns and 'rsi_14' in df.columns:
        df['overbought_squeeze'] = df['bb_pct'] * df['rsi_14']

    # 2. Trend Acceleration
    if 'ma_slope_20' in df.columns and 'ma_slope_50' in df.columns:
        df['trend_convergence'] = df['ma_slope_20'] / (df['ma_slope_50'] + 1e-9)

    # 3. Sector & Market Pulse Interaction
    if 'rel_sector_return_1d' in df.columns and 'market_z_relative_volume' in df.columns:
        df['sector_market_alpha_sync'] = df['rel_sector_return_1d'] * df['market_z_relative_volume']

    # 4. News-Momentum Fusion (If sentiment is available)
    if 'news_sentiment' in df.columns:
        if 'breakout_volume_ratio' in df.columns:
            df['sentiment_breakout_on_fire'] = df['news_sentiment'] * df['breakout_volume_ratio']
        if 'return_1d' in df.columns:
            df['sent_price_divergence'] = df['news_sentiment'] - df['return_1d']

    return df.fillna(0)
