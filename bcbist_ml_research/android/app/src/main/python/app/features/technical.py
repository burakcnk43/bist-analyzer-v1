import pandas as pd
import pandas_ta as ta
import numpy as np
import logging

logger = logging.getLogger(__name__)

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a comprehensive set of technical indicators with slope and acceleration.
    """
    df = df.copy()

    def safe_concat(base_df, new_cols):
        if new_cols is None: return base_df
        if isinstance(new_cols, pd.Series): new_cols = new_cols.to_frame()
        if new_cols.empty: return base_df
        cols_to_add = new_cols.columns.difference(base_df.columns)
        if cols_to_add.empty: return base_df
        return pd.concat([base_df, new_cols[cols_to_add]], axis=1)

    # 1. Base Indicators
    df = safe_concat(df, df.ta.sma(length=20))
    df = safe_concat(df, df.ta.sma(length=50))
    df = safe_concat(df, df.ta.sma(length=200))
    df = safe_concat(df, df.ta.rsi(length=14))
    df = safe_concat(df, df.ta.rsi(length=7))
    df = safe_concat(df, df.ta.macd())
    df = safe_concat(df, df.ta.bbands(length=20, std=2))
    df = safe_concat(df, df.ta.atr(length=14))
    df = safe_concat(df, df.ta.adx())

    df.columns = [c.lower() for c in df.columns]

    # 2. Slopes and Acceleration
    for col in ['rsi_14', 'macd_12_26_9', 'macdh_12_26_9']:
        if col in df.columns:
            df[f'{col}_slope_3'] = df[col].diff(3)
            df[f'{col}_slope_5'] = df[col].diff(5)
            df[f'{col}_accel_5'] = df[f'{col}_slope_3'].diff(2)

    # 3. Moving Average Relationships
    for length in [20, 50, 200]:
        ma_col = f'sma_{length}'
        if ma_col in df.columns:
            df[f'dist_sma_{length}'] = (df['close'] / df[ma_col]) - 1
            df[f'ma_slope_{length}'] = df[ma_col].pct_change(5)

    # 4. Bollinger Specifics
    bbl_cols = [c for c in df.columns if c.startswith('bbl')]
    bbm_cols = [c for c in df.columns if c.startswith('bbm')]
    bbu_cols = [c for c in df.columns if c.startswith('bbu')]

    if bbl_cols and bbm_cols and bbu_cols:
        df['bb_width'] = (df[bbu_cols[0]] - df[bbl_cols[0]]) / df[bbm_cols[0]]
        df['bb_pct'] = (df['close'] - df[bbl_cols[0]]) / (df[bbu_cols[0]] - df[bbl_cols[0]])
        df['bb_width_trend'] = df['bb_width'].pct_change(5)

    # 5. Price Structure & Volatility
    atr_cols = [c for c in df.columns if c.startswith('atr_14')]
    if atr_cols:
        df['atr_pct'] = df[atr_cols[0]] / df['close']

    df['rolling_std_20'] = df['close'].pct_change().rolling(20).std()
    # Normalize volatility
    avg_vol = df['rolling_std_20'].rolling(252).mean()
    df['volatility_regime'] = df['rolling_std_20'] / avg_vol

    # 6. Returns
    for h in [1, 3, 5, 10, 20]:
        df[f'return_{h}d'] = df['close'].pct_change(h)

    # 7. High/Low Dynamics
    df['dist_20d_high'] = df['close'] / df['high'].rolling(20).max() - 1
    df['dist_20d_low'] = df['close'] / df['low'].rolling(20).min() - 1

    range_high_low = (df['high'] - df['low'])
    df['high_low_range_pct'] = range_high_low / df['close']

    return df
