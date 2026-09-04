import pandas as pd
import numpy as np

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Volume MAs
    df['volume_sma_5'] = df['volume'].rolling(5).mean()
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_sma_50'] = df['volume'].rolling(50).mean()

    # Relative Volume
    df['relative_volume'] = df['volume'] / df['volume_sma_20']

    # Volume Momentum
    df['volume_change_1d'] = df['volume'].pct_change()
    df['volume_zscore_20'] = (df['volume'] - df['volume_sma_20']) / df['volume'].rolling(20).std()

    # OBV (On-Balance Volume)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_change_5'] = df['obv'].pct_change(5)

    # Price-Volume Correlation
    df['pv_corr_20'] = df['close'].rolling(20).corr(df['volume'])

    # Volume Breakout
    df['volume_spike'] = (df['relative_volume'] > 2.0).astype(int)

    return df
