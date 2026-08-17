import pandas as pd
import numpy as np

def detect_market_regime(bist_df: pd.DataFrame) -> pd.Series:
    """
    Detects market regime based on BIST100 index behavior.
    """
    df = bist_df.copy()

    # Use returns and volatility
    df['return_20d'] = df['close'].pct_change(20)
    df['vol_20d'] = df['close'].pct_change().rolling(20).std()

    regimes = pd.Series(index=df.index, data='SIDEWAYS')

    # Simple logic
    # Bull: positive 20d return and moderate vol
    bull_mask = (df['return_20d'] > 0.05) & (df['vol_20d'] < df['vol_20d'].median() * 1.5)
    regimes[bull_mask] = 'BULL'

    # Bear: negative 20d return
    bear_mask = (df['return_20d'] < -0.05)
    regimes[bear_mask] = 'BEAR'

    # Volatile: very high volatility regardless of direction
    vol_mask = (df['vol_20d'] > df['vol_20d'].quantile(0.9))
    regimes[vol_mask] = 'HIGH_VOLATILITY'

    return regimes
