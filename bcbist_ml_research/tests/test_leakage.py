import pytest
import pandas as pd
import numpy as np
from app.features.technical import add_technical_features

def test_no_look_ahead_leakage():
    """
    Verifies that technical features at time T do not depend on data at T+1.
    """
    dates = pd.date_range("2023-01-01", periods=100)
    data = {
        'open': np.random.randn(100) + 100,
        'high': np.random.randn(100) + 102,
        'low': np.random.randn(100) + 98,
        'close': np.random.randn(100) + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }
    df = pd.DataFrame(data, index=dates)

    # Calculate features on original data
    features_orig = add_technical_features(df.copy())

    # Modify the last day
    df_mod = df.copy()
    df_mod.iloc[-1] = df_mod.iloc[-1] * 2
    features_mod = add_technical_features(df_mod)

    # Check if features at T-1 are identical
    # We drop the last row as it's expected to be different
    pd.testing.assert_frame_equal(features_orig.iloc[:-1], features_mod.iloc[:-1])
