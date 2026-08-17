import pytest
import pandas as pd
import numpy as np
from app.labels.target_builder import add_targets

def test_target_alignment():
    """
    Verifies that target_return_1d on day T is indeed (close[T+1] / close[T]) - 1.
    """
    dates = pd.date_range("2023-01-01", periods=5)
    df = pd.DataFrame({
        'close': [100, 110, 105, 115, 120],
        'low': [99, 109, 104, 114, 119],
        'high': [101, 111, 106, 116, 121]
    }, index=dates)

    df_with_targets = add_targets(df, horizons=[1])

    # On day 0 (index 0), target_return_1d should be (110/100)-1 = 0.1
    assert df_with_targets['target_return_1d'].iloc[0] == pytest.approx(0.1)

    # On day 3, target_return_1d should be (120/115)-1
    assert df_with_targets['target_return_1d'].iloc[3] == pytest.approx(120/115 - 1)

    # Last day should have NaN target
    assert np.isnan(df_with_targets['target_return_1d'].iloc[-1])
