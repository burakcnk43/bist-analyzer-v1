import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from app.features.event_features import add_event_features

def test_kap_disclosure_timing():
    """
    Verifies that a KAP disclosure at 18:30 is NOT available for same-day features.
    """
    dates = pd.date_range("2024-05-15", periods=2)
    df = pd.DataFrame(index=dates)
    df['close'] = [100, 105]

    # Event at 18:30 (Market Closed)
    events = pd.DataFrame([
        {
            "symbol": "THYAO.IS",
            "timestamp": pd.to_datetime("2024-05-15 18:30:00"),
            "event_type": "contract",
            "importance": 5
        }
    ])

    features = add_event_features(df, events, "THYAO.IS")

    # On May 15, the event should NOT be counted in 1d features if cutoff is 18:15
    assert features.at[pd.to_datetime("2024-05-15"), 'event_count_1d'] == 0

    # On May 16, it should be visible in 5d features (lookback)
    assert features.at[pd.to_datetime("2024-05-16"), 'event_count_5d'] == 1

def test_market_open_disclosure():
    """
    Verifies that a KAP disclosure at 10:00 AM IS available for same-day features.
    """
    dates = pd.date_range("2024-05-15", periods=1)
    df = pd.DataFrame(index=dates)

    events = pd.DataFrame([
        {
            "symbol": "THYAO.IS",
            "timestamp": pd.to_datetime("2024-05-15 10:00:00"),
            "event_type": "contract",
            "importance": 5
        }
    ])

    features = add_event_features(df, events, "THYAO.IS")
    assert features.at[pd.to_datetime("2024-05-15"), 'event_count_1d'] == 1
