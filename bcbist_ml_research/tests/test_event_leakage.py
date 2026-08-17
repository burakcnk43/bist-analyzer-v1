import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.features.event_features import add_event_features

def test_event_point_in_time_rule():
    """
    Ensures an event happening on day T+1 is NOT visible in day T features.
    """
    # Create a simple 5-day price history
    dates = pd.date_range("2024-03-01", periods=5)
    df = pd.DataFrame(index=dates)
    df['dummy'] = 0

    # Create an event on 2024-03-03 10:00
    events = pd.DataFrame([
        {
            "symbol": "THYAO.IS",
            "timestamp": pd.to_datetime("2024-03-03 10:00:00"),
            "event_type": "contract",
            "importance": 5,
            "sentiment": 0.8
        }
    ])

    # Generate features
    features = add_event_features(df, events, "THYAO.IS")

    # On 2024-03-01 and 2024-03-02, event_count_1d should be 0
    assert features.at[pd.to_datetime("2024-03-01"), 'event_count_1d'] == 0
    assert features.at[pd.to_datetime("2024-03-02"), 'event_count_1d'] == 0

    # On 2024-03-03, event_count_1d should be 1
    assert features.at[pd.to_datetime("2024-03-03"), 'event_count_1d'] == 1

    # Check specific type counters
    assert features.at[pd.to_datetime("2024-03-03"), 'contract_event_count_20d'] == 1

    # On 2024-03-04, event_count_1d should be 0 (if 1d counts events on same day only)
    assert features.at[pd.to_datetime("2024-03-04"), 'event_count_1d'] == 0
    assert features.at[pd.to_datetime("2024-03-04"), 'event_count_5d'] == 1
