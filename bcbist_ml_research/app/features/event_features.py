import pandas as pd
import numpy as np
from typing import List

def add_event_features(symbol_df: pd.DataFrame, events_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Transforms event list into time-series features for a specific symbol.
    Strictly follows Point-in-Time rule.
    """
    df = symbol_df.copy()

    # Force naive index
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    elif isinstance(df.index[0], pd.Timestamp) and df.index[0].tzinfo is not None:
        df.index = [d.replace(tzinfo=None) for d in df.index]

    # Standard Features
    event_cols = [
        'event_count_1d', 'event_count_5d', 'event_count_20d',
        'max_event_importance_5d', 'days_since_last_event',
        'positive_event_count_5d', 'negative_event_count_5d',
        'contract_event_count_20d', 'financial_event_count_20d'
    ]
    for col in event_cols:
        df[col] = 0.0
    df['days_since_last_event'] = 999.0

    if events_df is None or events_df.empty:
        return df

    # Filter events for this symbol or global events
    s_events = events_df[(events_df['symbol'] == symbol) | (events_df['symbol'] == "BIST100")].copy()
    if s_events.empty:
        return df

    s_events['timestamp'] = pd.to_datetime(s_events['timestamp']).dt.tz_localize(None)

    trading_dates = df.index.unique()

    for dt in trading_dates:
        # Cutoff: Disclosure after market close (18:15) belongs to the next period's features
        cutoff = dt.replace(hour=18, minute=15, tzinfo=None)

        known_events = s_events[s_events['timestamp'] <= cutoff]

        if not known_events.empty:
            # 1. Day events (on exactly this trading day)
            day_events = known_events[known_events['timestamp'].dt.date == dt.date()]
            df.at[dt, 'event_count_1d'] = float(len(day_events))

            # 2. Rolling windows
            five_day_cutoff = (dt - pd.Timedelta(days=5)).replace(tzinfo=None)
            recent_5d = known_events[known_events['timestamp'] > five_day_cutoff]
            df.at[dt, 'event_count_5d'] = float(len(recent_5d))

            twenty_day_cutoff = (dt - pd.Timedelta(days=20)).replace(tzinfo=None)
            recent_20d = known_events[known_events['timestamp'] > twenty_day_cutoff]
            df.at[dt, 'event_count_20d'] = float(len(recent_20d))

            # 3. Specific Types
            df.at[dt, 'contract_event_count_20d'] = float(len(recent_20d[recent_20d['event_type'] == 'contract']))
            df.at[dt, 'financial_event_count_20d'] = float(len(recent_20d[recent_20d['event_type'] == 'financial_results']))

            # 4. Sentiment (if available)
            if 'sentiment' in recent_5d.columns:
                df.at[dt, 'positive_event_count_5d'] = float(len(recent_5d[recent_5d['sentiment'] > 0.1]))
                df.at[dt, 'negative_event_count_5d'] = float(len(recent_5d[recent_5d['sentiment'] < -0.1]))

            # 5. Importance & Recency
            if not recent_5d.empty:
                df.at[dt, 'max_event_importance_5d'] = float(recent_5d['importance'].max())

            last_event_ts = known_events['timestamp'].max()
            df.at[dt, 'days_since_last_event'] = (dt.replace(tzinfo=None) - last_event_ts).total_seconds() / (24 * 3600)

    return df
