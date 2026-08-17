import pandas as pd
import numpy as np
from typing import Dict, List

def analyze_event_impact(global_df: pd.DataFrame, events_df: pd.DataFrame, horizon='target_return_5d') -> pd.DataFrame:
    """
    Analyzes post-event returns for specific event types.
    """
    if events_df.empty or global_df.empty:
        return pd.DataFrame()

    results = []

    for event_type in events_df['event_type'].unique():
        type_events = events_df[events_df['event_type'] == event_type]

        type_returns = []
        for _, row in type_events.iterrows():
            symbol = row['symbol']
            date = pd.to_datetime(row['date'])

            if symbol == "BIST100":
                matches = global_df[global_df.index == date]
                if not matches.empty:
                    type_returns.extend(matches[horizon].dropna().tolist())
            else:
                matches = global_df[(global_df['symbol'] == symbol) & (global_df.index == date)]
                if not matches.empty:
                    type_returns.extend(matches[horizon].dropna().tolist())

        if len(type_returns) >= 5:
            results.append({
                "event_type": event_type,
                "sample_count": len(type_returns),
                "mean_return": np.mean(type_returns),
                "median_return": np.median(type_returns),
                "win_rate": np.mean([1 if r > 0 else 0 for r in type_returns]),
                "volatility": np.std(type_returns)
            })

    return pd.DataFrame(results).sort_values("mean_return", ascending=False)

def analyze_technical_event_interaction(global_df: pd.DataFrame, horizon='target_return_5d') -> pd.DataFrame:
    """
    Analyzes how technical conditions (e.g., RSI oversold) interact with KAP events.
    """
    if 'event_count_5d' not in global_df.columns:
        return pd.DataFrame()

    df = global_df.copy()
    df['has_event'] = (df['event_count_5d'] > 0).astype(int)

    # Define technical buckets
    if 'rsi_14' in df.columns:
        df['rsi_state'] = 'Neutral'
        df.loc[df['rsi_14'] < 30, 'rsi_state'] = 'Oversold'
        df.loc[df['rsi_14'] > 70, 'rsi_state'] = 'Overbought'

        interaction = df.groupby(['rsi_state', 'has_event'])[horizon].agg([
            ('count', 'count'),
            ('mean_return', 'mean'),
            ('win_rate', lambda x: (x > 0).mean())
        ]).reset_index()

        return interaction
    return pd.DataFrame()

def analyze_event_sector_sensitivity(global_df: pd.DataFrame, horizon='target_return_5d') -> pd.DataFrame:
    """
    Rank sectors by how sensitive they are to corporate events.
    """
    if 'event_count_5d' not in global_df.columns or 'sector' not in global_df.columns:
        return pd.DataFrame()

    df = global_df[global_df['event_count_5d'] > 0]
    if df.empty:
        return pd.DataFrame()

    sensitivity = df.groupby('sector')[horizon].agg([
        ('count', 'count'),
        ('mean_return', 'mean'),
        ('median_return', 'median'),
        ('win_rate', lambda x: (x > 0).mean()),
        ('volatility', 'std')
    ]).reset_index().sort_values("mean_return", ascending=False)

    return sensitivity
