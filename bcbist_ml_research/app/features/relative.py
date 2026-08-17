import pandas as pd
import numpy as np

def add_relative_features(global_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates cross-sectional relative features (sector and market wide).
    Should be called inside feature pipeline or training folds.
    """
    df = global_df.copy()

    # Group by Date and Sector
    # We want to know how the stock compares to its peers on that specific day

    # 1. Sector Relative Returns
    # Note: groupby level 0 is 'date', we need 'sector' as well if it's a column
    group_cols = [df.index.name or 'date', 'sector']

    # Avoid look-ahead: Sector mean for the day
    df['sector_mean_return_1d'] = df.groupby(group_cols)['return_1d'].transform('mean')
    df['rel_sector_return_1d'] = df['return_1d'] - df['sector_mean_return_1d']

    # 2. Sector Relative RSI
    if 'rsi_14' in df.columns:
        df['sector_mean_rsi'] = df.groupby(group_cols)['rsi_14'].transform('mean')
        df['rel_sector_rsi'] = df['rsi_14'] - df['sector_mean_rsi']

    # 3. Market Wide Ranking (Z-Scores)
    market_group = [df.index.name or 'date']
    for col in ['rsi_14', 'relative_volume', 'rolling_std_20']:
        if col in df.columns:
            m_mean = df.groupby(market_group)[col].transform('mean')
            m_std = df.groupby(market_group)[col].transform('std')
            df[f'market_z_{col}'] = (df[col] - m_mean) / m_std

    return df
