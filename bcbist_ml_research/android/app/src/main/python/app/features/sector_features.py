import pandas as pd
import numpy as np
from typing import Dict

def add_sector_features(symbol_df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds sector-relative features.
    sector_df should contain aggregated metrics for that sector.
    """
    df = symbol_df.copy()

    # Ensure indices align (dates)
    common_dates = df.index.intersection(sector_df.index)
    df = df.loc[common_dates]
    sector_df = sector_df.loc[common_dates]

    # Relative Returns
    df['sector_return_1d'] = sector_df['return_1d']
    df['sector_return_5d'] = sector_df['return_5d']

    df['relative_return_1d'] = df['return_1d'] - df['sector_return_1d']
    df['relative_return_5d'] = df['return_5d'] - df['sector_return_5d']

    # Relative Volatility
    if 'rolling_std_20' in sector_df.columns:
        df['sector_volatility_20'] = sector_df['rolling_std_20']
        df['relative_volatility_20'] = df['rolling_std_20'] - df['sector_volatility_20']

    # Sector Momentum
    if 'rsi_14' in sector_df.columns:
        df['sector_rsi_14'] = sector_df['rsi_14']
        df['relative_rsi_14'] = df['rsi_14'] - df['sector_rsi_14']

    return df

def calculate_sector_aggregates(all_stocks_data: Dict[str, pd.DataFrame], sector_mapping: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Calculates aggregate metrics for each sector.
    """
    sectors = set(sector_mapping.values())
    sector_aggregates = {}

    for sector in sectors:
        sector_stocks = [s for s, sec in sector_mapping.items() if sec == sector]
        sector_dfs = [all_stocks_data[s] for s in sector_stocks if s in all_stocks_data]

        if not sector_dfs:
            continue

        # Combine all stocks in sector and average their returns
        # This is a simple equal-weighted sector index
        combined_returns = pd.concat([df['return_1d'] for df in sector_dfs], axis=1).mean(axis=1)

        sector_agg = pd.DataFrame(index=combined_returns.index)
        sector_agg['return_1d'] = combined_returns
        sector_agg['return_5d'] = sector_agg['return_1d'].rolling(5).apply(lambda x: (1 + x).prod() - 1)

        # Average RSI and Volatility
        if 'rsi_14' in sector_dfs[0].columns:
            sector_agg['rsi_14'] = pd.concat([df['rsi_14'] for df in sector_dfs], axis=1).mean(axis=1)
        if 'rolling_std_20' in sector_dfs[0].columns:
            sector_agg['rolling_std_20'] = pd.concat([df['rolling_std_20'] for df in sector_dfs], axis=1).mean(axis=1)

        sector_aggregates[sector] = sector_agg

    return sector_aggregates
