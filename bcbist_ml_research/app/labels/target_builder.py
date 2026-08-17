import pandas as pd
import numpy as np

def add_targets(df: pd.DataFrame, horizons=[1, 3, 5, 10, 20]) -> pd.DataFrame:
    """
    Builds future-looking targets.
    Forward returns, classification labels, and risk metrics.
    """
    df = df.copy()

    for h in horizons:
        # Regression Target: Forward Return (Raw Return)
        df[f'target_return_{h}d'] = df['close'].shift(-h) / df['close'] - 1

        # Classification Target: Directional
        df[f'target_up_{h}d'] = (df[f'target_return_{h}d'] > 0).astype(int)

        # Threshold Targets
        df[f'target_gt_2pct_{h}d'] = (df[f'target_return_{h}d'] > 0.02).astype(int)
        df[f'target_gt_5pct_{h}d'] = (df[f'target_return_{h}d'] > 0.05).astype(int)

    # Risk Targets
    for h in [5, 20]:
        df[f'target_max_loss_{h}d'] = df['low'].shift(-h).rolling(h).min() / df['close'] - 1
        df[f'target_max_gain_{h}d'] = df['high'].shift(-h).rolling(h).max() / df['close'] - 1

    return df

def add_cross_sectional_targets(global_df: pd.DataFrame, horizons=[1, 3, 5, 10, 20]) -> pd.DataFrame:
    """
    Adds ranking targets (relative to other stocks in the same universe on same date).
    """
    df = global_df.copy()

    # Get Market Return for excess return calculation
    # We use macro_bist100_return if available, or calculate it from the universe mean
    if 'macro_bist100_return' in df.columns:
        mkt_ret_1d = df['macro_bist100_return']
    else:
        mkt_ret_1d = df.groupby(level=0)['return_1d'].transform('mean')

    for h in horizons:
        target_ret = f'target_return_{h}d'
        if target_ret not in df.columns: continue

        # 1. Excess Return vs Market
        # We need the h-day market return.
        # For simplicity, we use the average return of the universe for that period.
        mkt_ret_hd = df.groupby(level=0)[target_ret].transform('mean')
        df[f'target_excess_mkt_{h}d'] = df[target_ret] - mkt_ret_hd

        # 2. Excess Return vs Sector
        if 'sector' in df.columns or 'sector_col' in df.columns:
            sec_col = 'sector_col' if 'sector_col' in df.columns else 'sector'
            sec_ret_hd = df.groupby([df.index.name or 'date', sec_col])[target_ret].transform('mean')
            df[f'target_excess_sec_{h}d'] = df[target_ret] - sec_ret_hd

        # 3. Percentile Rank
        df[f'target_rank_{h}d'] = df.groupby(level=0)[target_ret].rank(pct=True)

        # 4. Outperformer Binary
        df[f'target_outperform_{h}d'] = (df[f'target_rank_{h}d'] > 0.5).astype(int)

        # 5. Ranking Label (Discrete for Rankers)
        # 0: bottom 20%, 1: 20-40, 2: 40-60, 3: 60-80, 4: top 20%
        df[f'target_label_{h}d'] = pd.cut(df[f'target_rank_{h}d'],
                                        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                        labels=[0, 1, 2, 3, 4],
                                        include_lowest=True).astype(float).fillna(0).astype(int)

    return df
