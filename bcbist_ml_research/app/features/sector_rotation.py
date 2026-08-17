import pandas as pd
import numpy as np
from typing import Optional

class SectorRotationEngine:
    """
    Engine for calculating sector-level rotation metrics and rankings.
    """

    def calculate_sector_stats(self, universe_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute daily aggregate metrics per sector.

        Args:
            universe_data: DataFrame containing stock-level features including:
                          'date', 'sector', 'return_1d', 'return_5d',
                          'relative_volume', 'dist_20d_high'

        Returns:
            DataFrame with sector-level statistics and rankings.
        """
        if universe_data.empty:
            return pd.DataFrame()

        # Ensure date is datetime
        df = universe_data.copy()
        df['date'] = pd.to_datetime(df['date'])

        # 1. Calculate market return daily as the baseline
        market_returns = df.groupby('date')['return_1d'].transform('mean')
        df['market_return_1d'] = market_returns

        # 2. Define box breakout density (stocks near 20-day highs)
        # Using 0.5% threshold for "near high"
        df['is_breakout'] = (df['dist_20d_high'] <= 0.005).astype(int)

        # 3. Aggregate metrics by sector and date
        sector_stats = df.groupby(['date', 'sector']).agg(
            sector_return_1d=('return_1d', 'mean'),
            sector_momentum_5d=('return_5d', 'mean'),
            count_positive=('return_1d', lambda x: (x > 0).sum()),
            count_total=('return_1d', 'count'),
            sector_volume_ratio=('relative_volume', 'mean'),
            count_breakouts=('is_breakout', 'sum'),
            sector_rsi=('rsi_14', 'mean'),
            sector_volatility=('rolling_std_20', 'mean'),
            market_return_1d=('market_return_1d', 'first')
        ).reset_index()

        # 4. Compute Derived Metrics
        # sector_relative_strength (vs Market)
        sector_stats['sector_relative_strength'] = sector_stats['sector_return_1d'] - sector_stats['market_return_1d']

        # sector_breadth (% stocks positive in sector)
        sector_stats['sector_breadth'] = sector_stats['count_positive'] / sector_stats['count_total']

        # sector_box_breakout_density
        sector_stats['sector_box_breakout_density'] = sector_stats['count_breakouts'] / sector_stats['count_total']

        # Sector Acceleration
        sector_stats['sector_momentum_change'] = sector_stats.groupby('sector')['sector_relative_strength'].diff(1)
        sector_stats['sector_acceleration'] = sector_stats.groupby('sector')['sector_momentum_change'].diff(1)

        # 5. Rank sectors daily by relative_strength
        # Higher relative strength = lower rank number (1 is best)
        sector_stats['sector_rank'] = sector_stats.groupby('date')['sector_relative_strength'].rank(ascending=False, method='min')

        # Leadership Detection
        # A sector is a "Leader" if it is top 3 in rank and has positive relative strength
        sector_stats['is_leader'] = ((sector_stats['sector_rank'] <= 3) & (sector_stats['sector_relative_strength'] > 0)).astype(int)

        # Sort for readability
        sector_stats = sector_stats.sort_values(['date', 'sector_rank'])

        # Final column selection
        cols_to_keep = [
            'date', 'sector', 'sector_relative_strength',
            'sector_momentum_5d', 'sector_breadth',
            'sector_volume_ratio', 'sector_box_breakout_density',
            'sector_rsi', 'sector_volatility',
            'sector_acceleration', 'sector_rank', 'is_leader'
        ]

        return sector_stats[cols_to_keep]

def get_sector_mapping(universe_config: list) -> dict:
    """Creates a symbol to sector mapping from config."""
    return {item['symbol']: item['sector'] for item in universe_config if 'symbol' in item and 'sector' in item}
