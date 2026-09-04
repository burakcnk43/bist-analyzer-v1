import pandas as pd
import numpy as np
import logging
from app.features.boxes import DarvasBoxEngine

logger = logging.getLogger(__name__)

class MarketBreadthEngine:
    def __init__(self):
        self.box_engine = DarvasBoxEngine()

    def calculate_breadth(self, universe_data: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a concatenated DataFrame of multiple stocks and returns daily breadth metrics.

        Vectorized calculation for:
        - pct_above_sma20, pct_above_sma50, pct_above_sma200
        - pct_rsi_above_50, pct_rsi_below_30
        - advancing_declining_ratio
        - pct_new_20d_high, pct_new_20d_low
        - pct_box_breakout_up, pct_box_breakout_down (using DarvasBoxEngine)
        """
        df = universe_data.copy()

        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            df.index = pd.to_datetime(df.index)
            df = df.reset_index().rename(columns={'index': 'date'})

        # 1. Per-symbol indicators (if not already present)
        df = df.sort_values(['symbol', 'date'])

        # Check if we need to calculate indicators
        needed_cols = [
            'is_breakout_close', 'is_breakout_down', 'is_advancing',
            'is_new_high_20d', 'is_new_low_20d', 'is_above_sma20',
            'is_pos_1d', 'is_pos_3d', 'is_pos_5d', 'is_macd_pos',
            'is_vol_exp', 'is_rel_strength_pos'
        ]

        if not all(col in df.columns for col in needed_cols):
            logger.info("Some breadth indicators missing. Calculating per-symbol...")
            df = df.groupby('symbol', group_keys=False).apply(self.prepare_indicators)

        # 2. Daily aggregation
        breadth = df.groupby('date').agg({
            'is_above_sma20': 'mean',
            'is_above_sma50': 'mean',
            'is_above_sma200': 'mean',
            'is_rsi_above_50': 'mean',
            'is_rsi_below_30': 'mean',
            'is_new_high_20d': 'mean',
            'is_new_low_20d': 'mean',
            'is_breakout_close': 'mean',
            'is_breakout_down': 'mean',
            'is_pos_1d': 'mean',
            'is_pos_3d': 'mean',
            'is_pos_5d': 'mean',
            'is_macd_pos': 'mean',
            'is_vol_exp': 'mean',
            'is_rel_strength_pos': 'mean',
            'is_advancing': 'sum',
            'is_declining': 'sum'
        })

        # Rename to target metrics
        breadth.rename(columns={
            'is_above_sma20': 'pct_above_sma20',
            'is_above_sma50': 'pct_above_sma50',
            'is_above_sma200': 'pct_above_sma200',
            'is_rsi_above_50': 'pct_rsi_above_50',
            'is_rsi_below_30': 'pct_rsi_below_30',
            'is_new_high_20d': 'pct_new_20d_high',
            'is_new_low_20d': 'pct_new_20d_low',
            'is_breakout_close': 'pct_box_breakout_up',
            'is_breakout_down': 'pct_box_breakout_down',
            'is_pos_1d': 'pct_stocks_positive_1d',
            'is_pos_3d': 'pct_stocks_positive_3d',
            'is_pos_5d': 'pct_stocks_positive_5d',
            'is_macd_pos': 'pct_stocks_macd_positive',
            'is_vol_exp': 'pct_stocks_volume_expansion',
            'is_rel_strength_pos': 'pct_stocks_relative_strength_positive'
        }, inplace=True)

        breadth['advancing_declining_ratio'] = breadth['is_advancing'] / (breadth['is_declining'] + 1e-9)

        # 3. Breadth Dynamics (Momentum, Acceleration)
        # Using pct_above_sma50 as the primary breadth proxy
        base_breadth = breadth['pct_above_sma50']
        breadth['breadth_momentum'] = base_breadth.diff(5)
        breadth['breadth_acceleration'] = breadth['breadth_momentum'].diff(5)

        # Divergence: Market return vs Breadth change
        # (Simplified: if market is up but breadth is down -> divergence)
        # We'd need market return here, but we can compute it if 'close' is averaged
        # or just stick to breadth-only metrics for now.

        breadth['breadth_extreme'] = ((base_breadth > 0.8) | (base_breadth < 0.2)).astype(int)

        # Clean up counts and handle NaNs
        res = breadth.drop(columns=['is_advancing', 'is_declining']).fillna(0)
        return res

    def prepare_indicators(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates per-symbol indicators needed for breadth.
        """
        group = group.copy().sort_values('date')

        # Box breakouts
        if 'is_breakout_close' not in group.columns:
            group = self.box_engine.add_box_features(group)
        if 'is_breakout_down' not in group.columns:
            if 'box_bottom' in group.columns:
                group['is_breakout_down'] = (group['close'] < group['box_bottom'].shift(1)).astype(int)
            else:
                group['is_breakout_down'] = 0

        # Returns
        if 'return_1d' not in group.columns:
            group['return_1d'] = group['close'].pct_change()

        group['is_pos_1d'] = (group['return_1d'] > 0).astype(int)
        group['is_pos_3d'] = (group['close'] > group['close'].shift(3)).astype(int)
        group['is_pos_5d'] = (group['close'] > group['close'].shift(5)).astype(int)

        # Advancing/Declining
        group['prev_close'] = group['close'].shift(1)
        group['is_advancing'] = (group['close'] > group['prev_close']).astype(int)
        group['is_declining'] = (group['close'] < group['prev_close']).astype(int)

        # New Highs/Lows (20d)
        group['is_new_high_20d'] = (group['close'] >= group['high'].rolling(20).max()).astype(int)
        group['is_new_low_20d'] = (group['close'] <= group['low'].rolling(20).min()).astype(int)

        # MA & RSI conditions
        group['is_above_sma20'] = (group['close'] > group['sma_20']).astype(int)
        group['is_above_sma50'] = (group['close'] > group['sma_50']).astype(int)
        group['is_above_sma200'] = (group['close'] > group['sma_200']).astype(int)
        group['is_rsi_above_50'] = (group['rsi_14'] > 50).astype(int)
        group['is_rsi_below_30'] = (group['rsi_14'] < 30).astype(int)

        # MACD
        if 'macd_12_26_9' in group.columns:
            group['is_macd_pos'] = (group['macd_12_26_9'] > 0).astype(int)
        else:
            group['is_macd_pos'] = 0

        # Volume expansion
        if 'volume' in group.columns:
            group['vol_sma_20'] = group['volume'].rolling(20).mean()
            group['is_vol_exp'] = (group['volume'] > 1.5 * group['vol_sma_20']).astype(int)
        else:
            group['is_vol_exp'] = 0

        # Relative strength (Simplified: stock ret > 0)
        group['is_rel_strength_pos'] = group['is_pos_1d'] # Placeholder for actual RS calculation in group context

        return group
