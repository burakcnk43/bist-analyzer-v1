import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DarvasBoxEngine:
    """
    Identifies consolidation structures and Darvas-style boxes.
    Strictly point-in-time (no lookahead).
    """
    def __init__(self, wait_days=3):
        self.wait_days = wait_days

    def add_box_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Rolling Box Boundaries
        # Box Top: Local high that hasn't been exceeded for wait_days
        # For PIT: At day T, we can only confirm a high at T-wait_days
        df['rolling_max'] = df['high'].rolling(self.wait_days + 1).max()

        # Traditional Darvas:
        # 1. New high is made.
        # 2. Next 3 days high is lower than that high.
        # This implementation uses a simpler rolling logic for feature engineering.
        df['box_top'] = df['high'].rolling(20).max()
        df['box_bottom'] = df['low'].rolling(20).min()

        df['box_height_pct'] = (df['box_top'] - df['box_bottom']) / df['box_bottom']
        df['box_mid'] = (df['box_top'] + df['box_bottom']) / 2

        # 2. Distances
        df['dist_box_top'] = (df['close'] / df['box_top']) - 1
        df['dist_box_bottom'] = (df['close'] / df['box_bottom']) - 1
        df['pos_in_box'] = (df['close'] - df['box_bottom']) / (df['box_top'] - df['box_bottom'] + 1e-9)

        # 3. Compression / Tightness
        df['box_tightness'] = df['box_height_pct'].rolling(10).mean()
        df['box_compression'] = df['box_height_pct'] / (df['box_height_pct'].shift(10) + 1e-9)

        # 4. Breakout Signals
        df['is_breakout_close'] = (df['close'] > df['box_top'].shift(1)).astype(int)
        df['is_breakout_wick'] = (df['high'] > df['box_top'].shift(1)).astype(int)
        df['breakout_strength'] = (df['close'] - df['box_top'].shift(1)) / (df['close'].rolling(20).std() + 1e-9)

        # 5. Volume Confirmation
        if 'relative_volume' in df.columns:
            df['breakout_volume_ratio'] = df['relative_volume'] # Already normalized
            df['volume_confirmed_breakout'] = (df['is_breakout_close'] & (df['relative_volume'] > 1.5)).astype(int)

        # 6. Box Age (Consecutive days boundaries haven't changed)
        top_stable = (df['box_top'] == df['box_top'].shift(1)).astype(int)
        df['box_age'] = top_stable.groupby((top_stable == 0).cumsum()).cumcount()

        # 7. Staircase score (Consecutive higher boxes)
        box_top_up = (df['box_top'] > df['box_top'].shift(5)).astype(int)
        df['box_staircase_score'] = box_top_up.rolling(20).sum()

        return df.drop(columns=['rolling_max'])
