import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class DarvasQualityScorer(DarvasConditionalExpert):
    """
    Advanced quality scorer that includes candle-level analysis.
    Differentiates 'Momentum Breaks' from 'Exhaustion Gaps'.
    """
    def _calculate_candle_quality(self, X: pd.DataFrame) -> pd.Series:
        body = (X['close'] - X['open']).abs()
        range_ = X['high'] - X['low'] + 1e-9
        return body / range_

    def train_with_quality(self, X: pd.DataFrame, y_success: pd.Series):
        df = X.copy()
        df['candle_quality'] = self._calculate_candle_quality(df)
        # Update feature list to include quality
        self.feature_cols.append('candle_quality')
        self.train(df, y_success)
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        self.feature_cols = []

    def train(self, X: pd.DataFrame, y_success: pd.Series):
        """
        Only trains on stocks that are CURRENTLY in a breakout state.
        Learns to distinguish between TRUE and FALSE breakouts.
        """
        # Handle case-insensitive column names
        cols = {c.lower(): c for c in X.columns}
        brk_close = cols.get('is_breakout_close')
        brk_wick = cols.get('is_breakout_wick')

        if not brk_close or not brk_wick:
            logger.warning("Breakout indicators missing for DarvasConditionalExpert training.")
            return

        is_breakout = (X[brk_close] > 0) | (X[brk_wick] > 0)
        X_train = X[is_breakout].fillna(0)
        y_train = y_success[is_breakout].astype(int)

        if len(X_train) < 50:
            logger.warning("Insufficient breakout samples for DarvasConditionalExpert.")
            return

        # Focus on structural and confirmation features
        feat_candidates = [
            'box_height_pct', 'breakout_volume_ratio', 'breakout_strength',
            'box_age', 'box_staircase_score', 'rsi_14', 'relative_volume',
            'market_z_rsi_14', 'pct_box_breakout_up'
        ]
        self.feature_cols = [cols[f] for f in feat_candidates if f in cols]

        logger.info(f"Training DarvasConditionalExpert on {len(X_train)} breakouts...")
        self.model.fit(X_train[self.feature_cols], y_train)

    def predict_success_prob(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns P(BreakoutSuccess | Context).
        If not in breakout, returns 0.5 (neutral).
        """
        if not self.feature_cols:
            return np.full(len(X), 0.5)

        cols = {c.lower(): c for c in X.columns}
        brk_close = cols.get('is_breakout_close')
        brk_wick = cols.get('is_breakout_wick')

        probs = np.full(len(X), 0.5)
        if not brk_close: return probs

        is_breakout = (X[brk_close] > 0)
        if brk_wick: is_breakout |= (X[brk_wick] > 0)

        if is_breakout.any():
            X_clean = X[is_breakout][self.feature_cols].fillna(0)
            probs[is_breakout] = self.model.predict_proba(X_clean)[:, 1]

        return probs
