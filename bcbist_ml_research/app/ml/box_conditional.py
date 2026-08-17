import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import joblib
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class DarvasConditionalExpert:
    """
    A specialist that evaluates Box Breakouts conditional on context.
    """
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        self.feature_cols = []

    def train(self, X: pd.DataFrame, y_success: pd.Series):
        cols = {c.lower(): c for c in X.columns}
        brk_close = cols.get('is_breakout_close')
        brk_wick = cols.get('is_breakout_wick')
        if not brk_close: return

        is_breakout = (X[brk_close] > 0)
        if brk_wick: is_breakout |= (X[brk_wick] > 0)

        X_train = X[is_breakout].fillna(0)
        y_train = y_success[is_breakout].astype(int)
        if len(X_train) < 20: return

        feat_candidates = [
            'box_height_pct', 'breakout_volume_ratio', 'breakout_strength',
            'box_age', 'box_staircase_score', 'rsi_14', 'relative_volume',
            'market_z_rsi_14', 'pct_box_breakout_up', 'candle_quality'
        ]
        self.feature_cols = [cols[f] for f in feat_candidates if f in cols]
        self.model.fit(X_train[self.feature_cols], y_train)

    def predict_success_prob(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.full(len(X), 0.5)
        cols = {c.lower(): c for c in X.columns}
        brk_close = cols.get('is_breakout_close')
        if not brk_close: return probs

        is_breakout = (X[brk_close] > 0)
        if is_breakout.any():
            X_clean = X[is_breakout].reindex(columns=self.feature_cols, fill_value=0)
            probs[is_breakout] = self.model.predict_proba(X_clean)[:, 1]
        return probs

    def save(self, path: Path):
        joblib.dump({'model': self.model, 'feature_cols': self.feature_cols}, path)

    @classmethod
    def load(cls, path: Path):
        data = joblib.load(path)
        obj = cls()
        obj.model = data['model']
        obj.feature_cols = data['feature_cols']
        return obj

class DarvasQualityScorer(DarvasConditionalExpert):
    """
    Advanced quality scorer that includes candle-level analysis.
    """
    def _calculate_candle_quality(self, X: pd.DataFrame) -> pd.Series:
        body = (X['close'] - X['open']).abs()
        range_ = X['high'] - X['low'] + 1e-9
        return body / range_

    def train_with_quality(self, X: pd.DataFrame, y_success: pd.Series):
        df = X.copy()
        df['candle_quality'] = self._calculate_candle_quality(df)
        self.train(df, y_success)
